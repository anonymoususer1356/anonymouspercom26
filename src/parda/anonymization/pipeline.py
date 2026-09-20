"""The readable one-pass windowed anonymisation pipeline."""

from dataclasses import dataclass, field

from Prompts.prompt_generator import build_anonymiser_rag_prompt, build_windowed_adversary_prompt
from rag_store import EditMemory, Profile
from transcript_utils import apply_edits, extract_edits, format_transcript
from validation import ProfileValidator, extract_json_result
from windowing import split_windows


@dataclass
class PipelineResult:
    original: str
    anonymised: str
    profile: list[dict]
    windows: list[dict]
    exchanges: dict
    errors: list[str] = field(default_factory=list)


class WindowProcessor:
    # Keep adversary findings, retrieved edits, and model exchanges across windows.
    def __init__(self, adversary, anonymiser, memory: EditMemory, *,
                 passes_per_window: int, temperature: float,
                 top_p: float, max_tokens: int):
        self.adversary = adversary
        self.anonymiser = anonymiser
        self.memory = memory
        self.passes_per_window = passes_per_window
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.profile = Profile(validator=ProfileValidator())
        self.errors = []
        self.exchanges = {}
        self.trace = []

    # Process one already-tokenized transcript window without resetting shared state.
    def process(self, window_index: int, line_numbers: list[int], text: str) -> str:
        current = text
        window_trace = {"window": window_index, "lines": line_numbers, "passes": []}
        for pass_number in range(self.passes_per_window):
            tag = f"w{window_index}p{pass_number}"
            system, user = build_windowed_adversary_prompt(current, self.profile.render())
            adversary_exchange = self.adversary.chat(
                system,
                user,
                self.temperature,
                self.top_p,
                self.max_tokens,
                "adversary_" + tag,
            )
            self.exchanges["adversary_" + tag] = adversary_exchange
            parse = (
                extract_json_result(adversary_exchange.content)
                if not adversary_exchange.error
                else None
            )
            if adversary_exchange.error:
                self.errors.append("adversary_" + tag + ": " + adversary_exchange.error)
                new_facts = []
                validation = {
                    "status": "backend_error",
                    "accepted": 0,
                    "rejected": 0,
                    "reasons": [],
                }
            elif parse.value is None:
                new_facts = []
                validation = {
                    "status": parse.status,
                    "accepted": 0,
                    "rejected": 1,
                    "reasons": [parse.detail or "malformed JSON"],
                }
                validation["json_status"] = parse.status
                if parse.detail:
                    validation["json_detail"] = parse.detail
                self.errors.extend(validation["reasons"])
            else:
                new_facts = self.profile.add(
                    parse.value,
                    source="adversary_" + tag,
                    line_numbers=set(line_numbers),
                )
                validation = self.profile.last_report.to_dict()
                validation["json_status"] = parse.status
                if parse.detail:
                    validation["json_detail"] = parse.detail
                self.errors.extend(validation["reasons"])

            retrieved = self.memory.query(current.splitlines())
            system, user = build_anonymiser_rag_prompt(
                current,
                self.profile.render(),
                retrieved,
            )
            anonymiser_exchange = self.anonymiser.chat(
                system,
                user,
                self.temperature,
                self.top_p,
                self.max_tokens,
                "anonymiser_" + tag,
            )
            self.exchanges["anonymiser_" + tag] = anonymiser_exchange
            if anonymiser_exchange.error:
                self.errors.append(
                    "anonymiser_" + tag + ": " + anonymiser_exchange.error
                )

            unique_edits = {}
            for number, replacement in extract_edits(anonymiser_exchange.content):
                unique_edits[number] = replacement
            proposed = [
                (number, replacement)
                for number, replacement in unique_edits.items()
                if number in line_numbers
            ]
            ignored = [
                number for number in unique_edits
                if number not in line_numbers
            ]
            self.errors.extend(
                f"anonymiser_{tag}: line {number} is outside its window"
                for number in ignored
            )
            before = {
                int(line.split(":", 1)[0]): line
                for line in current.splitlines()
            }
            current, applied = apply_edits(current, proposed, self.errors)
            landed = [
                {
                    "before": before[number],
                    "after": f"{number}: {replacement}",
                }
                for number, replacement in applied
                if number in before
                and replacement
            ]
            self.memory.add(landed)
            window_trace["passes"].append({
                "pass": pass_number,
                "new_inferences": new_facts,
                "profile_validation": validation,
                "retrieved": retrieved,
                "edits": landed,
            })
        self.trace.append(window_trace)
        return current


def run_streaming(turns, adversary, anonymiser, memory: EditMemory, tokenizer, *,
                  window_lines: int, passes_per_window: int, temperature: float,
                  top_p: float, max_tokens: int) -> PipelineResult:
    # Process each window once. Prior adversary findings and edits carry forward.
    original = format_transcript(turns)
    memory.remember(original.splitlines())
    processor = WindowProcessor(
        adversary,
        anonymiser,
        memory,
        passes_per_window=passes_per_window,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    finished = []
    token_budget = round(window_lines * 21.9625)
    for window in split_windows(original, tokenizer, token_budget):
        finished.append(
            processor.process(window.index, window.line_numbers, window.text)
        )
    return PipelineResult(
        original,
        "\n".join(finished),
        processor.profile.records,
        processor.trace,
        processor.exchanges,
        processor.errors,
    )
