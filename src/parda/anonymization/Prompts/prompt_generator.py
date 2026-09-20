# Load the prompt files and fill their placeholders.

import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent

def _load(stem: str) -> tuple[str, str]:
    #read a file and return a seperated version of it
    text = (PROMPT_DIR / f"{stem}.txt").read_text(encoding="utf-8")
    sections = text.split("----- ")
    if len(sections) < 3:
        raise ValueError(f"{stem}.txt is missing its system/user section markers")
    system = sections[1].split(" -----\n\n", 1)[1].rstrip()
    user = sections[2].split(" -----\n\n", 1)[1].rstrip()
    return system, user

def _render(stem: str, **values: object) -> tuple[str, str]:
    # replace generic placeholder with value
    system, user = _load(stem)
    for name, value in values.items():
        replacement = value.strip() if isinstance(value, str) else json.dumps(
            value, indent=2, ensure_ascii=False
        )
        user = user.replace("{{" + name + "}}", replacement)
    return system, user

# Full length adversary prompt used for evals
def build_adversary_prompt(transcript: str) -> tuple[str, str]:
    return _render("adversary", TRANSCRIPT=transcript)

# Infrence style adversary
def build_windowed_adversary_prompt(window: str, profile: object) -> tuple[str, str]:
    return _render("windowed_adversary", TRANSCRIPT=window, PROFILE=profile)

# Inference style anonymiser
def build_anonymiser_rag_prompt(window: str, profile: object, retrieved_edits: str) -> tuple[str, str]:
    return _render("anonymiser_rag", TRANSCRIPT=window, INFERENCES=profile, RAG=retrieved_edits)

# Utility judge
def build_utility_judge_prompt(original: str, anonymised: str) -> tuple[str, str]:
    return _render("utility_judge", TRANSCRIPT=original, ANONYMISED_TRANSCRIPT=anonymised)

# Privacy judge
def build_privacy_judge_prompt(items: str) -> tuple[str, str]:
    return _render("privacy_judge", ITEMS=items)

# Silver labeller
def build_silver_labeller_prompt(transcript: str) -> tuple[str, str]:
    return _render("silver_labeller", TRANSCRIPT=transcript)
