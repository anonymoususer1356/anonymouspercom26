from dataclasses import dataclass

AVERAGE_TOKENS_PER_LINE = 21.9625 # found emperically across all candor cliffhanger transcripts, using this as a baseline

@dataclass
class Window:
    index: int
    line_numbers: list[int]
    text: str
    tokens: int

def get_tokenizer(name: str = "Qwen/Qwen3.5-2B"):
    from transformers import AutoTokenizer
    # The benchmark must not fetch artifacts or depend on network availability.
    return AutoTokenizer.from_pretrained(name, local_files_only=True)

def lines_to_tokens(lines: int) -> int:
    return max(1, round(lines * AVERAGE_TOKENS_PER_LINE))


def split_windows(transcript: str, tokenizer, token_limit: int) -> list[Window]:
    windows, current_lines, current_numbers, used = [], [], [], 0
    for raw_line in transcript.splitlines():
        parts = raw_line.split(":", 2)
        if len(parts) < 3 or not parts[0].strip().isdigit():
            continue
        token_count = len(tokenizer.encode(parts[2].strip(), add_special_tokens=False))
        if current_lines and used + token_count > token_limit:
            windows.append(Window(len(windows), current_numbers, "\n".join(current_lines), used))
            current_lines, current_numbers, used = [], [], 0
        current_lines.append(raw_line.strip())
        current_numbers.append(int(parts[0]))
        used += token_count
    if current_lines:
        windows.append(Window(len(windows), current_numbers, "\n".join(current_lines), used))
    return windows
