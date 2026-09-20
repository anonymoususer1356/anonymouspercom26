import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "convert_candor.py"
SPEC = importlib.util.spec_from_file_location("convert_candor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_conversion_preserves_order_and_aliases(tmp_path):
    source = tmp_path / "transcript_cliffhanger.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker", "utterance"])
        writer.writeheader()
        writer.writerows([
            {"speaker": "person-9", "utterance": "Hello."},
            {"speaker": "person-3", "utterance": "Hi there."},
            {"speaker": "person-9", "utterance": "Um, welcome."},
        ])
    output = tmp_path / "conversation.txt"
    assert MODULE.convert_conversation(source, output) == 3
    assert output.read_text(encoding="utf-8") == (
        "1: speaker_a: Hello.\n"
        "2: speaker_b: Hi there.\n"
        "3: speaker_a: Um, welcome.\n"
    )
