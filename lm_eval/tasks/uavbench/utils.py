import re
from typing import Any


BOXED_RE = re.compile(r"\\boxed\s*\{\s*([A-G])\s*\}", re.IGNORECASE)
FINAL_RE = re.compile(
    r"(?:final\s+answer|answer|答案)\s*(?:is|:|：)?\s*[\(\[]?\s*([A-G])\s*[\)\]]?",
    re.IGNORECASE,
)
LETTER_ONLY_RE = re.compile(r"^\s*[\(\[]?([A-G])[\)\].:：-]?\s*$")
STANDALONE_RE = re.compile(r"(?:^|[^\w])([A-G])(?:[^\w]|$)")


def extract_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    for regex in (BOXED_RE, FINAL_RE, LETTER_ONLY_RE):
        match = regex.search(text)
        if match:
            return match.group(1).upper()

    matches = STANDALONE_RE.findall(text)
    if matches:
        return matches[-1].upper()

    return ""


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = extract_answer(results[0] if results else "")
    target = str(doc["correct_choice"]).strip().upper()

    return {
        "acc": 1.0 if prediction == target else 0.0,
    }
