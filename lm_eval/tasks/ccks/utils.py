import json
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_DIR = Path("data/ccks2026")
ANSWER_LINE_RE = re.compile(r"答案\s*(?:是|为|应该是|可能是)?\s*[:：]?\s*(.+)")
LEADING_PREFIX_RE = re.compile(
    r"^(?:最终)?(?:短)?答案(?:是|为|应该是|可能是)?[:：，,。\s]*"
)
PUNCTUATION = set(string.punctuation) | set("，。！？；：（）【】《》“”‘’、·—…「」『』￥")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _documents_by_id() -> dict[str, str]:
    docs = _read_json(DATA_DIR / "document.json")
    return {doc["doc_id"]: doc["content"] for doc in docs}


@lru_cache(maxsize=None)
def _knowledge_triples_for_doc(doc_id: str) -> str:
    triples = _read_json(DATA_DIR / "KG" / f"KG_{doc_id}.json")
    return "\n".join(
        f"{triple.get('sub', '')} --[{triple.get('relation', '')}]--> {triple.get('obj', '')}"
        for triple in triples
    )


def process_docs(dataset):
    documents = _documents_by_id()

    def _process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        doc_id = doc["doc_id"]
        doc["document"] = documents[doc_id]
        doc["knowledge_triples"] = _knowledge_triples_for_doc(doc_id)
        doc["answers"] = doc.get("answers", [])
        doc["answer"] = doc.get("answer") or (doc["answers"][0] if doc["answers"] else "")
        return doc

    return dataset.map(_process_doc)


def extract_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    for line in reversed(lines):
        match = ANSWER_LINE_RE.search(line)
        if match:
            return match.group(1).strip()

    return lines[-1] if lines else ""


def normalize_answer(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?|```$", "", text).strip()
    text = LEADING_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+", "", text)
    text = "".join(ch for ch in text if ch not in PUNCTUATION)
    return text.lower()


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    raw_prediction = results[0] if results else ""
    extracted = extract_answer(raw_prediction)
    prediction = normalize_answer(extracted)
    targets = [normalize_answer(answer) for answer in doc.get("answers", [])]
    targets = [target for target in targets if target]

    exact_match = any(prediction == target for target in targets)

    return {"exact_match": 1.0 if exact_match else 0.0}
