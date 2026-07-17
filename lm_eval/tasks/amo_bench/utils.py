from typing import Dict, List

try:
    from math_verify import parse, verify
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "`math_verify` is required for amo_bench. "
        "Install via: pip install lm-eval[math]"
    ) from e

from lm_eval.tasks.aime.utils import (
    is_equiv,
    last_boxed_only_string,
    passthrough_responses,
    remove_boxed,
)

__all__ = ["passthrough_responses", "process_results", "process_results_samples"]


def _normalize_target(doc: dict) -> str:
    """Strip \\boxed{} from target — AMO-Bench stores answers as \\boxed{...}."""
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])
    boxed = last_boxed_only_string(target)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except (AssertionError, IndexError):
            pass
    return target


def _extract_model_answer(response: str) -> str:
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    answer = response if len(indices) <= 1 else response[indices[0] + 1 : indices[-1]]
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        try:
            content = remove_boxed(boxed)
            if content is not None:
                answer = content
        except (AssertionError, IndexError):
            pass
    return answer


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    response = results[0]
    answer = _extract_model_answer(response)
    target = _normalize_target(doc)

    exact = 1 if is_equiv(answer, target) else 0

    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    raw_target = str(doc[answer_key])
    try:
        mv = verify(gold=parse(raw_target), target=parse(response))
        math_verify_score = 1 if mv else 0
    except Exception:
        math_verify_score = 0

    return {"exact_match": exact, "math_verify": math_verify_score}


def process_results_samples(doc: dict, results: List[List[str]]) -> Dict[str, float]:
    responses = results[0]
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    raw_target = str(doc[answer_key])
    target = _normalize_target(doc)

    exact_scores = []
    mv_scores = []
    for response in responses:
        exact_scores.append(1 if is_equiv(_extract_model_answer(response), target) else 0)
        try:
            mv = verify(gold=parse(raw_target), target=parse(response))
            mv_scores.append(1 if mv else 0)
        except Exception:
            mv_scores.append(0)

    result = {}
    for k in range(1, len(exact_scores) + 1):
        result[f"sample@{k}"] = exact_scores[k - 1]
        result[f"mv_sample@{k}"] = mv_scores[k - 1]
    result[f"avg@{len(exact_scores)}"] = sum(exact_scores) / len(exact_scores)
    result[f"mv_avg@{len(mv_scores)}"] = sum(mv_scores) / len(mv_scores)
    return result
