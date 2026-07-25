try:
    from math_verify import parse, verify
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "`math_verify` is required for openr1_math_16k. "
        "Install it with `pip install -e .[math]`."
    ) from e

from lm_eval.tasks.aime.utils import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)


def passthrough_responses(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Keep all sampled responses for the per-sample and average metrics."""
    return resps


def _extract_model_answer(response: str) -> str:
    """Extract the last boxed answer, with AIME's dollar-span fallback."""
    answer = response
    dollar_indices = [index for index, char in enumerate(response) if char == "$"]
    if len(dollar_indices) > 1:
        answer = response[dollar_indices[0] + 1 : dollar_indices[-1]]

    boxed = last_boxed_only_string(response)
    if boxed is not None:
        try:
            answer = remove_boxed(boxed)
        except (AssertionError, IndexError):
            pass
    return answer


def _parse_gold(target: str) -> list:
    """Put raw OpenR1 answers in a LaTeX environment before parsing.

    Math-Verify otherwise treats bare comma-separated answers as decimal
    expressions and cannot reliably detect unanchored LaTeX such as fractions.
    """
    boxed_target = r"\boxed{" + target + "}"
    return parse(boxed_target)


def _math_verify_score(response: str, parsed_gold: list) -> int:
    """Apply the same single-pass Math-Verify check used by MATH-500."""
    if not parsed_gold:
        return 0
    try:
        return 1 if verify(gold=parsed_gold, target=parse(response)) else 0
    except Exception:
        return 0


def _score_response(
    target: str, response: str, parsed_gold: list
) -> dict[str, int]:
    exact = 1 if is_equiv(_extract_model_answer(response), target) else 0
    return {
        "exact_match": exact,
        "math_verify": _math_verify_score(response, parsed_gold),
    }


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    target = str(doc["answer"])
    return _score_response(target, results[0], _parse_gold(target))


def process_results_samples(
    doc: dict, results: list[list[str]]
) -> dict[str, float]:
    """Score all eight responses and average both matching methods."""
    responses = results[0]
    target = str(doc["answer"])
    parsed_gold = _parse_gold(target)
    scores = [
        _score_response(target, response, parsed_gold) for response in responses
    ]

    result = {}
    for index, score in enumerate(scores, start=1):
        result[f"sample@{index}"] = score["exact_match"]
        result[f"mv_sample@{index}"] = score["math_verify"]
    result[f"avg@{len(scores)}"] = sum(
        score["exact_match"] for score in scores
    ) / len(scores)
    result[f"mv_avg@{len(scores)}"] = sum(
        score["math_verify"] for score in scores
    ) / len(scores)
    return result
