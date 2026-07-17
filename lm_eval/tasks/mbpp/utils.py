from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def process_results_samples(doc: dict, results: list[list[str]]) -> dict[str, float]:
    """Return each sample's score and the average over all samples."""
    predictions = results[0]
    reference = "\n".join(doc["test_list"][:3])
    _, execution_results = pass_at_k.compute(
        references=[reference],
        predictions=[predictions],
        k=[1],
    )

    task_results = execution_results.get(0)
    if task_results is None:
        task_results = next(iter(execution_results.values()))
    # evaluate/code_eval returns ``(completion_id, result_dict)`` tuples in
    # current releases, while some older releases returned result dicts.
    # Normalize both shapes so sample@N remains tied to generation order.
    task_results = sorted(
        task_results,
        key=lambda result: result[0]
        if isinstance(result, tuple)
        else result["completion_id"],
    )
    scores = [
        int((result[1] if isinstance(result, tuple) else result)["passed"])
        for result in task_results
    ]

    if len(scores) != len(predictions):
        raise ValueError(
            f"Expected {len(predictions)} execution results, got {len(scores)}"
        )

    metrics = {f"sample@{k}": scores[k - 1] for k in range(1, len(scores) + 1)}
    metrics[f"avg@{len(scores)}"] = sum(scores) / len(scores)
    return metrics


def extract_code_blocks(text: str) -> str:
    """Extract code generated after the task's opening `````python`` prefix.

    ``gen_prefix`` is part of the prompt, so the returned model text starts
    directly with code (usually ``def ...``) and only contains the closing
    fence.  Pretending that the response itself starts with ````` `` makes a
    regex interpret the leading ``def``/``from`` as a fence language tag and
    drops it, turning every otherwise valid completion into invalid Python.
    """
    return text.split("```", 1)[0].strip()


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
