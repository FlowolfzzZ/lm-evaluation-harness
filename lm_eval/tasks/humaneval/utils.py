import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def process_results_samples(doc: dict, results: list[list[str]]) -> dict[str, float]:
    """Return each sample's score and the average over all samples."""
    predictions = results[0]
    reference = f"{doc['test']}\ncheck({doc['entry_point']})"
    _, execution_results = compute_.compute(
        references=[reference],
        predictions=[predictions],
        k=[1],
    )

    task_results = execution_results.get(0)
    if task_results is None:
        task_results = next(iter(execution_results.values()))
    task_results = sorted(task_results, key=lambda result: result["completion_id"])
    scores = [int(result["passed"]) for result in task_results]

    if len(scores) != len(predictions):
        raise ValueError(
            f"Expected {len(predictions)} execution results, got {len(scores)}"
        )

    metrics = {f"sample@{k}": scores[k - 1] for k in range(1, len(scores) + 1)}
    metrics[f"avg@{len(scores)}"] = sum(scores) / len(scores)
    return metrics


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
