import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from typing import Any


eval_logger = logging.getLogger(__name__)

_GENERATED_TESTS_ENV = "CF_TESTS_FOLDER"
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_CPP_TAGS = {"c++", "c++17", "cpp", "cpp17"}
_PYTHON_TAGS = {"py", "python", "python3"}
_FENCED_CODE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```([^\n`]*)\n")
_PYTHON_CODE_START_RE = re.compile(
    r"^(?=(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+\s*\(|"
    r"class\s+\w+|if\s+__name__))",
    re.MULTILINE,
)
_missing_generated_tests_warning_emitted = False
_score_executor: ProcessPoolExecutor | None = None


def passthrough_responses(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Keep all sampled responses so avg@8 can score each one."""
    return resps


def extract_code(response: str, language: str) -> str:
    """Extract a solution while tolerating truncated or omitted code fences.

    Prefer a complete language-matching fence, preserving the original scoring
    behavior.  Long Codeforces reasoning can consume the generation budget
    after opening the final fence, though, so fall back to the contents of an
    unclosed matching fence.  Some instruction-tuned models return raw code;
    recognize an unambiguous program start for those responses as well.
    """
    expected_tags = _PYTHON_TAGS if language == "python" else _CPP_TAGS
    matches = _FENCED_CODE_RE.findall(response)
    tagged_matches = [
        code for tag, code in matches if tag.strip().lower() in expected_tags
    ]
    if tagged_matches:
        return tagged_matches[-1].strip()

    untagged_matches = [code for tag, code in matches if not tag.strip()]
    if untagged_matches:
        return untagged_matches[-1].strip()

    openings = [
        match
        for match in _OPEN_FENCE_RE.finditer(response)
        if match.group(1).strip().lower() in expected_tags
        or not match.group(1).strip()
    ]
    if openings:
        return response[openings[-1].end() :].strip()

    if language == "cpp":
        code_start = response.find("#include")
    else:
        match = _PYTHON_CODE_START_RE.search(response)
        code_start = match.start() if match else -1
    if code_start >= 0:
        return response[code_start:].strip()
    return ""


def process_results(doc: dict, results: list[Any]) -> dict[str, float]:
    """Execute the greedy completion and return pass@1."""
    responses = _unwrap_responses(results)
    if not responses:
        return {"pass@1": 0.0}
    score = _score_responses(doc, responses[:1])[0]
    return {"pass@1": float(score)}


def process_results_samples(doc: dict, results: list[Any]) -> dict[str, float]:
    """Execute all eight samples and return per-sample and average scores."""
    responses = _unwrap_responses(results)
    if len(responses) != 8:
        raise ValueError(f"Expected 8 Codeforces samples, got {len(responses)}")

    scores = _score_responses(doc, responses)
    metrics = {f"sample@{index}": score for index, score in enumerate(scores, 1)}
    metrics["avg@8"] = sum(scores) / 8
    return metrics


def _unwrap_responses(results: list[Any]) -> list[str]:
    if len(results) == 1 and isinstance(results[0], list):
        return results[0]
    return results


def _score_responses(doc: dict, responses: list[str]) -> list[float]:
    test_cases = _get_test_cases(doc)
    language = doc["language"]
    codes = [extract_code(response, language) for response in responses]
    if len(codes) == 1:
        return [float(_evaluate_submission(doc, codes[0], test_cases))]

    # avg@8 candidates are independent.  Judge them in isolated worker
    # processes so compilation, execution and per-test timeouts run in
    # parallel without using preexec_fn from threads.
    executor = _get_score_executor()
    return list(
        executor.map(
            _evaluate_submission_score,
            repeat(doc),
            codes,
            repeat(test_cases),
        )
    )


def _get_score_executor() -> ProcessPoolExecutor:
    global _score_executor
    if _score_executor is None:
        workers = max(1, int(os.environ.get("CF_EVAL_WORKERS", "8")))
        _score_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        )
    return _score_executor


def _evaluate_submission_score(
    doc: dict, code: str, test_cases: list[dict[str, str]]
) -> float:
    return float(_evaluate_submission(doc, code, test_cases))


def _get_test_cases(doc: dict) -> list[dict[str, str]]:
    official_tests = [
        {"input": str(test["input"]), "output": str(test["output"])}
        for test in doc["official_tests"]
    ]
    generated_test_count = int(doc.get("generated_tests") or 0)
    if generated_test_count == 0:
        return official_tests

    tests_folder = os.environ.get(_GENERATED_TESTS_ENV)
    if not tests_folder:
        global _missing_generated_tests_warning_emitted
        if not _missing_generated_tests_warning_emitted:
            eval_logger.warning(
                "%s is not set; Codeforces scores use only inline official tests. "
                "Set it to the downloaded open-r1/codeforces test directory for "
                "full verification.",
                _GENERATED_TESTS_ENV,
            )
            _missing_generated_tests_warning_emitted = True
        return official_tests

    generated_tests = _load_generated_tests(
        tests_folder=tests_folder,
        contest_id=str(doc["contest_id"]),
        problem_id=str(doc["id"]),
    )
    if len(generated_tests) != generated_test_count:
        raise ValueError(
            f"Expected {generated_test_count} generated tests for {doc['id']}, "
            f"found {len(generated_tests)}"
        )
    return official_tests + list(generated_tests)


@lru_cache(maxsize=1024)
def _load_generated_tests(
    tests_folder: str, contest_id: str, problem_id: str
) -> tuple[dict[str, str], ...]:
    root = Path(tests_folder).expanduser()
    filename = f"test_cases_{contest_id}.parquet"
    candidates = (root / "generated_tests" / filename, root / filename)
    test_file = next((path for path in candidates if path.is_file()), None)
    if test_file is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find generated tests for contest {contest_id}; searched {searched}"
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Loading Codeforces generated tests requires pyarrow"
        ) from exc

    table = pq.read_table(
        test_file,
        columns=["input", "output", "test_case_i"],
        filters=[("problem_id", "=", problem_id)],
    )
    rows = sorted(table.to_pylist(), key=lambda row: row["test_case_i"])
    return tuple(
        {"input": str(row["input"]), "output": str(row["output"])} for row in rows
    )


def _evaluate_submission(doc: dict, code: str, test_cases: list[dict[str, str]]) -> int:
    if not code or not test_cases:
        return 0

    language = doc["language"]
    with tempfile.TemporaryDirectory(prefix="lm_eval_codeforces_") as temp_dir:
        work_dir = Path(temp_dir)
        command = _prepare_submission(language, code, work_dir)
        checker = str(doc.get("generated_checker") or "")
        checker_path = work_dir / "checker.py"
        if checker:
            checker_path.write_text(checker, encoding="utf-8")

        time_limit = max(float(doc.get("time_limit") or 1.0), 0.1)
        memory_limit = max(float(doc.get("memory_limit") or 256.0), 1.0)
        input_mode = doc.get("input_mode") or "stdio"
        for test_case in test_cases:
            if not _run_test_case(
                command=command,
                work_dir=work_dir,
                test_case=test_case,
                checker_path=checker_path if checker else None,
                input_mode=input_mode,
                time_limit=time_limit,
                memory_limit=memory_limit,
            ):
                return 0
    return 1


def _prepare_submission(language: str, code: str, work_dir: Path) -> list[str]:
    if language == "python":
        source_path = work_dir / "main.py"
        source_path.write_text(code, encoding="utf-8")
        return [sys.executable, str(source_path)]

    if language == "cpp":
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("The Codeforces C++ task requires g++")
        source_path = work_dir / "main.cpp"
        executable_path = work_dir / "main"
        source_path.write_text(code, encoding="utf-8")
        compile_result = _run_process(
            [
                compiler,
                "-DONLINE_JUDGE",
                "-O2",
                "-std=c++17",
                "-pipe",
                str(source_path),
                "-o",
                str(executable_path),
            ],
            cwd=work_dir,
            timeout=60.0,
        )
        if compile_result != 0:
            return []
        return [str(executable_path)]

    raise ValueError(f"Unsupported Codeforces language: {language!r}")


def _run_test_case(
    command: list[str],
    work_dir: Path,
    test_case: dict[str, str],
    checker_path: Path | None,
    input_mode: str,
    time_limit: float,
    memory_limit: float,
) -> bool:
    if not command:
        return False

    input_text = _unix_newlines(test_case["input"])
    expected_text = _unix_newlines(test_case["output"])
    input_path = work_dir / ("input.txt" if input_mode == "file" else "case_input.txt")
    expected_path = work_dir / "correct_output.txt"
    output_path = work_dir / (
        "output.txt" if input_mode == "file" else "solution_output.txt"
    )
    input_path.write_text(input_text, encoding="utf-8")
    expected_path.write_text(expected_text, encoding="utf-8")
    output_path.unlink(missing_ok=True)

    return_code = _run_process(
        command,
        cwd=work_dir,
        timeout=time_limit,
        memory_limit_mb=memory_limit,
        stdin_path=input_path if input_mode == "stdio" else None,
        stdout_path=output_path if input_mode == "stdio" else None,
    )
    if return_code != 0 or not output_path.is_file():
        return False
    if output_path.stat().st_size > _MAX_OUTPUT_BYTES:
        return False

    if checker_path is not None:
        return _run_checker(
            checker_path=checker_path,
            input_path=input_path,
            expected_path=expected_path,
            output_path=output_path,
            work_dir=work_dir,
        )

    actual_text = output_path.read_text(encoding="utf-8", errors="replace")
    return _normalize_output(actual_text) == _normalize_output(expected_text)


def _run_checker(
    checker_path: Path,
    input_path: Path,
    expected_path: Path,
    output_path: Path,
    work_dir: Path,
) -> bool:
    expected_path.write_text(
        _remove_empty_lines(expected_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    output_path.write_text(
        _remove_empty_lines(output_path.read_text(encoding="utf-8", errors="replace")),
        encoding="utf-8",
    )
    checker_output = work_dir / "checker_output.txt"
    return_code = _run_process(
        [
            sys.executable,
            str(checker_path),
            str(input_path),
            str(expected_path),
            str(output_path),
        ],
        cwd=work_dir,
        timeout=10.0,
        stdout_path=checker_output,
    )
    if return_code != 0 or not checker_output.is_file():
        return False
    checker_score = checker_output.read_text(encoding="utf-8").strip()
    return checker_score in {"1", "100"}


def _run_process(
    command: list[str],
    cwd: Path,
    timeout: float,
    memory_limit_mb: float | None = None,
    stdin_path: Path | None = None,
    stdout_path: Path | None = None,
) -> int:
    stdin_handle = stdin_path.open("rb") if stdin_path is not None else None
    stdout_handle = stdout_path.open("wb") if stdout_path is not None else None
    try:
        process = subprocess.Popen(  # noqa: S603 - task explicitly executes model code
            command,
            cwd=cwd,
            stdin=stdin_handle or subprocess.DEVNULL,
            stdout=stdout_handle or subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=_resource_limiter(memory_limit_mb),
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return -1
        return process.returncode
    finally:
        if stdin_handle is not None:
            stdin_handle.close()
        if stdout_handle is not None:
            stdout_handle.close()


def _resource_limiter(memory_limit_mb: float | None):
    def set_limits() -> None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_FSIZE, (_MAX_OUTPUT_BYTES, _MAX_OUTPUT_BYTES)
        )
        if memory_limit_mb is not None:
            memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)
            resource.setrlimit(
                resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes)
            )

    return set_limits


def _unix_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_empty_lines(text: str) -> str:
    return "\n".join(line for line in _unix_newlines(text).splitlines() if line)


def _normalize_output(text: str) -> str:
    text = re.sub(
        r"\b(?:yes|no)\b",
        lambda match: match.group(0).upper(),
        _unix_newlines(text),
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())
