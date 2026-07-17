# Codeforces

This task evaluates complete Python and C++17 competitive-programming solutions
from the `test` split of the
[`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces)
`verifiable-prompts` subset. The subset contains one Python prompt and one C++
prompt for each held-out problem.

The generated code is executed locally against the official test cases included
in each dataset row. The evaluator supports standard-input and file-based
problems as well as the dataset's generated custom checkers.

The dataset stores its additional generated tests separately. For full
verification, download them and set `CF_TESTS_FOLDER` to the download root:

```bash
huggingface-cli download open-r1/codeforces \
  --repo-type=dataset \
  --include='generated_tests/*.parquet' \
  --local-dir=/path/to/codeforces
export CF_TESTS_FOLDER=/path/to/codeforces
```

Without `CF_TESTS_FOLDER`, evaluation still runs against the inline official
tests and emits a warning when a problem has additional tests available.

Because model-generated code and dataset checkers are executed, the task is
marked unsafe and must be explicitly enabled:

```bash
lm_eval --model hf \
  --model_args pretrained=MODEL \
  --tasks codeforces \
  --confirm_run_unsafe_code
```

`g++` is required for C++ rows.

## Tasks

- `codeforces`: greedy generation with `pass@1`.
- `codeforces_avg8`: eight samples at temperature 0.7, reporting each
  `sample@1` through `sample@8` and their `avg@8` mean.

## Citation

```bibtex
@misc{penedo2025codeforces,
  title={CodeForces},
  author={Guilherme Penedo and Anton Lozhkov and Hynek Kydl{\'i}{\v c}ek and
          Loubna Ben Allal and Edward Beeching and Agust{\'i}n Piqueres Lajar{\'i}n
          and Quentin Gallou{\'e}dec and Nathan Habib and Lewis Tunstall and
          Leandro von Werra},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/open-r1/codeforces}}
}
```
