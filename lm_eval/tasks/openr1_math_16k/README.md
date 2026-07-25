# OpenR1 Math 16K

This task evaluates the 1,000-example held-out split in
`data/openr1_math_16k/test.jsonl`. The examples were deterministically selected
from the verifiable prompts in `open-r1/OpenR1-Math-220k`; see the dataset
directory's README and filter report for the exact curation policy.

Install the math dependencies before running the task:

```bash
pip install -e '.[math]'
```

## Tasks

- `openr1_math_16k`: one greedy response per problem, reported with AIME-style
  exact matching and Math-Verify matching.
- `openr1_math_16k_avg8`: eight sampled responses per problem. `sample@1` through
  `sample@8` and `avg@8` use exact matching; the corresponding Math-Verify
  metrics are prefixed with `mv_`.

Both tasks use only the local `test.jsonl`; the 15,000-example training split is
not loaded by lm-eval.
