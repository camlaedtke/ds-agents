---
name: run-eval
description: How to run the benchmark harness, record results, and compare against prior runs. Use this whenever the task involves benchmarking, evaluating the pipeline, running an ablation, checking for regressions, comparing models, or producing numbers for the README. Trigger even for casual phrasing like "see if the reviewer is working" or "how does it do on the datasets now."
---

# Run an evaluation

The harness is the product. Treat every run as something that might end up in a results table.

## Subsets

| subset | datasets | when |
|---|---|---|
| `toy` | tests/fixtures/toy only | every session, sanity |
| `ci` | 3 fast datasets incl. 1 leakage trap | every push, regression gate |
| `full` | everything in `evals/datasets/manifest.yaml` | ablations, README numbers |

Never run `full` without confirming the config with the user first. It costs real money.

## Before running

1. Check `git status` is clean or the changes are intentional. The harness records the commit hash.
2. Set a run name that says what is different: `--name reviewer-haiku` not `--name test3`.
3. State the config out loud: models per node, reviewer on/off, loop cap, subset.

## Run

```
uv run ds-agents eval --subset ci --name <run-name> [--no-reviewer] [--reviewer-model sonnet]
```

Output lands in `evals/results/<date>_<run-name>.jsonl`, one line per dataset with:
`dataset, commit, config, holdout_score, baseline_score, score_ratio, leakage_planted,
leakage_caught, false_alarm, review_loops, wall_seconds, input_tokens, output_tokens, cost_usd`.

Use the `test-runner` subagent for anything beyond `toy` so the run output stays out of the main context.
Ask it to report only the summary table and any dataset that errored.

## After running

1. Compare: `uv run ds-agents eval-diff <prev.jsonl> <new.jsonl>`. Flag any dataset whose
   `score_ratio` moved more than 0.05 or whose leakage outcome flipped.
2. Append a dated entry to `evals/results/LOG.md`: run name, config, the headline numbers, one
   sentence on what it tells us. This log becomes the README results section.
3. If a regression: do not "fix" it by rerunning. Find the cause. Agent pipelines are noisy, so
   rerun once with the same config to check variance before calling it a regression.
4. Commit the results file. Results are part of the repo.

## Ablations we care about (for README)

- reviewer on vs off
- reviewer on Haiku vs Sonnet
- single generalist agent with the same tools vs the specialized team
- loop cap 1 vs 3

Each ablation is two runs on the same subset with one config difference. Name them as a pair.

## Do not

- Edit anything under `evals/datasets/` to make a number look better.
- Report a single run as a result when the ablation is close. Note variance.
- Quote a number in docs that is not traceable to a results file.
