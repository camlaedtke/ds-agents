---
name: run-eval
description: How to run the benchmark harness, record results, and compare against prior runs. Use this whenever the task involves benchmarking, evaluating the pipeline, running an ablation, checking for regressions, comparing models, or producing numbers for the README. Trigger even for casual phrasing like "see if the reviewer is working" or "how does it do on the datasets now."
---

# Run an evaluation

The harness is the product. Treat every run as something that might end up in a results table.

## The rule that comes before everything else

**One cell per arm cannot resolve a difference of 4 runs in 10.** Two cells in this repo running
identical behaviour returned `leakage_remediated` 9/10 and 6/10. Model nondeterminism alone moves a
10-run count by about 3, which is most of the size of every effect this project has reported.

So: **run at least 2 replicates per arm, and let `eval-diff` decide.** If it says `underpowered`,
the honest answer is "not resolved" -- not "directionally". The 5/10 -> 9/10 sticky-drop headline
was retired for exactly this reason; it cost $0.54 to find out and it is the most useful thing the
project has learned. Do not quote any 10-run count as an effect without a replicate.

## Subsets

| subset | what runs | cost | when |
|---|---|---|---|
| `toy` | `toy-default` only | ~$0.01 | every session, sanity |
| `ci` | 3 cells, one per fixture | ~$0.03/run | regression gate |
| `full` | errors today | -- | blocked on `evals/datasets/manifest.yaml` |

`full` is not implemented: it needs the manifest from PLAN.md Phase 4's first checkbox, which is
blocked on the open question of which OpenML suite has citable published baselines. The command
tells you this rather than running something else.

Never run anything beyond `toy` without confirming the config and the cost cap with the user first.

## Before running

1. `git status` clean, or know why it is not. Every row records `commit`, and a dirty tree is
   recorded as `<hash>-dirty` and grouped as its own cell -- a dirty run is not reproducible.
2. `--dry-run` first. Read the plan and the cost estimate before spending anything.
3. Name the run for what is different: `--name reviewer-haiku`, not `--name test3`.
4. State the config out loud: subset, replicates, n, models, cost cap.
5. Never change `random_seed`. It is a data seed (split and estimator), not a model seed. Moving it
   folds split variance into the number this project reports as model variance and breaks
   comparability with every committed row.

## Run

```
uv run ds-agents eval --subset ci --name <run-name> [--replicates 2] [--n 5] \
    [--max-cost-usd 0.50] [--dry-run] [--tools mcp|local]

uv run ds-agents eval-diff <before.jsonl> <after.jsonl>
```

Runs are ordered replicate-major, so a binding cost cap truncates every cell roughly equally
instead of starving the last one. A run that raises is counted and skipped rather than ending the
cell; three consecutive failures abort the invocation.

Use the `test-runner` subagent for anything beyond `toy` so the run output stays out of the main
context. Ask it to report only the summary table, any cell that errored, and `runs_failed`.

## Output

`evals/results/<date>_<run-name>.jsonl`, appended, one line per run.

**The row's definition is `PipelineState.results_row()`.** Do not enumerate its fields here -- an
inline list is what made the previous version of this document wrong for three sessions. On top of
those fields the harness annotates each row with `cell`, `replicate`, `run_index`, `eval_subset`
and `eval_name`; `commit` comes off the frozen config, not from the harness.

Rows are written only if `state.publishable()` passes. A refused row is counted and reported,
because a refusal changes a cell's denominator.

## After running

1. `uv run ds-agents eval-diff <prev> <new>`. It refuses to call an underpowered difference an
   effect, excludes `None` metrics from denominators rather than counting them as failures, and
   flags a leakage flip whether or not the comparison separates.
2. Append a dated entry to `evals/results/LOG.md`: run name, cells, n, replicates, commit, spend,
   the headline numbers WITH their intervals, `runs_failed`, and one sentence on what it tells us.
   Report every pre-registered endpoint, including the ones that failed. This log becomes the
   README results section.
3. Commit the results file. Results are part of the repo.
4. If a regression: do not "fix" it by rerunning. Find the cause. Rerun once with the same config
   to check variance before calling it a regression.

## Ablations we care about (for README)

- reviewer on vs off
- reviewer on Haiku vs Sonnet
- single generalist agent with the same tools vs the specialized team
- loop cap 1 vs 3

Each is two arms differing in one condition -- and each arm now needs at least 2 replicates, so
re-price them: a model arm needs a replicated Haiku baseline to be compared against, roughly double
the old estimate.

## Do not

- Edit anything under `evals/datasets/` or a committed results file to make a number look better.
- Quote a single cell as an effect, or quote a number in docs that is not traceable to a results
  file.
- Change `random_seed`.
- Pool rows across a `commit` boundary without saying so in the table.
- Use `--forced-drop-release`. It is a closed axis: exactly one legitimate use, already spent.
- Use `ds-agents run --results` for a comparison. It still exists for one-off cells and debugging,
  but its rows carry no `cell` or `replicate`, so they can never enter a powered comparison.
