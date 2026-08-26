# Architecture

Pressure-tested in session 0 against an adversarial review. `src/ds_agents/state.py` is the
authoritative contract; this file explains the reasoning and the parts that live outside the code.

## Graph

```
intake -> profiler -> feature_eng -> modeler -> reviewer -> reporter
                        ^              ^            |
                        |______________|____________|  (router may send back, max N loops)
```

The reviewer is adversarial: it can block a run and send it back to `feature_eng` or `modeler`
with a structured objection. Two things it does **not** control:

- **It does not increment the loop counter.** The router does. A model under test cannot be
  trusted to enforce its own cap.
- **It does not write the verdict.** It writes `reviewer_claim` (`pass` or `block`). The router
  derives `review_verdict`: a claim of `block` while `review_iterations >= loop_cap` becomes
  `exhausted`. Otherwise a reviewer that simply emits "pass" at the cap would be indistinguishable
  from one that passed on merit, and `exhausted` is the outcome DECISIONS.md calls interesting.

## State

See `src/ds_agents/state.py`. Two rules shape it:

1. **The system under test does not report its own grade.** The modeler writes
   `chosen_model.claimed_holdout_score`. The harness writes `verified_holdout_score`, scored
   outside the graph on a holdout the agents never see. The gap between them is a finding, not an
   error.
2. **Anything the eval counts is a field, never prose.** `Objection.columns` exists because
   `leakage_caught` must be a set comparison against ground truth. Computing it by string-matching
   a model's sentence would score a leakage objection naming the *wrong* column as a catch.

Invariants worth stating explicitly:

- `RunConfig` is frozen. It carries `run_id`, `arm`, `reviewer_enabled`, `reviewer_model`,
  `loop_cap`, `reviewer_sees_code`, and `random_seed`. Every results row is self-describing from
  the state object alone — without it, "reviewer disabled" and "reviewer crashed" are the same row.
- `split_artifact` is pinned by the profiler before `feature_eng` runs and never rewritten. Without
  a pinned split the `contamination` category is unfalsifiable and no run is reproducible. It
  partitions the rows the AGENTS were given, and is *not* the independent holdout behind
  `verified_holdout_score`, which is withheld before the graph starts and never appears in this
  manifest.
- `objections`, `review_passes`, `errors`, and `node_trace` are append-only, annotated with
  `operator.add` so LangGraph merges rather than replaces. An unannotated list field would be
  overwritten by whichever node wrote last, silently emptying the cost table.
- Objections are never mutated. Resolution is recorded as a `ReviewPass.dispositions` entry, so
  "the reviewer withdrew it" stays distinguishable from "the reviewer never looked again."
- `final_features` records what actually reached the model, so *reviewer caught the leak* and
  *the leak was removed* are two different numbers.

Objection categories (small and enumerable so they are countable): `leakage`, `contamination`,
`overfit`, `metric_mismatch`, `implausible_importance`, `spec_violation`, `other`. Every objection
also carries a free-text `subcategory`, so a reviewer catching something unanticipated is not
forced to mislabel it.

## Node contracts

| node | reads | writes | tools | routing |
|---|---|---|---|---|
| intake | dataset_id, task_description | spec | read_artifact | profiler |
| profiler | spec, dataset_id, config.random_seed | profile, split_artifact | run_python | feature_eng |
| feature_eng | spec, profile, open objections | feature_code_artifact, feature_summary, final_features, dropped_features | run_python, write_artifact | modeler |
| modeler | spec, feature_code_artifact, split_artifact, open objections | candidates, chosen_model, shap_artifact, top_importances | run_python, write_artifact, log_metric | reviewer |
| reviewer | everything above; feature code only if `config.reviewer_sees_code` | objections, reviewer_claim, review_passes | read_artifact | router decides |
| router | reviewer_claim, review_iterations, config.loop_cap | review_iterations, review_verdict | none | feature_eng / modeler / reporter |
| reporter | everything | report_artifact | read_artifact, write_artifact | END |

No node writes `planted_leakage_columns`, `verified_holdout_score`, or `baseline_score`. Those are
harness-written ground truth; a node that could see them could game them.

A node's signature is `node(state, *, tools, model) -> dict`. It returns a **narrow dict of the
fields it changed**, never the state object: with `operator.add` on the history fields, returning
state would concatenate the accumulated trace onto itself. `graph.py` binds `tools` and `model`
with `functools.partial`, which is also what keeps the Phase 2 MCP swap confined to that one file.

Two things the table does not show:

- **Intake sees the dataset as an artifact.** It is registered at run start under
  `dataset:<dataset_id>` with columns, dtypes, row count, and per-column cardinality in its
  metadata, so intake can name a target without being given `run_python`.
- **A snippet can create an artifact without calling `write_artifact`.** Anything a snippet drops
  in `DS_ARTIFACTS` is detected and indexed when `run_python` returns, and its id comes back in
  `artifacts_written`. That is how the profiler's split manifest is written without shipping the
  whole row-id list back through stdout. Detection is by content change, not by filename, and the
  file is copied under its artifact id: an artifact is immutable once issued, or `split_artifact`
  being "pinned and never rewritten" means nothing. **Phase 2's server must keep this behaviour.**
- **The profiler refuses split strategies it has not implemented.** `temporal` and `grouped` are
  valid on `TaskSpec`; the snippet handles `random` and `stratified`. Falling back to a random
  split while labelling the manifest `grouped` would put one entity on both sides of the partition
  while the file claims otherwise, which is the exact contamination the manifest exists to make
  falsifiable. It writes no manifest and records an unrecoverable error instead.
- **The profiler is split in two.** The mechanical statistics -- including each column's
  normalized mutual information with the target -- come from a fixed snippet through `run_python`.
  The model is asked only which of those numbers mean "this column encodes the answer" rather than
  "this column is a strong legitimate predictor", which is the part the eval measures.

## MCP server (ours)

Phase 1 satisfies this surface in-process with `src/ds_agents/tools/local.py`, which has the same
signatures and none of the isolation. Nodes type-hint against the `Tools` Protocol and never
import a concrete implementation, so both arms of the single-agent-vs-team ablation are guaranteed
the same surface and the Phase 2 swap does not reach into `nodes/`.

Tools exposed:

- `run_python(code: str, timeout_s: int) -> {stdout, stderr, exit_code, artifacts_written}`
  Runs in a subprocess inside the sandbox container with the dataset mounted read-only and an
  artifacts dir mounted read-write. No network.
- `read_artifact(id) -> {content | path, meta}`
- `write_artifact(name, content | path) -> id`
- `log_metric(run_id, name, value)` — `run_id` comes from `state.config.run_id`.

Why MCP and not plain Python functions: the agents then have exactly the same tool surface as any
external MCP client, which makes the "single generalist agent vs team" ablation honest, and lets us
point Claude Code at the same server as a comparison.

## Observability

LangSmith tracing on by default (env var). Every node appends a `NodeEvent` to `node_trace` with
token counts and cost so results files do not depend on the tracing backend. All timestamps are
timezone-aware UTC; naive ones make committed JSONL inconsistent across machines.

## Eval outcomes per dataset

Produced by `PipelineState.results_row()`. If a number in the published tables cannot be traced to
that method, it is not a real number.

Scores: `claimed_holdout_score`, `verified_holdout_score`, `holdout_claim_gap`, `baseline_score`,
`score_ratio` (direction-aware — a ratio means the opposite thing for RMSE and AUC), `metric`.

Leakage, as a set comparison against ground truth rather than two booleans: `leakage_planted`,
`leakage_flagged`, `leakage_caught`, `leakage_remediated`, `leakage_recall`, `leakage_precision`,
`false_alarm_columns`, `false_alarm`, plus `leakage_flagged_standing` and `false_alarm_standing`.
The `_standing` pair drops columns whose every objection was later resolved or withdrawn: catching
a leak at any point is a genuine catch, but taking back a false alarm is better behaviour than
leaving it standing, and one number cannot say both. A binary pair cannot express "caught the real one and also
flagged three clean columns."

Loop: `review_verdict`, `review_loops`, `objections_raised`, `objections_open_at_end`.

Cost and reliability: `wall_seconds` (real elapsed, not the sum of node events), `cost_usd`,
`errored`. The harness must emit a row for every dataset even on hard failure, or the hardest
datasets disappear and every table biases upward.
