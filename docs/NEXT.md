# Next session

## Start here
**Phase 4's last box is closed. `baseline_score` and `score_ratio` are retired, not implemented.**
`score_ratio` computed `verified / baseline`; AMLB's convention is `(x − zero) / (unit − zero)`, and
those are different functions that disagree about what 1.0 means. What ships instead is two raw
points plus one derived column: `baseline_zero_score`, `baseline_unit_score`,
`baseline_normalised_score`, `baseline_status`, `baseline_detail`, `baseline_recipe`. Retiring two
published columns was free and that is **checked rather than claimed** — all 149 committed rows
carried both as `null`, pinned by test, and no committed results file was edited.

**The zero point is a correctness assertion, and it fired.** A constant class-prior predictor scores
exactly 0.5 roc_auc by construction — every pair is a tie — so 0.5 is not a measurement with a
tolerance, it is an assertion that the grader resolved the positive class, applied the scorer sign,
and scored the rows it meant to. **Exactly 0.5 on 4 of 4 live rows.**

**The pooling hazard is closed by a gate that cannot fire yet.** `baseline_normalised_score` returns
`None` whenever `planted_leakage_columns` is non-empty, because the baseline is fit on every raw
column including the trap. Said plainly: that contradiction is currently *unreachable* — fixtures
have `withheld_fraction = 0.0`, so no fixture row is graded and every row that can carry a baseline
has an empty planted list. The gate is written now because it costs one `if` now, and because
widening the carve to fixtures is already parked as a future cell. The two raw points are **not**
gated; only the quotient is.

**The baseline is independent, and it runs in its own process.** It sees the raw frame through the
grader's own mechanical encoder — not the agents' `feature_code_artifact` — because a yardstick that
inherits the decisions it measures cannot say whether those decisions helped. Its own sandbox, its
own timeout, its own ten-value `BaselineStatus`, because a RandomForest that dies on a wide frame
must not take `verified_holdout_score` with it; `unit_point_failed` keeps the zero point.

**Live: 4 rows at $0.0645, all four pre-registered endpoints passed.** `baseline_status` ok 4/4,
zero exactly 0.5, `baseline_recipe` `rf-v1` 4/4, $0.0161/run against a measured $0.0168 — the two
extra fits cost no tokens. Characterisation only, explicitly not a result: unit 0.7660, verified
0.7476, normalised **0.9309**. `evals/results/2026-08-31_baseline-smoke.jsonl`.

The floor: **741 tests pass** (up from 716), 27 skipped, ruff clean, `uv run ds-agents run --dataset
toy` green live at $0.0130 and reporting `baseline_status: no_withheld_holdout`. Session spend
~$0.11 all in.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. **`--subset full` has one blocker
left and it is money, not code.** The grading is complete: a manifest dataset runs, is scored on a
withheld holdout, and is placed on a measured scale.

**Start from the timing table, because it is the finding that changes the plan.** The unit point was
timed at three shapes: **0.20s at `credit_g` (1000×20), 9.85s at `adult` (48842×14), 72.09s at
`higgs` (98050×28)**, against an LLM portion of ~21s per run. So the baseline is invisible at the
cheapest dataset and roughly quadruples wall time at the largest, and **none of that was visible
from `bench-smoke`**. This is the second time the cheapest dataset has hidden a cost term from this
project — the first was `bench-smoke`'s own $0.040-guessed / $0.017-measured error.

Three things that follow, in order:

1. **Measure a mid-sized dataset before funding thirteen.** `phoneme` or `australian` are cheap and
   also answer the standing "is that the dataset or the pipeline?" question. Then one of `adult` /
   `nomao` / `jasmine` to get a second point on the wall-clock curve — one point is not an estimate.
2. **Decide whether `higgs` is worth 72s of yardstick per run**, or whether `n_estimators` drops for
   the big frames. If it drops, `baseline_recipe` must change with it, and rows at two recipes must
   not be pooled — that is what the column is for.
3. **`full` is 13 cells**, because `dataset_id` is an `eval-diff` condition field. Price it as 13
   cells × replicates, not as 13 runs.

## Open questions

- **Is `higgs` affordable, and at what recipe?** 72s of RandomForest per run against ~21s of LLM. A
  cheaper `n_estimators` for large frames is defensible and is exactly what `baseline_recipe`
  exists to make visible — but it means the manifest is graded against two yardsticks, and the
  writeup has to say so.
- **`credit_g` produced four identical rows again, at a second commit.** Every substantive column
  matches; the only field that varies is `cost_usd`. The claim is still that this is the dataset —
  no leak, no objections, no drops, so no decision available to vary — and it still needs one cheap
  run on a second manifest dataset before anyone writes "the pipeline is deterministic on real
  data".
- **`baseline_normalised_score` of 0.9309 says the pipeline landed just below a raw-column
  RandomForest on `credit_g`.** Not quotable: n=1 effective, the grid is ours, and the encoder keeps
  high-cardinality columns `feature_eng` skips, which makes the unit point a **floor** rather than a
  strong model. Whether a floor is the right unit point at all is a real question for the writeup.
- **Is `amazon_employee_access` a real dataset or a degenerate one?** Unchanged and still cheap to
  answer. Nine integer-encoded high-cardinality ID columns land in the baseline's *numeric* branch
  and pass straight through, so this run now probes the encoder as well as `min_usable_features`.
- **Is a partially-labelled dataset gradeable at all?** Unchanged. `bank_marketing`'s documented
  `V12` is still unscored, deliberately, because `planted_leakage_columns` is read as complete. Note
  it would also be the first dataset where the leak-suppression gate could matter — if `V12` were
  ever promoted, `baseline_normalised_score` would go null on that row by design.
- **200 withheld rows put roc_auc's standard error near 0.04**, so a `holdout_claim_gap` under
  ~0.08 is not distinguishable from noise at n=1. Unchanged, and it binds every gap published here.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
  Unchanged. `claims_timing`'s trap would survive it. Bounds what `holdout_claim_gap` can mean.
- **`published_reference` picks the max over uploaded runs**, which is not a protocol-stable anchor.
  Unchanged. `credit_g` reads 0.7889 against our 0.7476 on a different protocol.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Unchanged. It now
  has two companions: `rescore_status` would read `no_model` and `baseline_status`
  `rescore_unavailable`, so the two-columns fix exists twice over while `review_verdict` is still
  wrong.
- **No CI job and no thresholds.** Unchanged.
- **`errored` needs a companion column.** Unchanged, and `baseline_status` is now the *third* worked
  example of the fix, after `leakage_graded` and `rescore_status`.
- **`exhausted` is still uninformative.** Unchanged.
- **Late detection against the cap** is the remaining structural reviewer defect. Unchanged.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's own per-dataset numbers, which
  is why the unit point's grid is ours.
- **Two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`, md5 mismatches from
  OpenML itself). Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **The manifest prose rename is deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`; both are rendered verbatim into `evals/datasets/manifest.yaml` (15 occurrences),
  which only `ds-agents datasets refresh` may write, and a refresh re-fetches every OpenML response
  and can move `published_reference`. Changing prose is not worth a silently-moved citation.
  Verified: no offline test compares the two, so nothing is red. The non-rendered docstrings and
  comments in both modules were updated.
- **The unit point is a floor, not a ceiling, and that is deliberate.** The grader's encoder keeps
  high-cardinality columns `feature_eng` skips and ordinal-encodes nominal codes, imposing a false
  ordering. Stated so nobody later reads "beat the baseline" as "beat a strong model".
- **`ModelResult.model_artifact` is still declared and never written.** Both the re-scorer and the
  baseline refit instead. Unchanged.
- **The sandbox is not a jail, and the docstrings say so.** A snippet could open the withheld CSV by
  relative path. What the layout buys is an *assertable* property, checked by test, not isolation.
  Real isolation waits on the Docker backend behind `SandboxPool`.
- **`register_dataset` does an unbounded `pd.read_csv` + per-column `nunique` per run**, and copies
  the file. Free at `credit_g`'s 139 KB. **Measure before `higgs` at 46 MB** — and note this now
  sits alongside a 72s baseline fit on the same dataset.
- **`.mcp.json` still hardcodes the toy dataset on argv.** Unchanged.
- **`SUBSETS` has no `full` entry**, and `_resolve_subset`'s message has now been wrong three times
  for the same reason: the blocker keeps moving. It currently names cost, and a test asserts it does
  NOT still name `baseline_score`, `score_ratio` or `verified_holdout_score`.
- **The `credit_g` withheld indices are pinned as a sha256 digest**, not 200 literals.
- **`_withhold_rows` uses numpy directly rather than `train_test_split`** — sklearn's stratified
  shuffling is not contracted stable across versions.
- **`csv_sha256` mismatches warn rather than refuse** (`benchmark.require_cached_csv`).
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"`.**
- **The cost cap is invocation-level; a failed run is charged its estimate.** Unchanged.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Unchanged.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **`feature_eng.py`'s comment on the justification string is still wrong** for a future reader.
- **`harness.py` and `cli.py` still import each other inside functions.** Deferred again.
- **The loop-cap default `3` is written in three places.**
- **`cmd_run` has no per-run `try/except`.** Unchanged.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
