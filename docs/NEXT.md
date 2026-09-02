# Next session

## Start here
**Phase 4 is closed. `--subset full` runs, and what it waits on is a decision, not code.**
The last four datasets are priced -- `adult` $0.0159, `bank_marketing` $0.0178, `numerai28_6`
$0.0172, `higgs` $0.0246, all n=4 at one commit, `evals/results/2026-09-02_bench-tall.jsonl`, 16
rows for $0.3021. `SUBSETS["full"]` exists with 13 cells (9 measured means, 4 modelled), and
`_resolve_subset`'s bespoke `full` message is deleted after four corrections. Running `full`
properly is `--replicates 2 --n 2` = **52 runs, about $1.10 and 45-70 minutes.** Nobody has been
asked whether to spend that. **Asking is the first thing next session does.**

**The arm was a pre-registered hypothesis test and it returned the null option, which is the useful
part.** The column-only cost model under-predicted all four cells; with `credit_g` that is 5 of 5
out-of-sample datasets under-predicted, sign test p=0.031. But neither registered hypothesis
produced its ordering: H_categorical predicted `adult` and `bank_marketing` high and only
`bank_marketing` was; H_rows predicted the two tall cells high and only `higgs` was; and
`bank_marketing` at 45k rows ran dearer than `numerai28_6` at 96k, which no row term orders.
Refitting on nine datasets, a row term cuts residual sd $0.00296 -> $0.00224 and a categorical term
makes it **worse** ($0.00320). **H_categorical is refuted** -- with a real mechanism behind it and
prior evidence pointing its way. A confound is a reason to run the experiment, not a result.

**Two things NEXT.md itself had wrong, both corrected before any money moved.** `MODEL_TIMEOUT_S`
was never the `higgs` risk: `permutation_importance` costs `n_repeats x n_columns x n_candidates`
CALLS, so `higgs` at 28 columns issues 560 against `jasmine`'s 2,880. Measured 12.6s at the real
binary shapes and 23.6s live, against a 240s budget. And `register_dataset` was not "never
measured" -- it was measured for nine datasets on 2026-09-01, as prose in a commit that touched no
code. `tests/test_register_dataset_cost.py` now prints a re-derivable table for all thirteen;
`higgs` is worst at 0.25s a call, 0.75s a run, 1.2% of budget.

The floor: **799 tests pass** (up from 783, plus two new modules), ruff clean, toy pipeline green
live at $0.0304. Every number in the cost model is now derived from the committed results files by
`tests/test_cost_model.py` rather than quoted in prose -- which caught, on its first run, that
`amazon_employee_access` was priced $0.000007 BELOW its own measured mean against a rounding rule
`harness.py` had stated correctly for two sessions.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 5, and the "Start here" above. **Ask Cameron whether to spend
~$1.10 on `--subset full` before doing anything else** -- it is the only thing standing between this
repo and a results table covering all thirteen datasets, and it is his call rather than a technical
question. If yes, follow /run-eval: `--subset full --name full --replicates 2 --n 2
--max-cost-usd 1.60`, pre-register per cell first, and expect the four never-run datasets
(`australian`, `kc1`, `sylvine`, `kr_vs_kp`) to be the interesting ones.

If no, or while waiting, the two defects below are both small, both real, and both found by this
session rather than assumed.

## Open questions

- **`errored` is TRUE on four completely healthy runs, and the "closed" mark was wrong.** All four
  `adult` rows carry `errored: true`, `halted_at: null`, no objection, verdict `pass`, and a
  `verified_holdout_score` of 0.9244 that nothing objected to. `feature_eng` records "columns skipped as too
  high-cardinality to one-hot encode at 20 levels" as a `PipelineError` with `recoverable=True`, and
  `errored` is `bool(self.errors)`. So a table using `errored` as a rate reports `adult` as a 100%
  failure cell. `halted_at` fixed fatal-versus-non-fatal, which is what it was for; it did nothing
  about recoverable-and-not-a-problem. **The fix is not a sixth companion column -- it is that a
  routine decision should not be a `PipelineError`.** That is a node change and it needs
  /add-node. Reopens the item the 2026-09-01 entry marked CLOSED.
- **`baseline_normalised_score` divides by a span that can approach zero.** `numerai28_6` returns
  **2.089**, the largest value anywhere here: zero is 0.5, the unit point reaches only 0.5101, so
  the denominator is 0.0101. The arithmetic is exactly AMLB's and it is working; the dataset is
  near-chance (published reference 0.530). Three of four cells are now above 1.0. Nothing warns and
  `baseline_status` is `ok`. Wants a guard -- a status value, or `None` below some minimum span, the
  way it already returns `None` on a planted leak. Which one is a design decision.
- **A fully numeric dataset DID have a decision available, and it cost 0.09 roc_auc.** One `higgs`
  run of four kept 24 of 28 features with no objection raised -- `feature_eng` proposed the drops
  itself -- and scored 0.7085 verified against 0.8006 for the other three, `baseline_normalised_score`
  0.716 against 1.031. This refutes the pre-registered claim that numeric datasets have nothing to
  decide differently, and it is the first measured case of an unforced drop costing real
  performance. Worth its own cell before it is quoted.
- **Determinism on real data: settled enough to state carefully.** Across four new datasets at n=4,
  `bank_marketing` and `numerai28_6` returned 4 identical rows each, `adult` gave 3 distinct
  outcomes and `higgs` 2. So "identical where no decision was available" survives; the general
  claim was already refuted by `nomao` on 2026-09-01, and `higgs` now refutes the numeric-datasets
  half of the excuse too.
- **Is `halted_at` the right shape for `eval-diff`?** Unchanged, and now with a second instance of
  the same shape: `errored` cannot be a rate either, for a different reason. `--metrics` accepting a
  null/non-null predicate would serve both.
- **The four never-run datasets.** `australian`, `kc1`, `sylvine`, `kr_vs_kp` have prices but no
  runs. `kr_vs_kp` is 36 columns, all categorical, 73 one-hot levels -- H_categorical was refuted at
  7 to 13 categorical columns, which is not refuted at 36, so its $0.022 is the softest number in
  `SUBSETS`.
- **The cost of a run that loops is still unmeasured on any manifest dataset.** All 16 runs raised
  zero objections and looped once. A run that loops three times costs about 2.5x, and `full`'s
  $1.10 assumes 52 runs that all pass first time.
- **`register_dataset`'s two grader calls are unbounded under BOTH transports.** Only the graph's
  registration passes through `_select_tools` and sees `_CONNECT_TIMEOUT_S`; `rescore` and
  `baseline` build `LocalTools` directly. A hang there does not raise, does not abort, and appears
  in no column. Measured at 1.2% of budget at the largest shape, so the exposure is small and the
  guard is worth designing rather than bolting on. Nothing tests the `SystemExit` path either.
- **`bank_marketing`'s `V12` was kept by all four runs and objected to by none.** No longer
  hypothetical: the dataset runs, the documented `recorded_after_outcome` column survives to
  `final_features` every time, and it stays outside `planted_leakage_columns` deliberately.
- **200 withheld rows put roc_auc's standard error near 0.04.** Unchanged, and it binds every gap
  here. The four new datasets withhold 9,042 to 19,609 rows, so their gaps are much tighter numbers.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
  Unchanged. `claims_timing`'s trap would survive it.
- **`published_reference` picks the max over uploaded runs**, which is not a protocol-stable anchor.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still half open: the
  zero-objection `block` path.
- **No CI job and no thresholds.** Unchanged, and the reason is policy rather than code.
- **`exhausted` is still uninformative.** Unchanged.
- **Late detection against the cap** is the remaining structural reviewer defect. Unchanged.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's own per-dataset numbers.
- **Two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`). Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.
- ~~**`errored` needs a companion column.**~~ **REOPENED 2026-09-02** -- see the first item. The
  2026-09-01 close was too strong.

## Parking lot

- **The manifest prose rename is still deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`; both render verbatim into `evals/datasets/manifest.yaml`, which only
  `ds-agents datasets refresh` may write, and a refresh re-fetches every OpenML response and can
  move `published_reference`.
- **The retained cost model** is `$0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`,
  residual sd $0.0022 on 9 datasets and 3 parameters. Planning only -- `est_cost_usd` never reaches
  a results row -- and the row term's improvement rests heavily on `higgs`. Re-derived by
  `tests/test_cost_model.py`, which also pins the pre-registered `bench-mid` fit against the exact
  four estimates it was computed on, so that band cannot be retroactively moved.
- **`test_baseline_cost.py`'s synthetic bound is loose by about 1.6x on real data.** `higgs` measured
  36.2-36.8s live against the table's 57.18s. The table says it is an upper bound and it is.
- **The new cliff is 1,048,172 agent rows**, pinned by test. Nothing in the manifest is within two
  orders of magnitude.
- **The encoding narrows what the artifact can say, deliberately.** A split that discards rows is
  unrepresentable and the encoder raises rather than writing a manifest with a hole in it.
- **`store.path_of` still has no caller anywhere in `src/` or `mcp_server/`.** Dead code.
- **The unit point is a floor, not a ceiling** -- and that framing is now insufficient. It describes
  values slightly above 1.0; it does not cover `numerai28_6`'s 2.089. See the open question.
- **`ModelResult.model_artifact` is still declared and never written.** Unchanged.
- **The sandbox is not a jail, and the docstrings say so.** Real isolation waits on the Docker
  backend behind `SandboxPool`.
- **`.mcp.json` still hardcodes the toy dataset on argv.** Unchanged.
- **`SUBSETS` HAS a `full` entry as of 2026-09-02**, and `_resolve_subset` has no special case for
  it. The message that was corrected four times is gone rather than corrected a fifth time.
- **The `credit_g` withheld indices are pinned as a sha256 digest**; the toy split likewise as
  `TOY_ASSIGNMENT_DIGEST`.
- **`_withhold_rows` uses numpy directly rather than `train_test_split`.**
- **`csv_sha256` mismatches warn rather than refuse** (`benchmark.require_cached_csv`).
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"`.** Timing
  tests follow the same pattern on `DS_AGENTS_TIMING_TESTS=1`; there are now two of them.
- **The cost cap is invocation-level; a failed run is charged its estimate.** A HALTED run is not a
  failed run: it returns a state and is charged what it spent.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Unchanged.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **`feature_eng.py`'s comment on the justification string is still wrong** for a future reader.
- **`harness.py` and `cli.py` still import each other inside functions.** Deferred again.
- **The loop-cap default `3` is written in three places.**
- **`cmd_run` has no per-run `try/except`.** Unchanged.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
