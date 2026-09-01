# Next session

## Start here
**`--subset full` waits on money and nothing else.** The last code blocker is gone: the split
manifest no longer lists row indices. It carries an `assignment` string of one character per agent
row -- `h` for holdout, a digit for the fold that row *validates* in -- and every other partition is
derived. `higgs` goes **2,690,410 B to 78,831 B, 7.5% of `read_artifact`'s cap**, with headroom to
about a million rows. All four datasets that could not previously complete a run now do.

**Why it was not a cap bump, which is the thing worth carrying.** The split JSON text is substituted
verbatim into **four** snippet sources -- `feature_eng`, `modeler`, and the grader's two bodies --
so the representation was never private to the artifact store. Raising `max_bytes` at the three read
sites would have put a 2.7 MB string into snippet source four times per run.

**Two guards make the derivation honest.** `fold_train: "complement"` is recorded IN the file and
checked against what the splitter actually returned on every run, so implementing `TaskSpec`'s
`temporal` strategy later cannot silently hand later folds training rows from the future. And
`decode_split` takes the frame length as a **required** argument: under an assignment string every
position is in range by construction, so shrinking the artifact disarmed both the `truncated` check
and each consumer's `0 <= i < len(df)` bound, and the replacement had to be deliberate. Fixing the
thing a guard was watching can disarm the guard.

**`recoverable` is read now, and that was the general defect.** It was written by thirteen call
sites and read by nothing in the graph. Every straight-line edge is `halt_or(destination)`; a fatal
refusal goes straight to `reporter` and no further node spends. The row is **still written** --
refusing it would delete exactly the hardest datasets from the results file, which is the same
failure wearing the opposite coat -- and carries `halted_at`, the node that refused. `publishable()`
is unchanged. Fourth instance of the companion-column fix, and it closes the standing "`errored`
needs a companion column" item.

**Proof, cheapest first.** All four formerly-broken datasets ran end to end offline with `--no-live`
for **$0** before anything was funded. The size projection in `tests/test_split_manifest_size.py` is
within **0.02%** of the artifacts measured on disk. The toy run's `cv_scores` are byte-identical to
the pre-change commit, so the encoding is a numeric no-op -- the fold-ordering change was landed as
its own commit (`745614a`) first, exactly so that could be said. And `adult` produced the first
graded row of the four, live at **$0.0158**: `rescore_status` ok, `baseline_status` ok,
`refit_claim_gap` exactly 0.0, `baseline_zero_score` exactly 0.5, verified 0.9244 against a claimed
0.9240. `evals/results/2026-09-01_adult-smoke.jsonl`.

The floor: **783 tests pass** (up from 751, plus two new modules), ruff clean, toy pipeline green
live at $0.0322 (it looped to the cap, which is the fixture behaving as designed).

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. **The next task is pricing the
four datasets that just became runnable, which is the only thing between here and `--subset full`.**

`bench-mid` measured a cost model of roughly `a + b x n_features` (5 cols $0.011, 9 cols $0.013,
118 cols $0.041, 144 cols $0.042). It predicts $0.014 for `adult` at 14 columns; the one live run
came in at **$0.0158, 13% high**. That is n=1 and the model has never been checked on a dataset
wider than what it was fitted on, so the arm is worth running rather than extrapolating -- follow
/run-eval and pre-register the prediction per cell before spending.

The other three are `bank_marketing` (16 cols), `numerai28_6` (21 cols), `higgs` (28 cols). Money is
not the risk; **time is**, and it is concentrated in `higgs`:
1. `MODEL_TIMEOUT_S` is 240s and was already 45% consumed at 144 columns on `jasmine`. `higgs` is
   28 columns but 98,050 rows, and `permutation_importance` is `10 x n_columns` scoring passes per
   candidate over the holdout.
2. `register_dataset` does an unbounded `pd.read_csv` plus a per-column `nunique` and a file copy,
   **per run**, on a 46 MB CSV. Never measured. This is the session to measure it.
3. The baseline's RandomForest was quoted at 72s on `higgs`; `tests/test_baseline_cost.py`
   (`DS_AGENTS_TIMING_TESTS=1`) re-derived it at the correct shape and says the slowest shape uses
   6.4% of the 900s timeout, so this is the least of the three.
The offline `--no-live` runs are free and take under 30s each; run them first at any shape you are
unsure about.

## Open questions

- **Is `halted_at` the right shape for `eval-diff`?** It is a node name or `None`, so it cannot be a
  rate the way `errored` is -- `DEFAULT_METRICS` is boolean-rate machinery. A cell where 3 of 10
  runs halted is exactly the thing a table must not hide, and today it would only show as
  `errored`. Either `--metrics` needs to accept a null/non-null predicate or there should be a
  `halted` boolean beside the name. Not decided.
- **`baseline_normalised_score` is 1.054 on `adult`** -- the first row anywhere in this project
  above the raw-column floor. Consistent with the standing note that the unit point is a floor and
  not a ceiling, and not quotable: n=1, the grid is ours, and `feature_eng` dropped `native-country`
  as too high-cardinality to one-hot while the grader's encoder kept it. Same floor-vs-pipeline
  asymmetry `amazon_employee_access` showed, sign reversed. Two datasets now bracket 1.0, which
  makes "is a floor the right unit point at all" a sharper question for the writeup, not a duller one.
- **Fold ORDER is gone and cannot come back.** The assignment array carries membership only. It cost
  nothing measurable -- `CANDIDATE_SPECS` holds no order-sensitive estimator and the baseline's
  RandomForest fits on `train`, which was already sorted -- but a future bootstrap or subsample
  candidate would make the manifest under-specify what was fitted.
- **`credit_g` produced four identical rows again, at a second commit.** Unchanged. Still needs one
  cheap run on a second manifest dataset before anyone writes "the pipeline is deterministic on real
  data" -- and note `nomao` already refuted the general claim on 2026-09-01, so what is left is the
  narrower "identical where no decision was available to be made differently".
- **Is a partially-labelled dataset gradeable at all?** Unchanged. `bank_marketing`'s documented
  `V12` is still unscored, deliberately. Note this is now a dataset that will actually be RUN next
  session, so the question stops being hypothetical.
- **200 withheld rows put roc_auc's standard error near 0.04**, so a `holdout_claim_gap` under
  ~0.08 is not distinguishable from noise at n=1. Unchanged, and it binds every gap published here.
  `adult` withholds 9,768 rows, so its -0.0004 gap is a much tighter number than `credit_g`'s.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
  Unchanged. `claims_timing`'s trap would survive it.
- **`published_reference` picks the max over uploaded runs**, which is not a protocol-stable anchor.
  Unchanged.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** **Half closed.** On
  the halt path the router never runs, so the verdict stays `pending` -- never adjudicated is not the
  same as adjudicated and cleared. Still wrong on the zero-objection `block` path, which is a
  different bug.
- **No CI job and no thresholds.** Unchanged.
- **`exhausted` is still uninformative.** Unchanged.
- **Late detection against the cap** is the remaining structural reviewer defect. Unchanged.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's own per-dataset numbers.
- **Two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`). Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.
- ~~**`errored` needs a companion column.**~~ **CLOSED 2026-09-01** by `halted_at`, the fourth
  instance after `leakage_graded`, `rescore_status` and `baseline_status`.

## Parking lot

- **The manifest prose rename is still deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`; both are rendered verbatim into `evals/datasets/manifest.yaml`, which only
  `ds-agents datasets refresh` may write, and a refresh re-fetches every OpenML response and can
  move `published_reference`.
- **The new cliff is 1,048,172 agent rows**, pinned by `test_the_new_cliff_sits_around_a_million_
  agent_rows`. One byte per row plus a ~380-byte header against a 1 MiB cap, so the number is nearly
  the cap itself. Nothing in the manifest is within two orders of magnitude of it.
- **The encoding narrows what the artifact can say, deliberately.** There is no character for a row
  in no partition, so a split that discards rows is unrepresentable and the encoder raises rather
  than writing a manifest with a hole in it. Two unit-test fixtures had to be reshaped by this.
- **`store.path_of` still has no caller anywhere in `src/` or `mcp_server/`.** Its docstring is
  corrected but it remains dead code, and the split manifest is no longer the argument for it.
- **The unit point is a floor, not a ceiling, and that is deliberate.** Unchanged, and `adult` is
  now the first row on the other side of it.
- **`ModelResult.model_artifact` is still declared and never written.** Unchanged.
- **The sandbox is not a jail, and the docstrings say so.** Real isolation waits on the Docker
  backend behind `SandboxPool`.
- **`.mcp.json` still hardcodes the toy dataset on argv.** Unchanged.
- **`SUBSETS` has no `full` entry**, and `_resolve_subset`'s message has now been corrected a fourth
  time for the same reason: the blocker keeps moving. It names cost alone again.
- **The `credit_g` withheld indices are pinned as a sha256 digest**; the toy split is now pinned the
  same way as `TOY_ASSIGNMENT_DIGEST` in `tests/test_toy_pipeline.py`.
- **`_withhold_rows` uses numpy directly rather than `train_test_split`** -- and the same argument
  now rules out re-deriving the split from the seed inside each snippet.
- **`csv_sha256` mismatches warn rather than refuse** (`benchmark.require_cached_csv`).
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"`.**
- **The cost cap is invocation-level; a failed run is charged its estimate.** Unchanged. Note a
  HALTED run is not a failed run: it returns a state, is charged what it actually spent, and now
  spends far less than it used to.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Unchanged.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **`feature_eng.py`'s comment on the justification string is still wrong** for a future reader.
- **`harness.py` and `cli.py` still import each other inside functions.** Deferred again.
- **The loop-cap default `3` is written in three places.**
- **`cmd_run` has no per-run `try/except`.** Unchanged.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
