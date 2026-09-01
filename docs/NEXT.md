# Next session

## Start here
**`--subset full` is blocked on code again, and the money question is answered.** Nine of the
thirteen datasets are priced. The other four -- `adult`, `bank_marketing`, `higgs`, `numerai28_6` --
**cannot complete a run at all**, and nobody knew because nobody had run one.

**The mechanism, because it is the most useful thing here.** The profiler writes the train/test
split as a JSON artifact holding every row index -- `train`, `holdout`, and five folds of
(train, valid), so roughly six times the agent row count in integers -- and `read_artifact` caps
every read at `DEFAULT_READ_BYTES = 1 MiB`. Above roughly 36k rows those collide. `feature_eng` then
refuses on the truncated read with `recoverable=False`, which is correct. But **nothing in the graph
or the router reads `recoverable`**, so the run continues through reviewer and reporter, spends a
full run's tokens, and `publishable()` accepts it. A dataset that cannot be run does not look like a
failure. It looks like a measurement. Pinned by `tests/test_split_manifest_size.py`, which projects
the size from `n_rows` with no CSV read and was proved to fire by temporarily adding `adult`.

**The pre-registered cost model was half wrong, and the half that was wrong is the one this phase
has been worrying about.** Prediction: tokens track columns, wall clock tracks rows. Tokens track
columns almost exactly -- `jasmine` and `nomao` cost **within 2.4%** of each other across an 11.5x
row difference. But wall clock tracks columns too: `jasmine` (2984 rows, 144 cols) is the **slowest
cell in the run at 119.6s**, beating `nomao` at 11.5x its size, while `amazon` (32769 rows, 9 cols)
finished in 19.3s. `node_seconds` names it: the modeler is **108.2s at 144 columns against 8.0s at
9**, because `permutation_importance` is `10 x n_columns` scoring passes per candidate. **Price
`full` as `a + b x n_features`, one term.**

**So the baseline was the wrong thing to worry about.** `baseline_seconds` is the one row-driven
term measured here and it is **never more than 4% of a run** (0.4s at `jasmine`, 4.2s at `nomao`).
The 72s-on-`higgs` figure that motivated deferring `score_ratio` and versioning `baseline_recipe`
was real but not binding. The binding constraint is `MODEL_TIMEOUT_S`, already 45% consumed at 144
columns, and `SELECTION_RULE.max_features = 200` -- set without a measurement -- is what stands
between the manifest and a timeout.

**The old timing table was also measured at the wrong shape.** The baseline fits `SPLIT["train"]`,
which is 0.64 x `n_rows`, not `n_rows`. `tests/test_baseline_cost.py`
(`DS_AGENTS_TIMING_TESTS=1`) re-derives it at the correct shapes and answers an open question: the
slowest shape in the manifest uses **6.4% of the 900s timeout**, so `n_estimators` does NOT need to
drop and `baseline_recipe` does NOT need to fork.

Live: **8 rows at $0.2120** against a $0.35 estimate, 0 failed, `rescore_status` and
`baseline_status` ok 8/8, `baseline_zero_score` exactly 0.5 on 8/8, `refit_claim_gap` exactly 0.0 on
8/8. `evals/results/2026-09-01_bench-mid.jsonl`. The floor: **751 tests pass** (up from 741, plus
two new opt-in modules), ruff clean, toy pipeline green.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. **The next task is making the
split manifest fit, because it is the only thing between here and `--subset full`.**

It is NOT a cap bump. Three things make it a design change worth a session:
1. The whole split JSON is substituted into the modeler's snippet SOURCE
   (`nodes/modeler.py`, `split_json=split_payload.content`), so the representation is not private to
   the artifact store.
2. `credit_g`'s withheld indices are pinned in a test as a sha256 digest, and `tests/test_holdout.py`
   asserts properties of the current form.
3. Whatever replaces it has to keep the property that makes the split auditable -- a reader must
   still be able to check which rows were trained on.

Options worth costing before picking one: raise `max_bytes` only at the three call sites that read
this artifact (smallest change, but the 1 MiB cap exists because artifacts get rendered into prompts
and snippet source, and this one is 2.7 MB at `higgs`); store the split as a run-length or
fold-assignment array rather than explicit index lists (`n_rows` bytes instead of ~6x, and it stays
readable); or re-derive the split inside each snippet from the seed rather than passing it (smallest
artifact, but then the split is no longer a recorded object and auditability is lost).

**Second, and cheap: `recoverable` is written and never read.** That is what turned a correct refusal
into a paid-for row. Worth fixing independently of the split, because it is the general defect and
the split is only the instance that exposed it.

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
- ~~**Is `amazon_employee_access` a real dataset or a degenerate one?**~~ **ANSWERED 2026-09-01:
  real, and it is the most interesting of the four.** It graded cleanly -- `rescore_status` ok,
  verified 0.8126 -- and returned the **lowest `baseline_normalised_score` of the run at 0.8751**.
  It is also the only cell where `feature_eng` dropped columns the grader kept (7 final features
  from 9), which is the floor-vs-pipeline asymmetry the parking lot describes reaching a results
  file for the first time. Its nine integer-encoded ID columns do pass straight through the
  baseline's numeric branch, so the yardstick keeps identifiers the pipeline discarded. That is the
  concrete case the writeup's "is a floor the right unit point at all" question needs.
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
