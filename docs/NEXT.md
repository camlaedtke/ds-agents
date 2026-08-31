# Next session

## Start here
**A dataset nobody in this repo wrote can now be run and graded.** `ds-agents run --dataset
credit_g` works, and so does `eval --subset bench-smoke`. 20% of the rows are carved out *before*
`run_pipeline` is called, above the tools boundary, and never enter `$DS_DATASET`, `split_artifact`
or the run's artifact store; `verified_holdout_score` is the promoted model measured on them.
Three new modules: `runnable.py`, `holdout.py`, `rescore.py`. `_fixture_state` became `_run_state`.

**The number is earned by a self-check, not asserted.** No fitted model is persisted anywhere
(`ModelResult.model_artifact` is still declared and never written), so grading means refitting --
from `modeler.CANDIDATE_SPECS`, never from `ModelResult.params`, which is a flat merge across
pipeline steps and is lossy by construction. Before the withheld rows are touched, the same
pipeline is fit on the agents' train split and scored on the agents' *own* holdout and compared to
what the modeler claimed there. That one comparison validates transform re-application, seed,
positive class, scorer sign and split parsing at once. **`refit_claim_gap` is exactly 0.0 on every
run so far**, offline and live.

**Sensitivity is proved by construction, not by the live rows.** `tests/test_rescore.py` builds a
dataset whose one informative column is informative in exactly the rows the agents get and pure
noise in the rows they are graded on: claimed 1.0, verified 0.45, gap 0.55. Without that, the
module would be a number generator with no evidence it would ever disagree with the agents.

**The live rows: 4 at $0.0673, `rescore_status` ok 4/4, claimed 0.7520 against verified 0.7476.
And all four are numerically identical in every column**, so the two replicates bought *no variance
estimate at all* -- four samples of one deterministic outcome. `credit_g` has no planted leak, the
reviewer raised zero objections and `feature_eng` dropped nothing, so no decision was available to
be made differently. The +0.0044 gap is a fifth of the pre-registered noise floor and is not
evidence the agents did not overstate themselves.

**Fixtures withhold nothing, by decision.** `withheld_fraction` is 0.0 for every fixture, so all
145 committed rows and every toy test are byte-for-byte unaffected. Carving a 200-row toy would
change what the agents see for no gain: planted leakage is a *column*, so a random holdout still
contains it.

The floor: **716 tests pass** (up from 681), 27 skipped, ruff clean, `uv run ds-agents run
--dataset toy` green live at $0.0127. Session spend ~$0.14 all in.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. **The remaining Phase 4 box is
`baseline_score`, and it needs a decision before it needs code.**

The zero point is free: a constant class-prior predictor scores exactly 0.5 roc_auc by
construction, so it is a correctness assertion on the re-scorer rather than a measurement. The unit
point is the problem. AMLB publishes the *convention* (normalise from constant predictor to tuned
RandomForest) but not a grid this repo can re-fetch -- `openml1.win.tue.nl` presents a self-signed
certificate -- so any recipe is **ours**, untraceable in a way nothing else in the manifest is.

**Settle the pooling hazard first, because it may mean `score_ratio` should not be published at
all.** The baseline is fit on every column, including a leak. On a labelled dataset a pipeline that
correctly drops the trap scores *below* a baseline that kept it, so `score_ratio < 1` is evidence
of good behaviour there and of bad behaviour on an unlabelled dataset -- the same column meaning
opposite things with nothing on the row to separate them. Options worth weighing: compute the
baseline on the post-`feature_eng` matrix instead of raw columns (changes what it measures);
publish `baseline_zero_score` and `baseline_unit_score` as separate columns and let a reader
normalise (honest, more work downstream); or gate `score_ratio` on `leakage_graded` the way the
nine leakage columns are gated. Pre-register whichever, then build.

After that, `--subset full` needs a measured cost per dataset. Do NOT extrapolate from
`bench-smoke`: its own estimate was wrong by more than a factor of two (0.040 guessed, 0.017
measured), and `higgs` is 98x `credit_g`'s row count.

## Open questions

- **Should `score_ratio` exist at all?** See the first prompt. It is the only column in the project
  whose meaning depends on a property of the dataset rather than on the run, and it is currently
  null everywhere, which is the cheapest moment to decide.
- **`credit_g` produced four identical rows. Is that the dataset or the pipeline?** The claim above
  is that it is the dataset -- no leak, no objections, no drops, so no decision to vary. Worth one
  cheap check on a second manifest dataset before anyone writes "the pipeline is deterministic on
  real data" anywhere. `australian` or `phoneme` are the next cheapest.
- **Is `amazon_employee_access` a real dataset or a degenerate one?** Unchanged, and now *cheap to
  answer*: the run path exists. Nine integer-encoded high-cardinality ID columns may produce the
  zero-signal run `min_usable_features` exists to prevent. One run settles it.
- **Is a partially-labelled dataset gradeable at all?** Unchanged. `bank_marketing`'s documented
  `V12` is still unscored, deliberately, because `planted_leakage_columns` is read as complete.
  Now testable against a real run rather than in the abstract.
- **200 withheld rows put roc_auc's standard error near 0.04**, so a `holdout_claim_gap` under
  ~0.08 is not distinguishable from noise at n=1. This is the "no 10-run count without a replicate"
  lesson arriving on a *continuous* column, and it binds every gap this project will publish.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.** It
  catches a model that overfits rows. `claims_timing`'s trap would survive it. Not a defect, but it
  bounds what `holdout_claim_gap` can ever mean and belongs in the writeup.
- **`published_reference` picks the max over uploaded runs**, which is not a protocol-stable
  anchor. Unchanged. `credit_g` reads 0.7889 against our 0.7476 on a different protocol -- plausible
  and lower, which is the sanity check it is good for and nothing more.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Unchanged, and now
  it has a companion: `rescore_status` would read `no_model` on such a row, so the two-columns fix
  exists for the score even though `review_verdict` itself is still wrong.
- **No CI job and no thresholds.** Unchanged.
- **`errored` needs a companion column.** Unchanged, and `rescore_status` is now the *second*
  worked example of what that fix looks like, after `leakage_graded`.
- **`exhausted` is still uninformative.** Unchanged.
- **Late detection against the cap** is the remaining structural reviewer defect. Unchanged.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's own per-dataset numbers.
- **Two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`, md5 mismatches from
  OpenML itself). Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **The sandbox is not a jail, and the docstrings say so rather than implying otherwise.** It has
  no filesystem namespace, so a snippet could open the withheld CSV by relative path. What the
  layout buys is an *assertable* property -- no file the run's store holds contains a withheld row,
  checked by test -- not isolation. Real isolation waits on the Docker backend behind `SandboxPool`.
- **`ModelResult.model_artifact` is still declared and never written.** The re-scorer refits
  instead, which is defensible and is now proved faithful to 1e-6. Persisting the fitted model
  would make the grader cheaper but would also make it grade a *pickle* rather than a recipe.
- **`cmd_run --repeat 1` now uses `run-0/` instead of collapsing into the invocation root.**
  Deliberate: containment is a property a test has to check, and it cannot check a layout that is
  sometimes one shape and sometimes another. No path reaches a results row.
- **`register_dataset` does an unbounded `pd.read_csv` + per-column `nunique` per run**, and now
  also copies the file. Free at `credit_g`'s 139 KB. **Measure before `higgs` at 46 MB** -- this is
  the parking-lot item most likely to become a real cost.
- **`.mcp.json` still hardcodes the toy dataset on argv.** Unchanged and now more visibly wrong.
- **`SUBSETS` has no `full` entry**, and `_resolve_subset`'s message has now been wrong twice for
  the same reason: the blocker keeps moving. It currently names `baseline_score`, and a test
  asserts it does NOT still name `verified_holdout_score`.
- **The `credit_g` withheld indices are pinned as a sha256 digest**, not 200 literals. Equally
  loud, and a diff nobody can read is a diff nobody checks.
- **`_withhold_rows` uses numpy directly rather than `train_test_split`** -- sklearn's stratified
  shuffling is not contracted stable across versions, and this is the one partition every headline
  number sits on.
- **`csv_sha256` mismatches warn rather than refuse** (`benchmark.require_cached_csv`), because the
  hash depends on pandas' float formatting. `RunConfig.dataset_hash` now carries the hash of the
  *agent-view* CSV instead, which is the bytes that actually ran.
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"`.**
  Unchanged.
- **The cost cap is invocation-level; a failed run is charged its estimate.** Unchanged.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Unchanged.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **`feature_eng.py`'s comment on the justification string is still wrong** for a future reader.
- **`harness.py` and `cli.py` still import each other inside functions.** Deferred again.
- **The loop-cap default `3` is written in three places.**
- **`cmd_run` has no per-run `try/except`.** Unchanged.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
