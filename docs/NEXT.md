# Next session

## Start here

**The README is installed and Phase 5 has two boxes left.** `README.md` is the writeup as a short
paper: setup, five results sections, design notes, limits, reproducing. Every table was regenerated
from the committed rows at install time and all of them matched. `docs/README-draft.md` is deleted.
`docs/readme_numbers.py` was **kept** rather than deleted, and the README now references it, because
regenerating the tables is what keeps a number from resting on prose in a log. Run it after adding
results and update whatever moved. The prose was rewritten plainly on request: no em-dashes, no
"not X but Y" constructions.

**Two $0 investigations ran off the committed rows and both landed.** Neither needed an API call.

**The loop-rate term is declined, and the argument is recorded.** The recorded next step was to add
one; it was worked out first and then rejected. Looping is piled into 3 of 13 datasets
(`australian` 3/4, `kr_vs_kp` 2/4, `sylvine` 1/4, and 0/4 on the other ten), and a goodness-of-fit
against one constant rate applied uniformly gives chi-square(12) = 27.5, p = 0.0065, so the data
reject a pooled rate on their own terms before sample size is argued. Applying one anyway moves the
error around: a global expected multiplier of 1.1431 predicts $1.2620 against the actual $1.2165
while charging the ten quiet datasets a 17.3% premium and still under-pricing the three noisy ones
at 71%. Instead: a contingency note beside the coefficients in `harness.py` ($0.05 to $0.15 on a
`full` run, held against the three named datasets and not smeared across thirteen cells), and the
two multipliers pinned in `tests/test_cost_model.py` as a record of 2026-09-02 and not a model of
the next run. No coefficient changed. DECISIONS.md 2026-09-21, second entry.

**The reviewer's dataset selectivity now has a mechanism, a falsification test, and a defect behind
it that is worth more than the answer.** Two structural facts: the reviewer is never shown
`verified_holdout_score`, `baseline_separation`, `baseline_normalised_score` or
`holdout_claim_gap`, so it cannot react to underperforming its own baseline and that whole family of
hypotheses is dead; and the base prompt admits an objection only if it names a column and cites a
number, with `_adjudicate` rejecting a column-scoped objection whose columns match nothing in the
profile. **`top_importances` is the only per-column number in the reviewer's entire input.**
H_concentration follows: a profiler nomination supplies the name (through the "Kept despite a
profiler flag" line) and a concentrated importance head supplies the number, and both are required.
Recomputed offline, columns needed to reach 80% of positive permutation importance is 1, 1, 3 on the
three objectors and 4+ on all ten silent datasets; the reviewer named the head and cut it at the
cliff; and all 4 `metric_mismatch` objections are on `australian`, on passes where the concentrated
column was already gone and nothing nameable was left. **It is a hypothesis, not a result**: post-hoc
on one ranking at n=13, p ~ 0.02 adjusted, and the importances are a proxy RandomForest rather than
the pipeline's own. DECISIONS.md 2026-09-21, third entry.

**The defect: the results row carries 83 columns and not one of them is a fact the reviewer is
shown.** `top_importances`, per-candidate `cv_mean` and `claimed_holdout_score`,
`chosen_model.name`, `dropped_features` and `feature_summary` all live on `PipelineState` and none
reached the row. The harness records what it measured while the reviewer adjudicates on a disjoint
set of facts, which is why the committed rows needed an offline reconstruction to say anything.
`top_importance_share`, `top_importance_n80` and a `top_importance_status` companion are now
emitted to close the part H_concentration needs, computed over the same 15 entries the reviewer is
shown. The emitted row goes 83 columns to 86. Sanity-checked against the five committed run
envelopes rather than a live run: `reissued_ids` 1.00 at n80=1, `claims_timing` 0.521 at n80=3,
`phoneme` 0.360 at n80=4, `kc1` 0.199 at n80=8. `kc1` is a silent dataset reading well above the
proposed threshold, which is weak independent support for the offline proxy.

The floor: **864 tests pass** (up from 854), ruff clean, toy green live at $0.0131 / 17.3s / first
pass. That toy run is also the first live emission of the three new columns, and it read
`status=ok`, `share=0.378`, `n80=3`, reproducing exactly what the committed toy envelope gives
offline. Total spend this session was the one toy run; both investigations were $0.

## First prompt

Read CLAUDE.md, `README.md`, and the "Start here" above.

**If anything is funded, spend the first $0.30 on the `claims_timing` reproduction check**, the
first open question below. It is the only item that threatens numbers already published in the
README, and it is cheap. Result 3 quotes those arms heavily, and there is recorded evidence in both
directions about whether the fixture still behaves the way those rows say. Settle it before
building anything on top of Result 3.

After that, the best value is the **redundancy ablation**, about $0.40, the falsification test for
H_concentration: build fixture variants of `australian` and `sylvine` under `tests/fixtures/` that
add three near-duplicate
copies of the dominant column, so permutation importance collapses while per-column NMI, row count,
class balance and categorical fraction all hold. H_concentration predicts the objection rate falls
from about 5/9 to roughly 0 on the duplicated arm. It is the only intervention identified that moves
this trigger without moving any other hypothesis. Pre-register the endpoints first. With
`top_importance_n80` now on the row, the result reads straight off the results file instead of
needing a reconstruction.

If more is funded, the two remaining Phase 5 boxes are reviewer on/off and single-agent-vs-team,
which need pre-registration per cell, and note that a leakage endpoint must run on the fixtures
because `--subset full`'s rows are `leakage_graded: false` 52/52. Re-running the quoted fixture
cells at HEAD closes the provenance gap at the same time.

## Open questions

- **Do the fixture arms still reproduce at HEAD? Still unresolved, and evidence points both ways.**
  The 2026-09-09 walkthrough session recorded that `claims_timing` no longer loops: both fresh
  replicates passed first-loop with zero objections and both planted leaks were already dropped by
  `feature_eng`'s own plan, against 10/10 committed rows that looped at older commits. The previous
  session's two toy runs went the other way, one exhausting three loops and one passing at once.
  **The README quotes `claims_timing` arms heavily.** If the fixture behaviour has genuinely
  shifted, several of Result 3's numbers describe a pipeline that no longer exists; if it is
  nondeterminism, they are fine and underpowered. A handful of `--repeat` runs on `claims_timing
  --naming opaque --reviewer-prompt which_column` answers it for about $0.30. **This is the largest
  standing risk to a number already published in the README**, and it was not settled before
  installing because nothing was funded.
- **H_concentration is untested.** See the first prompt. Note also what it does not explain: only 6
  of the 12 runs on those three datasets objected, and concentration is a dataset-level property, so
  the within-dataset coin flip is unaccounted for. The recorded sub-pattern is that a first pass
  which kept the concentrated column objected 5 times in 9, and one that dropped it unprompted
  objected once in 3 (using the column-free `metric_mismatch`). Directionally right, n=3 in the
  second cell, nothing decides it.
- **The reviewer and the profiler are two detectors that fail on opposite data.** The profiler reads
  univariate NMI and the reviewer reads multivariate permutation importance. They agree on an
  isolated strong column and diverge under redundancy, where importance collapses and NMI does not.
  `bank_marketing` is concentrated in importance and flat in NMI; `nomao` and `kc1` are NMI outliers
  and flat in importance. Nothing has been built on this yet and it may be the more useful half of
  the finding. See `docs/LEARNING.md`, [[two-statistics-one-intuition]].
- **Three more facts the reviewer sees are still unrecorded.** In priority order after the three
  columns just added: `top_importances` itself truncated to the reviewer's 15, which would make "did
  it name the head of the list" exact instead of inferred; a **per-pass feature count**, because
  `n_final_features` is post-loop and that is what confounds every first-pass-versus-post-loop score
  comparison; and `chosen_model_name` with per-candidate `cv_mean`.
- **`errored` is true on a run where the recovery WORKED.** Unchanged. Exactly 1 of 52 rows, a
  `kr_vs_kp` run whose two errors are the zero-objection block-retry firing and then succeeding. The
  taxonomy is not two-valued: a routine decision is not an error (fixed by reclassifying), a real
  anomaly with no column needs a companion column (fixed by adding one), and **a real anomaly that
  was handled is a third thing with no answer yet**.
- **`objections_raised` is 0 on 36 of 52 rows.** Those rows are in the control arm, so "not looking"
  has direct fixture evidence behind it and "nothing to object to" is the weaker hypothesis.
  H_concentration is now the third reading and the most specific one: the reviewer had nothing
  admissible to object with, because a flat importance list supplies no citable per-column number.
- **The `higgs` unforced drop did NOT reproduce.** 1 event in 8. Still the only measured case of an
  unforced drop costing real performance, still needs its own cell before it is quoted.
- **`register_dataset`'s two grader calls are still unbounded under BOTH transports.** Pinned by
  `test_only_one_of_the_three_registrations_is_bounded_by_anything`; the guard wants a watchdog
  thread or a signal, which is a design question, and the exposure is 1.2% of budget.
- **`bank_marketing`'s `V12` was kept by all four `full` runs and objected to by none.** 8 of 8
  across two files. The documented `recorded_after_outcome` column stays outside
  `planted_leakage_columns` deliberately. Worth rereading against H_concentration: `bank_marketing`
  is importance-concentrated (top-1 share 0.571) and flat in NMI, so the profiler never nominated
  it and the reviewer never got the name.
- **`kr_vs_kp`'s unit point is exactly 1.0000.** A tuned RandomForest solves it perfectly; the
  pipeline gets 0.9662. [[reference-system-independence]]'s price paid on a dataset with no trap.
- **200 withheld rows put roc_auc's standard error near 0.04** on the fixtures, which is wide next
  to several of the README's fixture differences. Unchanged.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
- **`published_reference` picks the max over uploaded runs**, not a protocol-stable anchor.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still half open.
- **No CI job and no thresholds.** Unchanged, and the reason is policy rather than code.
- **`exhausted` is still uninformative.** 4 real ones on manifest data to read.
- **Late detection against the cap** is the remaining structural reviewer defect, and the README
  names it as the one no prompt fixes.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's per-dataset numbers, and **two
  AMLB candidates could not be fetched at all** (`guillermo`, `Robert`).
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- **The system `git` is blocked by an unaccepted Xcode license** on this machine. Sessions commit
  through `/Users/cameronlaedtke/anaconda3/bin/git`. `sudo xcodebuild -license` fixes it; until then
  any tool shelling out to `/usr/bin/git` fails with a license notice rather than a git error.

## Parking lot

- **The provenance asymmetry is a recorded constraint and the README states it.** The 52 benchmark
  rows are uniform (83 columns, one commit). The 145 fixture rows carrying every leakage finding
  hold 36-62 columns and are missing `commit`, so no leakage number can be attributed to a tree. **No
  committed row anywhere carries `leakage_graded: true`** (the 85 that have the column are benchmark
  runs where it is correctly false; filter on non-empty `leakage_planted` instead), and the
  claimed-versus-verified check exists **only** on the benchmark side. DECISIONS.md 2026-09-21,
  first entry. The three `top_importance_*` columns widen the row to 86, which is normal for this
  schema but means future rows differ from the 52 by more than a commit. The README's provenance
  paragraph was corrected at install time: `src/` HAS changed since `70f7547` (capture tooling, a
  CLI flag, and now the three columns), and the earlier draft wording claimed it had not. No node
  and no part of the grading path changed, which is the claim that actually holds.
- **`docs/readme_numbers.py` is now a kept tool, not a temporary one.** Promote it into
  `src/ds_agents/` if it earns a CLI entry and tests; until then it is a docs-adjacent script that
  the README points at.
- **`capture._forced_drop_columns` duplicates `binding_objections`' predicate on purpose.**
  `test_state.py` pins `binding_objections` to one caller, so the walkthrough's replay reproduces
  the predicate via public helpers. Until `state.py` blesses a read-only path, two copies can drift.
- **Untested paths in `capture.py`, accepted as nits:** the reviewer-step crashed-pass headline
  fallback, `model_artifact` id recovery in `_cited_artifact_ids`, and `envelope()`'s
  SystemExit-on-missing-fixture path.
- **`n_candidates_failed_to_fit` has never fired.** 0 on 52/52 live rows. The column is shown to be
  quiet, not shown to work. Do not read a zero here as evidence the estimators are healthy.
- **The manifest prose rename is still deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`. The README deliberately quotes neither.
- **The retained cost model** is `$0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`,
  residual sd $0.0022 on 9 datasets and 3 parameters. Accurate to +/-8% on first-pass runs and
  silent about looping ones, with the contingency held as a note rather than a coefficient.
- **The toy run's 2.5x loop spread** has three observations and agrees with the manifest's x2.46.
  Still thin, and still the reason no toy cost should be quoted from one run.
- **H_categorical was refuted** with a real mechanism behind it, and it has no separating power on
  the objection question either.
- **`test_baseline_cost.py`'s synthetic bound is loose by about 1.6x on real data.**
- **The new cliff is 1,048,172 agent rows**, pinned by test.
- **The encoding narrows what the artifact can say, deliberately.**
- **`ModelResult.model_artifact` is still declared and never written.**
- **The sandbox is not a jail, and the docstrings say so.**
- **`.mcp.json` still hardcodes the toy dataset on argv.**
- **The `credit_g` withheld indices are pinned as a sha256 digest**; the toy split likewise.
- **`_withhold_rows` uses numpy directly rather than `train_test_split`.**
- **`csv_sha256` mismatches warn rather than refuse.**
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"`.** Timing
  tests follow the same pattern on `DS_AGENTS_TIMING_TESTS=1`.
- **The cost cap is invocation-level; a failed run is charged its estimate.**
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing**, and **`cmd_run`
  has no per-run `try/except`.** Both behaviour changes wanting their own decision.
- **`forced_drop_release` is closed to further use.**
- **`harness.py` and `cli.py` still import each other inside functions.**
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
