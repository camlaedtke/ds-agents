# Next session

## Start here
**Phase 4 is closed and `--subset full` has run. Phase 5 is the writeup, and it has its tables.**
52 rows, 13 cells x n=4, one commit (`70f7547`), **$1.2165** against a $1.1040 estimate and a $1.60
cap, 0 refused, 0 failed -- `evals/results/2026-09-02_full.jsonl`. Every pre-registered grader
endpoint held **52/52 with no tolerance**: `rescore_status` ok, `baseline_status` ok,
`baseline_zero_score` exactly 0.5, `refit_claim_gap` exactly 0.0, `baseline_recipe` `rf-v1`,
`halted_at` null. The four never-run datasets ran live exactly as the $0 offline pass said.

**The overrun is not a cost-model failure, and that is the session's main finding.** All of the
+10.2% sits in two of thirteen cells. Every one of the 46 runs that reviewed exactly once landed
within 0.95x-1.04x of its cell's price; `australian` (+81%) and `kr_vs_kp` (+87%) each sent 2 of 4
runs to `review_loops=3`. `kc1` -- also modelled, also never run, but four first-pass runs -- came
in at **-0.5%**, and that control is what makes the rest readable. Restricted to first-pass runs the
two overrunning cells read -7% and +7.7%. The error was not in the coefficients; it was in a
sentence printed above them in `SUBSETS`, correct and committed for two sessions, saying every price
assumes a run that passes first time. **The loop multiplier is now measured on manifest data for the
first time: x1.80 at two loops (n=2), x2.46 at three (n=4)**, against a standing "about 2.5x"
carried from the toy fixture. A 13-dataset refit is DECLINED -- the residuals are a bimodal split on
a variable the model does not contain, and what the estimate needs is a loop-rate term.

**The reviewer objects only on the four never-run datasets.** 0 objections across all 36 runs of the
nine previously-run datasets; 6 objecting runs out of the 16 new ones. First `exhausted` verdicts
ever recorded on a manifest dataset (4). And one `kr_vs_kp` row reproduces the
`actionable-objection` failure outside the trap fixtures for the first time: two
`implausible_importance` objections addressed to `modeler`, a node with no column lever, finishing
with `objected_columns_unremediated: ['bxqsq', 'rimmx', 'wknck']`.

**`ModelResult.fit_error` landed first, deliberately, so all 52 rows carry it.** Three columns --
`n_candidates`, `n_candidates_failed_to_fit`, `candidates_failed_to_fit`. The `PipelineError` was
LEFT IN PLACE: opposite repair to the cardinality skip, from an identical symptom.

The floor: **835 tests pass** (up from 822), ruff clean, toy pipeline green live twice at $0.0135
and $0.0128.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 5, and the "Start here" above. Phase 5's remaining boxes are
**reviewer on/off**, **single-agent-vs-team**, and the **README as a short paper**. The README is
the only one that needs no money and it is the phase's actual deliverable; the two ablations do.
Decide which before pricing anything. If an ablation: pre-register per cell first, and remember
that `--subset full`'s rows are `leakage_graded: false` 52/52, so a leakage endpoint must run on
the fixtures and not on the manifest.

## Open questions

- **`errored` is true on a run where the recovery WORKED.** New, and the third variant of this
  question in three sessions. Exactly 1 of 52 rows is `errored`, a `kr_vs_kp` run, and both its
  errors are the zero-objection block-retry firing and then succeeding -- reviewer claimed `block`
  with nothing actionable, was re-asked once, retry produced a usable objection. The taxonomy is not
  two-valued: a routine decision is not an error (cardinality skip, fixed by reclassifying), a real
  anomaly with no column needs a companion column (failed fit, fixed by adding one), and **a real
  anomaly that was handled is a third thing with no answer yet**. `errored` cannot say "and it was
  recovered". Left open deliberately rather than patched at the end of a session.
- **Why do `australian` and `kr_vs_kp` loop when the other eleven never do?** `australian` is the
  smallest dataset in the manifest (690 rows) and `kr_vs_kp` the most categorical (36 columns, all
  categorical, 73 one-hot levels). Nothing connects either fact to a reviewer that objects, and
  this is now the most interesting unexplained thing in the results file. $0 to investigate from the
  committed rows before anything is funded.
- **The cost estimate needs a loop-rate term and there are 6 looping runs to fit one from.** Do not
  re-fit the size model on 13 datasets; it would launder a known mechanism into a larger intercept.
- **`objections_raised` is 0 on 36 of 52 rows.** A reviewer that never objects is not obviously
  working, and this file cannot distinguish "nothing to object to" from "not looking". The nine
  datasets in question carry no planted trap, so there may genuinely be nothing -- but that is an
  assumption, not a measurement.
- **The `higgs` unforced drop did NOT reproduce.** Downgraded, and this was free. All four `full`
  runs kept 28 of 28 features and scored **0.8006 identically**, as did three of four in
  `bench-tall`. The 24-feature run at 0.7085 is now **1 event in 8**, not 1 in 4, and the other
  seven runs are byte-identical to each other. Still the only measured case of an unforced drop
  costing real performance, and still needs its own cell before it is quoted -- but it is no longer
  "the single most interesting open item", because the loop finding displaced it and the rate halved.
- **`register_dataset`'s two grader calls are still unbounded under BOTH transports.** Unchanged,
  and explicitly declined this session rather than forgotten. Pinned by
  `test_only_one_of_the_three_registrations_is_bounded_by_anything`; the guard wants a watchdog
  thread or a signal, which is a design question, and the exposure is 1.2% of budget.
- **`bank_marketing`'s `V12` was kept by all four `full` runs and objected to by none.** Now 8 of 8
  across two files. The documented `recorded_after_outcome` column stays outside
  `planted_leakage_columns` deliberately.
- **`kr_vs_kp`'s unit point is exactly 1.0000.** A tuned RandomForest on the raw frame solves it
  perfectly; the pipeline gets 0.9662. That is [[reference-system-independence]]'s price paid on a
  dataset with no trap in it, and it is worth a sentence in the README rather than a fix.
- **200 withheld rows put roc_auc's standard error near 0.04** on the fixtures. Unchanged.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
- **`published_reference` picks the max over uploaded runs**, not a protocol-stable anchor.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still half open.
- **No CI job and no thresholds.** Unchanged, and the reason is policy rather than code.
- **`exhausted` is still uninformative** -- though there are now 4 real ones on manifest data to
  read, which there were not before.
- **Late detection against the cap** is the remaining structural reviewer defect.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's own per-dataset numbers, and
  **two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`).
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- ~~**`ModelResult` has no field for a per-candidate `fit_error`.**~~ **CLOSED 2026-09-02.** The
  `rationale` half is NOT a gap and will not be built: the modeler's prose about its own choice is
  what [[measurement-independence]] says no number may come from, and nothing reads it.

## Parking lot

- **`n_candidates_failed_to_fit` has never fired.** 0 on 52/52 live rows and `n_candidates` is 2 on
  all of them. The column is shown to be quiet, not shown to work; its sensitivity rests on unit
  tests. Do not read a zero here as evidence the estimators are healthy until something has failed.
- **The manifest prose rename is still deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`; a refresh re-fetches every OpenML response and can move `published_reference`.
- **The retained cost model** is `$0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`,
  residual sd $0.0022 on 9 datasets and 3 parameters. Re-derived by `tests/test_cost_model.py`.
  Now known to be accurate to +/-8% on first-pass runs and silent about looping ones.
- **The toy run's 2.5x loop spread** is no longer only a toy observation: manifest data agrees at
  x2.46. Both remain thin.
- **H_categorical was refuted** with a real mechanism behind it. `kr_vs_kp` at 36 categorical
  columns was the softest number in `SUBSETS` and it overran -- for loop reasons, not dtype reasons,
  which is a second, independent failure to find the categorical term.
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
  has no per-run `try/except`.** Both left open again on 2026-09-02, explicitly declined rather
  than forgotten: they are behaviour changes wanting their own decision.
- **`forced_drop_release` is closed to further use.**
- **`harness.py` and `cli.py` still import each other inside functions.**
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
