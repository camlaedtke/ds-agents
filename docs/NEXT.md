# Next session

## Start here

**The claims_timing reproduction check ran, and Result 3 reproduces at HEAD.** This was the recorded
largest standing risk to a published number and it is now closed.
`evals/results/2026-09-21_claims-repro.jsonl`, n=10, **$0.3006**, 0 refused, 0 failed, cost estimate
out by 0.2%. All four primary pre-registered endpoints reproduced: `reviewer_caught` 10/10 against
10/10, `leakage_remediated` 9/10 against 9/9 non-null, `review_loops` mean **2.30 against 2.30**,
and the falsification endpoint 0/10 for the second time. Full pre-registration and results in
`evals/results/LOG.md`, 2026-09-21.

**Half of that risk was retired for $0 before the run, and the lesson is worth more than the run.**
The contrary evidence was a conditions mismatch. `docs/explainers/capture_runs.py` invokes
`ds-agents run --dataset claims_timing` with no `--naming`, so the 2026-09-09 walkthrough's two
replicates are the **descriptive** arm, while every row in README Result 3 is **opaque**. The
descriptive arm's own committed rows pass first-loop 8 times in 10, so the walkthrough saw that
distribution's most likely outcome. Check which arm produced an alarming number before paying to
re-measure it. DECISIONS.md 2026-09-21, fourth entry.

**A new subset, `claims-repro`, and it shares the `Cell` object with `ci` rather than retyping its
conditions.** Pinned by a test asserting `is`, not `==`: a reproduction check whose conditions are
retyped is a check of the typing, and an edit to the `ci` cell would otherwise turn it silently
into a comparison of two different arms.

**`errored` went 2/10 to 4/10 and none of it is a regression.** All four are the same shape: a
column-scoped objection rejected for naming no surviving column, then the zero-objection block-retry
firing and succeeding. That is the recorded "`errored` is true on a run where the recovery WORKED"
open question, now at 5 of 20 in this cell. **No candidate-fit failure occurred at HEAD** -- the
baseline's one genuine error did not recur, so HEAD is the healthier of the two files.

**The unregistered finding, and it is bigger than what the run was for: the profiler's opaque-arm
recall has moved.** `profiler_caught` on `claims_timing --naming opaque` reads **2/10 (2026-08-27),
5/10 (2026-08-31), 8/10 (2026-09-21)**. chi-square(2) = 7.20, **p = 0.027**; first against last by
Fisher exact, p = 0.023. The movement is specific: `var_01`, a false alarm, is nominated in 29 of
those 30 runs and all the change is in how often the planted `var_08` appears beside it. Four
explanations were eliminated at $0: the profiler runs once, before the reviewer, with no
`graph.ROUTES` edge back to it, so the conditions that differ between those files are downstream of
it; `leakage_planted` and `random_seed` are identical; the intervening `naming.py` change is a pure
`Fixture`-to-`Runnable` refactor; the intervening `profiler.py` change touches `SPLIT_SNIPPET` and
the split-manifest encoder only, neither of which reaches `_user_message`. What is left is
model-side drift over 25 days, or chance at n=10. **It is a flag, not a result** -- post-hoc, n=10,
three points. DECISIONS.md 2026-09-21, sixth entry.

**It puts a published number in question.** README Result 2's "2 of 20 under opaque names, recall
0.10" is the 2026-08-27 cell; the same measurement at HEAD reads **8 of 20, recall 0.40**. The
direction survives comfortably (0.95 against 0.40 is still most of the finding) and the mechanism
argument does not rest on the rate. The README now carries the caveat and the three-row table
beside Result 2 rather than waiting for the replication.

**The provenance defect the run walked into, now fixed.** All 10 rows carry `commit: null` despite
being produced at a clean `46bd4ed` -- a hash asserted by this file and by `LOG.md`, not readable
off the rows, which is narrative provenance standing in for the real thing.
`provenance.git_commit()` shells out to bare `git`, which is `/usr/bin/git` under the unaccepted
Xcode license recorded in this file for several sessions, and
it returned `None` **silently**. Returning `None` is correct; the silence is the defect. Now:
`describe_commit()` returns `(commit, reason)`; `git_commit()` still never raises and prints the
reason to stderr once per process; the harness prints before the first run, because a null `commit`
fragments an `eval-diff` cell and is only actionable before the money is spent; and `DS_AGENTS_GIT`
names a git to use instead of the one on PATH. **Prefix eval invocations on this machine with
`DS_AGENTS_GIT=/Users/cameronlaedtke/anaconda3/bin/git` until `sudo xcodebuild -license` is run.**
Deliberately not done: falling back to searching PATH for a git that works. A provenance string from
an unknown binary is a worse record than a null, because a null is visibly missing.
DECISIONS.md 2026-09-21, fifth entry.

**The diff was reviewed: 3 blocking issues, 5 nits, all 8 fixed.** Two of the blockers had turned
the provenance fix into a smaller version of the bug it fixes. Narrowing `except Exception` to four
named subprocess exceptions looked like an improvement and dropped `UnicodeDecodeError`, a
`ValueError` that `text=True` decoding can raise, out of a function documented as unable to end a
run; and the new stderr `print` was itself unguarded, so a closed stderr would have made the
logging added to fix a silent failure into a loud one. The third was a missing test on the
harness's own warning, which now also pins that it is printed **before** the first run. The nit
worth repeating: the README asserted the reproduction ran at `46bd4ed`, a hash not readable off any
row in the file, which is the exact substitution `commit` exists to prevent. Every mention now says
the hash is asserted by the session record rather than traceable to the data.

**Two recorded gaps closed as side effects.** The claims-repro file holds the **first 10 rows
anywhere carrying `leakage_graded: true`** -- the parking-lot claim that no committed row does is
now out of date, though those rows carry no `commit`, so the provenance asymmetry is narrowed and
not closed. And the three `top_importance_*` columns got their second live emission: `claims_timing`
reads `status=ok`, `share=0.521`, `n80=3` on 7 of 10 rows, reproducing the committed envelope value
exactly.

The floor: **891 tests pass** in the full suite with 29 skipped, **798 of them fast** (up from
778 and 869 at session start), ruff clean. Toy green live at
the end of session: $0.0133, 16.2s, first pass, reading `status=ok`, `share=0.378`, `n80=3`, which
reproduces the committed toy envelope exactly and matches the previous session's first-pass run.
Spend this session was $0.3006 on the reproduction plus $0.0133 on the toy, about $0.314 total;
both investigations were $0.

## First prompt

Read CLAUDE.md, `README.md`, and the "Start here" above.

**If anything is funded, the first $0.30 goes to the naming-ablation replication.** It is now the
only item threatening a number in the README, it is the headline result of the project, and it is
cheap. Two cells differing in `naming` alone, `claims_timing`, n=10 each, at one commit, with
`DS_AGENTS_GIT` set so the rows carry it. Pre-register `profiler_recall` per arm; the 2026-08-27
values are descriptive 0.950 and opaque 0.100, and the open prediction is that the descriptive arm
holds while the opaque arm lands near 0.40. Add the cells as a `naming-repro` subset the same way
`claims-repro` was added, sharing objects where a cell already exists. **Do not edit README Result 2
until this runs** -- the caveat is honest and a rewritten number on n=10 post-hoc would not be.

Note what it also tests. If the opaque arm comes back near 0.10, the 8/10 was chance and the three
`claims_timing` files are one distribution. If it comes back near 0.40, the profiler's behaviour has
changed under a fixed prompt on a fixed fixture, and **that is a finding about agent pipelines in
general and not about this one**: every number this project has published rests on a model that is
not pinned, and nothing in the repo currently measures that.

After that, the **redundancy ablation** is still the best value at about $0.40: the falsification
test for H_concentration, fixture variants of `australian` and `sylvine` under `tests/fixtures/`
adding three near-duplicate copies of the dominant column so permutation importance collapses while
per-column NMI, row count, class balance and categorical fraction all hold. H_concentration predicts
the objection rate falls from about 5/9 to roughly 0 on the duplicated arm. Pre-register the
endpoints first. With `top_importance_n80` on the row the result reads straight off the results
file.

If more is funded, the two remaining Phase 5 boxes are reviewer on/off and single-agent-vs-team,
which need pre-registration per cell, and note that a leakage endpoint must run on the fixtures
because `--subset full`'s rows are `leakage_graded: false` 52/52.

## Open questions

- **Is the profiler's opaque recall drifting, or was 8/10 chance?** The first prompt. This is the
  new largest standing risk to a published number, and it replaces the `claims_timing` question
  that closed this session. Note the asymmetry in what the two outcomes cost: "chance" costs
  nothing and "drift" invalidates the comparability of every cross-date comparison in the repo.
- **Nothing in the repo pins the model behind `haiku`, and that is a deliberate decision with a
  comment.** `RunConfig.default_model` carries `"haiku"`; `tools/pricing.py`'s `ALIASES` resolves it
  to `claude-haiku-4-5`, which is what `NodeEvent.model` records, and that is an undated family
  alias rather than a snapshot. The comment above `ALIASES` gives the reason: "a config field
  reading `haiku` survives a model refresh and one reading a dated id does not." That is the right
  call for a **pricing** table and the wrong one for a **provenance** record, and the two are
  currently the same string. If the question above resolves to drift, this is the defect behind it:
  no committed row can say which model produced it. The fix is a second field carrying whatever
  dated id the API actually answered with, not a change to `ALIASES`. Check what the SDK response
  exposes before designing it.
- **The fixture arms other than `claims-opaque-which` are still unreproduced at HEAD.**
  `reissued_ids` and the toy cell were dropped from this run deliberately to buy n=10 on the cell
  that carried Result 3. `reissued_ids` carries Result 3's second half (the innocent-identifier
  finding, profiler nominates the control 10/10 and the trap 3/10) and has not been re-run since
  2026-08-31.
- **H_concentration is untested.** See the first prompt. What it does not explain: only 6 of the 12
  runs on those three datasets objected, and concentration is a dataset-level property, so the
  within-dataset coin flip is unaccounted for. The recorded sub-pattern is that a first pass which
  kept the concentrated column objected 5 times in 9, and one that dropped it unprompted objected
  once in 3. Directionally right, n=3 in the second cell, nothing decides it. **A free data point
  arrived this session and should not be over-read**: `claims_timing` opaque reads n80=3 and its
  reviewer objects 10/10, which is consistent, but it is a fixture under the `which_column` prompt
  and H_concentration is about the `base` prompt on benchmark data.
- **The reviewer and the profiler are two detectors that fail on opposite data.** The profiler reads
  univariate NMI and the reviewer reads multivariate permutation importance. They agree on an
  isolated strong column and diverge under redundancy, where importance collapses and NMI does not.
  `bank_marketing` is concentrated in importance and flat in NMI; `nomao` and `kc1` are NMI outliers
  and flat in importance. Nothing has been built on this and it may be the more useful half of the
  finding. See `docs/LEARNING.md`, [[two-statistics-one-intuition]].
- **Three more facts the reviewer sees are still unrecorded.** In priority order: `top_importances`
  itself truncated to the reviewer's 15, which would make "did it name the head of the list" exact
  instead of inferred; a **per-pass feature count**, because `n_final_features` is post-loop and
  that is what confounds every first-pass-versus-post-loop score comparison; and `chosen_model_name`
  with per-candidate `cv_mean`.
- **`errored` is true on a run where the recovery WORKED.** Now 5 of 20 rows in the
  `claims-opaque-which` cell, up from 1. The taxonomy is not two-valued: a routine decision is not
  an error (fixed by reclassifying), a real anomaly with no column needs a companion column (fixed
  by adding one), and **a real anomaly that was handled is a third thing with no answer yet**. It
  now has enough instances to be worth fixing rather than noting.
- **`objections_raised` is 0 on 36 of 52 benchmark rows.** Those rows are in the control arm, so
  "not looking" has direct fixture evidence behind it. H_concentration is the third and most
  specific reading: the reviewer had nothing admissible to object with, because a flat importance
  list supplies no citable per-column number.
- **The `higgs` unforced drop did NOT reproduce.** 1 event in 8. Still the only measured case of an
  unforced drop costing real performance, still needs its own cell before it is quoted.
- **`register_dataset`'s two grader calls are still unbounded under BOTH transports.** Pinned by
  `test_only_one_of_the_three_registrations_is_bounded_by_anything`; the guard wants a watchdog
  thread or a signal, which is a design question, and the exposure is 1.2% of budget.
- **`bank_marketing`'s `V12` was kept by all four `full` runs and objected to by none.** The
  documented `recorded_after_outcome` column stays outside `planted_leakage_columns` deliberately.
  Worth rereading against H_concentration: `bank_marketing` is importance-concentrated (top-1 share
  0.571) and flat in NMI, so the profiler never nominated it and the reviewer never got the name.
- **`kr_vs_kp`'s unit point is exactly 1.0000.** A tuned RandomForest solves it perfectly; the
  pipeline gets 0.9662. [[reference-system-independence]]'s price paid on a dataset with no trap.
- **200 withheld rows put roc_auc's standard error near 0.04** on the fixtures, which is wide next
  to several of the README's fixture differences.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
- **`published_reference` picks the max over uploaded runs**, not a protocol-stable anchor.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still half open.
- **No CI job and no thresholds.** The reason is policy rather than code.
- **`exhausted` is still uninformative.** 4 real ones on manifest data to read, plus 1 new one on
  `claims_timing` this session.
- **Late detection against the cap** is the remaining structural reviewer defect, and the README
  names it as the one no prompt fixes.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's per-dataset numbers, and **two
  AMLB candidates could not be fetched at all** (`guillermo`, `Robert`).
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- **The system `git` is blocked by an unaccepted Xcode license on this machine.** `sudo xcodebuild
  -license` fixes it. Until then, `provenance` says so on stderr instead of dropping the commit
  quietly, and `DS_AGENTS_GIT` is the per-session workaround. Sessions commit through
  `/Users/cameronlaedtke/anaconda3/bin/git`.

## Parking lot

- **The provenance asymmetry is narrower and still real.** The 52 benchmark rows are uniform (83
  columns, one commit). The 145 older fixture rows carrying most leakage findings hold 36-62 columns
  and are missing `commit`. **The 2026-09-21 claims-repro rows are the first anywhere with
  `leakage_graded: true`** (10 of them), which corrects the older claim that no committed row has
  it -- but they carry no `commit`, so they add a graded-leakage tree-attribution of exactly zero.
  Filter on non-empty `leakage_planted` for anything older. The claimed-versus-verified check still
  exists only on the benchmark side. The row is now 86 columns.
- **`docs/readme_numbers.py` is a kept tool, not a temporary one.** It picked the new results file
  up with no edit and independently reproduced this session's numbers, which is the argument for
  keeping it. Promote it into `src/ds_agents/` if it earns a CLI entry and tests.
- **`capture_runs.py` runs the descriptive arm and the walkthrough HTML does not say so.** That is
  what made a routine capture read as a contradiction of a published result for twelve days. A
  caption naming the arm on each captured run would have cost one line.
- **`capture._forced_drop_columns` duplicates `binding_objections`' predicate on purpose.** Until
  `state.py` blesses a read-only path, two copies can drift.
- **Untested paths in `capture.py`, accepted as nits:** the reviewer-step crashed-pass headline
  fallback, `model_artifact` id recovery in `_cited_artifact_ids`, and `envelope()`'s
  SystemExit-on-missing-fixture path.
- **`top_importance_n80`'s trailing `return len(positive)` is an untested defensive branch.**
  Unreachable in exact arithmetic; it exists as a floating-point guard.
- **`n_candidates_failed_to_fit` has never fired.** 0 on 52/52 benchmark rows and 0 on the 10 new
  ones. Quiet, not shown to work.
- **The manifest prose rename is still deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`.
- **The retained cost model** is `$0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`,
  residual sd $0.0022 on 9 datasets and 3 parameters, with the loop contingency held as a note
  rather than a coefficient. It predicted the claims-repro cell to within 0.2%, which is a fixture
  cell and not one of the 9 it was fit on.
- **The toy run's 2.5x loop spread** has four observations and keeps agreeing with the manifest's
  x2.46. Still thin, still the reason no toy cost should be quoted from one run.
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
