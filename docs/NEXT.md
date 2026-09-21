# Next session

## Start here
**The README is drafted and Phase 5's last two boxes are still unfunded.** `docs/README-draft.md`
is the whole writeup as a short paper -- setup, five results sections, design notes, limits,
reproducing -- and `docs/readme_numbers.py` regenerates every table in it from the committed rows,
so no number in the draft rests on prose in a log. It is a draft in `docs/` on purpose: installing
it is `git mv` plus deleting the script, and that should happen after it is read, not before.

**Two claims changed during drafting, because the rows disagreed with the notes.** First, the
52-row coverage run is in the **control arm** -- `naming=descriptive`, `reviewer_prompt=base`,
`objection_routing=as_addressed`, `objection_closure=off`. Every reviewer number on the manifest
measures the shipped default, not the best arm this repo has found; the same configuration scores
1/10 on fixture data that definitely contains a trap. "0 objections in 36 runs" is therefore much
weaker evidence of a healthy reviewer than it reads as. Second, `reissued_ids` turns out to carry
the name-reading finding independently: the fixture plants a genuine unique identifier
(`application_ref`, `var_01` under opaque) beside the trap (`member_number`, `var_02`), and across
10 runs the profiler nominated **the innocent control 10/10 and the guilty column 3/10**. Two
columns that look identical to a name reader, differing only in what generated them, is exactly
the case it cannot handle -- and the fixture was built to contain it.

**The provenance asymmetry is now a recorded constraint rather than an unstated one.** The 52
benchmark rows are uniform: 83 columns, one commit, no `src/` change between the run and the
writeup. The 145 fixture rows carrying every leakage finding hold 36-62 columns and are missing
`commit`, so no leakage number can be attributed to a tree. Two consequences: **no committed row
anywhere carries `leakage_graded: true`** (the 85 that have the column are benchmark runs where it
is correctly false; filter on non-empty `leakage_planted` instead), and the claimed-versus-verified
check exists **only** on the benchmark side. Decision and the three options weighed: DECISIONS.md
2026-09-21.

**The 2026-09-09 walkthrough session was committed this session, not written this session.** It was
sitting uncommitted in the tree -- `capture.py`, `--state-json`, five captured runs, the
pipeline-walkthrough viewer -- with its DECISIONS and LEARNING entries already written. It is
`9e9f7a2` now.

The floor: **854 tests pass** (up from 835), ruff clean, toy green live twice -- and the two runs
were $0.0303/`exhausted`/3 loops and $0.0132/`pass`/1 loop, which is the 2.5x loop spread showing
up a third time rather than a regression.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 5, and the "Start here" above. Then read
`docs/README-draft.md` end to end -- it is the phase's deliverable and it has never been read by a
person. Two ways to go. **Free:** mark it up, install it over `README.md`, and spend what is left
on the two $0 investigations below (why `australian` and `kr_vs_kp` loop; the loop-rate term).
**Funded:** the last two Phase 5 boxes, reviewer on/off and single-agent-vs-team, which need
pre-registration per cell first -- and note that a leakage endpoint must run on the fixtures, since
`--subset full`'s rows are `leakage_graded: false` 52/52. If anything is funded, re-running the
quoted fixture cells at HEAD is the cheapest way to close the provenance gap at the same time.

## Open questions

- **Do the fixture arms still reproduce at HEAD? There is now evidence in both directions and it
  is unresolved.** The 2026-09-09 walkthrough session recorded that `claims_timing` no longer
  loops: both fresh replicates passed first-loop with zero objections and both planted leaks
  already dropped by `feature_eng`'s own plan, against 10/10 committed rows that looped at older
  commits. This session's two toy runs went the other way, one exhausting three loops and one
  passing at once. **The README draft quotes `claims_timing` arms heavily.** If the fixture
  behaviour has genuinely shifted, several of Result 3's numbers describe a pipeline that no longer
  exists; if it is nondeterminism, they are fine and underpowered. This is the first thing to
  settle before the draft is installed, and a handful of `--repeat` runs on `claims_timing --naming
  opaque --reviewer-prompt which_column` answers it for about $0.30.
- **Why do `australian` and `kr_vs_kp` loop when the other eleven never do?** Unchanged and still
  $0 from committed rows. `australian` is the smallest dataset in the manifest (690 rows) and
  `kr_vs_kp` the most categorical (36 columns, all categorical, 73 one-hot levels). Nothing
  connects either fact to a reviewer that objects. Still the most interesting unexplained thing in
  the results file, and now also the biggest hole in the draft's Result 5.
- **The cost estimate needs a loop-rate term and there are 6 looping runs to fit one from.** Do not
  re-fit the size model on 13 datasets; it would launder a known mechanism into a larger intercept.
- **`errored` is true on a run where the recovery WORKED.** Unchanged. Exactly 1 of 52 rows, a
  `kr_vs_kp` run whose two errors are the zero-objection block-retry firing and then succeeding.
  The taxonomy is not two-valued: a routine decision is not an error (fixed by reclassifying), a
  real anomaly with no column needs a companion column (fixed by adding one), and **a real anomaly
  that was handled is a third thing with no answer yet**.
- **`objections_raised` is 0 on 36 of 52 rows.** Sharpened rather than resolved: those rows are in
  the control arm, so the reading "not looking" now has direct fixture evidence behind it and
  "nothing to object to" is the weaker of the two hypotheses. Still not measured.
- **The `higgs` unforced drop did NOT reproduce.** 1 event in 8. Still the only measured case of an
  unforced drop costing real performance, still needs its own cell before it is quoted.
- **`register_dataset`'s two grader calls are still unbounded under BOTH transports.** Pinned by
  `test_only_one_of_the_three_registrations_is_bounded_by_anything`; the guard wants a watchdog
  thread or a signal, which is a design question, and the exposure is 1.2% of budget.
- **`bank_marketing`'s `V12` was kept by all four `full` runs and objected to by none.** 8 of 8
  across two files. The documented `recorded_after_outcome` column stays outside
  `planted_leakage_columns` deliberately.
- **`kr_vs_kp`'s unit point is exactly 1.0000.** A tuned RandomForest solves it perfectly; the
  pipeline gets 0.9662. [[reference-system-independence]]'s price paid on a dataset with no trap.
  It has its sentence in the draft.
- **200 withheld rows put roc_auc's standard error near 0.04** on the fixtures, which is wide next
  to several of the draft's fixture differences. Unchanged.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
- **`published_reference` picks the max over uploaded runs**, not a protocol-stable anchor.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still half open.
- **No CI job and no thresholds.** Unchanged, and the reason is policy rather than code.
- **`exhausted` is still uninformative.** 4 real ones on manifest data to read.
- **Late detection against the cap** is the remaining structural reviewer defect, and the draft
  names it as the one no prompt fixes.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's per-dataset numbers, and
  **two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`).
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- **The system `git` is blocked by an unaccepted Xcode license** on this machine. This session
  committed through `/Users/cameronlaedtke/anaconda3/bin/git`. `sudo xcodebuild -license` fixes it;
  until then any tool shelling out to `/usr/bin/git` fails with a license notice rather than a
  git error.

## Parking lot

- **`docs/README-draft.md` and `docs/readme_numbers.py` are both temporary.** The script exists so
  the draft's tables can be checked, not because the README needs regenerating on a schedule.
  Delete both when the draft is installed, or promote the script into `src/ds_agents/` if it turns
  out to be wanted.
- **`capture._forced_drop_columns` duplicates `binding_objections`' predicate on purpose.**
  `test_state.py` pins `binding_objections` to one caller, so the walkthrough's replay reproduces
  the predicate via public helpers. Whether a read-only consumer should get a blessed path instead
  of a copy is `state.py`'s call; until then two copies can drift.
- **Untested paths in `capture.py`, accepted as nits:** the reviewer-step crashed-pass headline
  fallback, `model_artifact` id recovery in `_cited_artifact_ids`, and `envelope()`'s
  SystemExit-on-missing-fixture path.
- **`n_candidates_failed_to_fit` has never fired.** 0 on 52/52 live rows. The column is shown to be
  quiet, not shown to work. Do not read a zero here as evidence the estimators are healthy.
- **The manifest prose rename is still deferred and must ride the next `datasets refresh`.**
  `benchmark.py`'s `BASELINE_DEFINITION.note` and `benchmark_build.py`'s `HEADER` still name
  `baseline_score`. The draft deliberately quotes neither.
- **The retained cost model** is `$0.010163 + $0.000230 * n_features + $0.005609 * (n_rows/1e5)`,
  residual sd $0.0022 on 9 datasets and 3 parameters. Accurate to +/-8% on first-pass runs and
  silent about looping ones.
- **The toy run's 2.5x loop spread** now has a third observation ($0.0303 vs $0.0132 at one commit,
  this session) and agrees with the manifest's x2.46. Still thin, and still the reason no toy cost
  should be quoted from one run.
- **H_categorical was refuted** with a real mechanism behind it.
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
