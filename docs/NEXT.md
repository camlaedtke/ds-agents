# Next session

## Start here
**The three live defects are closed, and `--subset full` is the only thing left before Phase 5.**
Asked and answered on 2026-09-02: not yet, fix the bugs first. That was the right order for a
reason that is now concrete -- running `full` a day earlier would have written 52 rows carrying a
known-wrong `errored` and no `baseline_separation`, and this repo does not edit committed results
files. Running `full` properly is still `--replicates 2 --n 2` = **52 runs, about $1.10 and 45-70
minutes**, and it is still a decision for a person rather than a technical question.

**`errored` is now a rate a reader can trust.** `feature_eng` was recording a routine one-hot
cardinality skip as a `PipelineError`, and `errored` is `bool(self.errors)`. The fix was upstream of
`errored`, whose definition never changed: the skip is a `skipped_high_cardinality` list on the
state and an `n_skipped_high_cardinality` count on the row. Checked against the rows rather than
asserted -- all four `adult` rows in `bench-tall` carry that error **and no other**, so the defect
accounts for 100% of the wrong value and all four read `errored: false` under the new code. The
other twenty-one `PipelineError` sites were read and **none was touched**: every one is a genuine
anomaly, which is what makes this a rule about one site rather than one the codebase broke
everywhere.

**`baseline_separation` puts the denominator on the row.** `numerai28_6`'s 2.089 now ships next to
the 0.0101 that explains it. Computed across every committed baseline row, `numerai28_6` is **26x
narrower than the next narrowest** (`credit_g`, 0.266) -- a visible singleton rather than one end of
a spectrum, so the number can be quoted with its explanation instead of dropped from a table for
looking wrong. Not a `baseline_status` value and not a suppression, both rejected for reasons in
DECISIONS.md.

**`eval-diff --metrics` exists, and accepts `column:notnull`.** `compare()` had taken a `metrics`
argument for its whole life with no way to pass one. Run on the two newest results files it puts the
first defect on screen beside its own diagnosis: `adult` reads `errored: 4/4` next to
`halted_at:notnull: 0/4`.

The floor: **822 tests pass** (up from 799), ruff clean, toy pipeline green live twice at $0.0132.

**Three things this session found that the last one had recorded wrong.** The `errored` blast radius
is **five** committed rows, not four -- `2026-09-01_adult-smoke.jsonl` carries a fifth, missed
because the defect was found by reading the bench-tall cell rather than by searching the corpus.
`DEFAULT_LOOP_CAP` was written in **four** places, not three; the fourth was `--loop-cap`'s own
argparse default, literal included in its help text. And the `SystemExit` path **was** already
tested, by `test_a_systemexit_propagates_instead_of_counting_as_a_failure`.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 5, and the "Start here" above. **Ask Cameron whether to spend
~$1.10 on `--subset full`.** If yes, follow /run-eval: `--subset full --name full --replicates 2
--n 2 --max-cost-usd 1.60`, pre-register per cell first, and expect the four never-run datasets
(`australian`, `kc1`, `sylvine`, `kr_vs_kp`) to be the interesting ones. Phase 5's untouched arms
are reviewer on/off and single-agent-vs-team.

## Open questions

- **A fully numeric dataset DID have a decision available, and it cost 0.09 roc_auc.** One `higgs`
  run of four kept 24 of 28 features with no objection raised -- `feature_eng` proposed the drops
  itself -- and scored 0.7085 verified against 0.8006 for the other three,
  `baseline_normalised_score` 0.716 against 1.031. This refutes the pre-registered claim that
  numeric datasets have nothing to decide differently, and it is the first measured case of an
  unforced drop costing real performance. **Needs its own cell before it is quoted**, which needs
  money. The single most interesting open item here.
- **`register_dataset`'s two grader calls are still unbounded under BOTH transports.** Only the
  graph's registration under `--tools mcp` sees `_CONNECT_TIMEOUT_S`; `rescore` and `baseline` build
  `LocalTools` directly, and `--tools local` bypasses it too. A hang there does not raise, does not
  abort, and appears in no column. **Half closed on 2026-09-02**: the asymmetry is now pinned by
  `test_only_one_of_the_three_registrations_is_bounded_by_anything`, which fails if a fourth
  unbounded site appears or if one of the two is bounded without the other. The guard itself is
  still open -- a timeout on a synchronous in-process call means a watchdog thread or a signal, and
  the exposure is 1.2% of budget, so it wants designing rather than bolting on.
- **The cost of a run that loops is still unmeasured on any manifest dataset.** All 16 runs raised
  zero objections and looped once. A run that loops three times costs about 2.5x, and `full`'s
  $1.10 assumes 52 runs that all pass first time.
- **The four never-run datasets.** `australian`, `kc1`, `sylvine`, `kr_vs_kp` have prices but no
  runs. `kr_vs_kp` is 36 columns, all categorical, 73 one-hot levels -- H_categorical was refuted at
  7 to 13 categorical columns, which is not refuted at 36, so its $0.022 is the softest number in
  `SUBSETS`. It is also the dataset most likely to exercise `n_skipped_high_cardinality`.
- **`bank_marketing`'s `V12` was kept by all four runs and objected to by none.** The documented
  `recorded_after_outcome` column survives to `final_features` every time, and it stays outside
  `planted_leakage_columns` deliberately.
- **200 withheld rows put roc_auc's standard error near 0.04.** Unchanged, and it binds every gap
  on the fixtures. The four new datasets withhold 9,042 to 19,609 rows, so their gaps are much
  tighter numbers.
- **The withheld holdout is a random split, so it cannot catch a temporal or grouped leak.**
  `claims_timing`'s trap would survive it.
- **`published_reference` picks the max over uploaded runs**, which is not a protocol-stable anchor.
- **A run produced no model at all and was recorded `review_verdict: "pass"`.** Still half open: the
  zero-objection `block` path.
- **No CI job and no thresholds.** Unchanged, and the reason is policy rather than code.
- **`exhausted` is still uninformative.** Unchanged.
- **Late detection against the cap** is the remaining structural reviewer defect. Unchanged.
- **The AMLB self-signed certificate** still blocks vendoring AMLB's own per-dataset numbers.
- **Two AMLB candidates could not be fetched at all** (`guillermo`, `Robert`). Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`. The
  `fit_error` half is a real gap: `modeler.py:448` raises a `PipelineError` naming it, so the fact
  reaches `errors` as prose but never reaches a column.
- ~~**`errored` needs a companion column.**~~ **CLOSED 2026-09-02** -- and the answer was that it
  needed no column. See below.

## Closed on 2026-09-02

- ~~**`errored` is TRUE on four completely healthy runs.**~~ Closed by making the skip a decision
  rather than an error. Five committed rows stay wrong permanently and cannot be edited: four in
  `2026-09-02_bench-tall.jsonl` and one in `2026-09-01_adult-smoke.jsonl` -- **every row `adult` has
  ever produced here.** Pinned by count, not banned by string, because a ban would require the edit
  the rule forbids. Any table quoting `errored` for `adult` needs that footnote.
- ~~**`baseline_normalised_score` divides by a span that can approach zero.**~~ Closed by
  `baseline_separation`.
- ~~**Is `halted_at` the right shape for `eval-diff`?**~~ Closed by `column:notnull` / `column:isnull`
  and a `--metrics` flag. The shape was fine; the tally had no way to ask about it.
- ~~**`feature_eng.py`'s comment on the justification string is wrong.**~~ It named two places the
  string does not go. It reaches the model's PROMPT via `already_dropped`, and `feature_summary`.
- ~~**The loop-cap default `3` is written in three places.**~~ Four, now one `DEFAULT_LOOP_CAP`.
- ~~**`store.path_of` is dead code.**~~ Closed as a **decision to keep it**. Its docstring is the
  record of why the split manifest stays reachable through `read_artifact` rather than behind
  `sandbox_path`, and its three test callers are the only exercise `_paths` gets. Deleting three
  lines would have deleted the reasoning.

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
- **H_categorical was refuted** with a real mechanism behind it and prior evidence pointing its way.
  A confound is a reason to run the experiment, not a result.
- **The toy run's 2.5x loop spread**, measured twice at one commit ($0.0304 and $0.0135). Every
  price in `SUBSETS` assumes it away.
- **`test_baseline_cost.py`'s synthetic bound is loose by about 1.6x on real data.** `higgs` measured
  36.2-36.8s live against the table's 57.18s. The table says it is an upper bound and it is.
- **The new cliff is 1,048,172 agent rows**, pinned by test. Nothing in the manifest is within two
  orders of magnitude.
- **The encoding narrows what the artifact can say, deliberately.** A split that discards rows is
  unrepresentable and the encoder raises rather than writing a manifest with a hole in it.
- **The unit point is a floor, not a ceiling** -- and as of 2026-09-02 that framing is backed by a
  column rather than by prose. `baseline_separation` is where a reader checks a value above 1.0.
- **`ModelResult.model_artifact` is still declared and never written.** Unchanged.
- **The sandbox is not a jail, and the docstrings say so.** Real isolation waits on the Docker
  backend behind `SandboxPool`.
- **`.mcp.json` still hardcodes the toy dataset on argv.** Unchanged.
- **`SUBSETS` HAS a `full` entry**, and `_resolve_subset` has no special case for it. The message
  that was corrected four times is gone rather than corrected a fifth time.
- **The `credit_g` withheld indices are pinned as a sha256 digest**; the toy split likewise as
  `TOY_ASSIGNMENT_DIGEST`.
- **`_withhold_rows` uses numpy directly rather than `train_test_split`.**
- **`csv_sha256` mismatches warn rather than refuse** (`benchmark.require_cached_csv`).
- **`network` tests are gated on `DS_AGENTS_NETWORK_TESTS=1`, NOT on `-m "not network"`.** Timing
  tests follow the same pattern on `DS_AGENTS_TIMING_TESTS=1`; there are now two of them.
- **The cost cap is invocation-level; a failed run is charged its estimate.** A HALTED run is not a
  failed run: it returns a state and is charged what it spent.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Left open
  deliberately on 2026-09-02 as a behaviour change wanting its own decision.
- **`cmd_run` has no per-run `try/except`.** Same -- left open as a behaviour change, not swept in
  with the column work. `run_eval` already has one, and its tests describe this as the parking-lot
  fix it was modelled on.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **`harness.py` and `cli.py` still import each other inside functions.** Deferred again, and
  `cli._build_parser` now does the same to `evaldiff` for `DEFAULT_METRICS`.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
