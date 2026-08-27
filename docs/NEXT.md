# Next session

## Start here
**Phase 3's central question got its first real answer, and the answer is that the reviewer misses.**
Two new trap fixtures shipped -- `claims_timing` (two columns recorded after the label) and
`reissued_ids` (an identifier re-keyed by outcome, with a genuine unique id alongside as a control)
-- plus `src/ds_agents/fixtures.py`, the registry that had to exist before a second dataset could be
run at all. Three things are worth knowing before touching anything.

First, **the traps were tuned on the wrong axis.** The premise was that the toy leak gets caught
every run because it screams statistically, so the new ones were built to sit inside the range a
legitimate feature occupies: `member_number` reads 0.0945 normalized mutual information against
`credit_score`'s 0.0952, which is as indistinguishable as a fixture can be made. It changed nothing.
Across 6 live Haiku runs the profiler nominated the planted column 6 of 6 and `feature_eng` dropped
it, and the reviewer got a clean matrix again. A name ablation says why: the same `claims_timing` CSV
with the traps renamed to `metric_a7` and `metric_b3` -- identical values, identical everything the
profiler computes -- was nominated in 1 of 3 runs against 3 of 3 for the descriptive names. n=3 a
side, so directional, not a rate. The profiler is reading column names, not the association number.

Second, **in the 3 runs where a trap did survive upstream, the reviewer raised nothing.** One live
`claims_timing` run where `adjuster_touches` survived at a claimed roc_auc of 0.954, and two ablation
runs where both traps survived as the number one and two permutation importances at a claimed 0.986
against a legitimate ceiling near 0.82. All three returned `claim: "pass"` with zero `leakage` or
`contamination` objections. It did not argue the columns were fine; it never mentioned them. Needs a
bigger n and a Sonnet arm before it goes in a README, but it is the first time the question has been
asked at all.

Third, **the `metric_mismatch` mystery from last session is solved and the objection is legitimate.**
Evidence text captured from three `reissued_ids` runs: the reviewer is comparing each candidate's
`cv_mean` against its `claimed_holdout_score` and objecting when they diverge implausibly --
"hist_gbdt shows cv_mean=0.7083 but claimed_holdout_score=0.83, a 0.1217 point gap (17% relative
uplift) on 160 train rows". That is a correct and sharp observation. What makes it end `exhausted`
every time is that the objection is about a candidate the modeler *rejected*, so there is nothing for
the modeler to change and no path to a `pass`. That is a different failure from a false alarm, and
the contract cannot currently tell them apart.

The floor: 298 tests pass in 43.6s full, 261 in 7.8s fast, ruff clean. `uv run ds-agents run
--dataset toy` is green -- leak dropped, `publishable: yes`, $0.0335, ended `exhausted` on the
metric_mismatch pattern above. New fixtures run at ~$0.014 / 18s (`claims_timing`) and ~$0.033 / 45s
(`reissued_ids`, which hits the cap every time).

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above, then continue Phase 3 in plan mode: make
**name transparency an explicit ablation axis** rather than an accident of what the fixture author
called a column. The finding above says trap difficulty is set mostly by the column name, which means
every leakage number this project plans to publish currently has an uncontrolled variable underneath
it. Concretely: decide how a fixture declares a naming condition (a `naming: descriptive | opaque`
field on the manifest, or a paired fixture, or a rename applied by the harness at load time -- the
third keeps the rows identical across arms, which is the only version that makes the comparison
clean), then re-run `claims_timing` both ways at n>=10 a side so the 3-of-3-versus-1-of-3 becomes a
rate. Also decide whether `evals/results/` should start taking JSONL rows now, before Phase 4's
harness: the numbers in this session's docs are prose over ad hoc runs and nothing committed backs
them.

## Open questions
- **How should a naming condition be represented?** A harness-applied rename keeps the rows byte
  identical across arms and is the only shape that isolates the variable, but it means the CSV on
  disk no longer matches what the agents saw, which complicates reproducing a single run. A paired
  fixture is reproducible and doubles the fixture count. Decide before building.
- **A legitimate objection with no available remedy is not a false alarm, and nothing records the
  difference.** The `metric_mismatch` objections are correct on the evidence but target a rejected
  candidate, so the modeler cannot satisfy them and the run always ends `exhausted`. Options: let
  the reviewer scope an objection to a candidate rather than to a node, teach the modeler to respond
  by dropping a candidate, or add an `unactionable` disposition. Until then every `reissued_ids` run
  burns 3 loops and ~$0.033 to reach the cap.
- **`results_row()` cannot see a profiler false alarm.** `false_alarm` counts columns on reviewer
  objections only. The profiler nominated `claim_ref` and `application_ref` -- genuine ids, not
  planted leakage -- in every run, and none of it appears in a results row. The same blind spot as
  the non-column objections noted last session, one node upstream.
- **Duplicate-rows-across-split is still unbuilt, and the reason is structural.** The reviewer never
  sees the split, the profile, or a row, and `results_row()` scores leakage as a set comparison over
  columns, so a trap with no guilty column yields `leakage_recall: None`. Needs a ground-truth shape
  and probably a `n_duplicate_rows` signal from the profiler before the fixture is worth writing.
- **`customer_id` as a false positive.** Unchanged for six sessions and now joined by `claim_ref` and
  `application_ref`, so it is no longer a toy-only quirk: every fixture ships an id column that gets
  correctly flagged and scored as a false alarm. Options unchanged (an `acceptable_flags` set, count
  id columns as planted, or publish and explain), but the cost of not deciding has tripled.
- **A refused model call loses its token accounting.** `AnthropicModel.generate` raises `ModelRefusal`
  before building the `Completion`, so the node event reads $0.00 for a call that was billed. Touches
  `llm.py`, `_run.py` and all six nodes.
- **LangSmith is wired but never exercised.** Treat "tracing works" as unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot
- **The trap fixtures are calibrated against a number that may not be the operative one.** Each
  manifest records `mutual_info_with_target`, and the generators were tuned to land it in a band.
  Given the naming finding, that number is closer to documentation than to a difficulty dial. Keep
  recording it; stop treating it as the knob.
- **A stub-model survival test is the only thing standing between a fixture and silent uselessness.**
  `tests/test_trap_survival.py` asserts each planted column reaches `final_features` *and*
  `top_importances` under `StubModel`. Without it a trap that upstream mechanically removes produces
  a `leakage_recall` of 0.0 that reads exactly like a reviewer miss. Any new fixture needs this.
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it.** Consistent with what `_toy_state`
  did before, but `SystemExit` is now doing application control flow in two places. A dedicated
  `FixtureNotFoundError` would be cleaner, and would let all three "do not run" paths in `cmd_run`
  agree on an exit code. Flagged in review, deliberately not taken.
- **`_toy_state` is now a wrapper around `_fixture_state(load_fixture("toy"))`**, kept because two
  test modules import it. There is an equivalence test pinning them together; migrating the callers
  and deleting it is a five-minute cleanup nobody has needed yet.
- **A per-item rule on a response schema is a whole-response rule.** Still worth checking `intake`
  and `modeler`'s LLM-facing schemas for the shape that took down a reviewer pass on 2026-08-27.
- **The split manifest is still embedded in snippet text**, and the 1 MiB read cap makes it urgent at
  Phase 4 sizes. Every artifact carries `sandbox_path`; the fix is `nodes/feature_eng.py`,
  `nodes/modeler.py` and their tests.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.** Buys a memory cap and a
  network block, neither of which matters while every snippet is templated by us.
- **The reviewer-off arm still runs the reviewer node** as a zero-cost no-op, so `node_trace` has a
  `reviewer` row in both arms. Anything comparing them must read `model`/`cost_usd`.
- **`--reviewer-model` exists and is unexercised.** It is the Haiku-vs-Sonnet ablation's mechanism,
  and the reviewer's 0-of-3 miss above is the first result that makes running it interesting.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate. Restrict
  to the best-by-CV candidate at Phase 4 sizes.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`, so a future
  `dict[str, SomeModel]` field would round-trip wrong. Fix is to dispatch on `get_origin`.
- `ArtifactStore` copies the dataset per run and chmods it 0444. Wasteful for a benchmark set.
- **`.mcp.json` hardcodes the toy dataset on argv.** Now that a registry exists this is a smaller fix
  than it was, but it is still wrong for Phase 5's generalist arm, which must face each dataset.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
