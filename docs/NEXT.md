# Next session

## Start here
**Phase 3's two open questions both got answered with numbers this session, and the second answer is
the better finding.** Name transparency is now a recorded run condition: `src/ds_agents/naming.py`
rewrites a fixture CSV's header line and copies every remaining byte through, so the descriptive and
opaque arms differ in one line of one file. `naming` lives on the frozen `RunConfig` and lands in
`results_row()`, and `ds-agents run` grew `--naming`, `--repeat` and `--results`. Twenty live Haiku
runs on `claims_timing` are committed at `evals/results/2026-08-27_naming-ablation.jsonl` -- the
project's first results file. Three things to know before touching anything.

First, **the name effect is enormous.** The profiler nominated a planted trap in 10 of 10
descriptive runs and 2 of 10 opaque ones, over byte-identical rows; mean `profiler_recall` 0.95
against 0.10. The downstream consequence is the part that matters: because `feature_eng` drops what
the profiler nominates, the trap survived into the matrix 2 of 10 descriptive versus **10 of 10
opaque**, and mean claimed roc_auc rose from 0.840 to 0.974 against a legitimate ceiling near 0.82.
A benchmark of descriptively named fixtures mostly does not test the reviewer at all. One control
sharpens the mechanism: the honest identifier `claim_ref` was nominated in 10/10 descriptive and
9/10 opaque runs, so the profiler detects *statistically* visible leakage regardless of naming and
*semantically* visible leakage only when the name says so. Both traps were tuned to be statistically
ordinary, which is exactly the class it cannot see.

Second, **the reviewer now misses in 12 of 12 runs where it had a leak in front of it**, up from
last session's 3 of 3, and the shape of the miss is now visible. It is not passive: the opaque arm
raised nine objections to the descriptive arm's two, because a claimed 0.974 is conspicuous. In one
run at a claimed 0.9886 it raised `metric_mismatch` ("a 0.074-point improvement ... not credible")
*and* `implausible_importance` naming `var_05, var_06` -- the two **noise** columns, flagged because
their importance was near zero -- while `var_07` and `var_08`, the traps at the top of the ranking
and solely responsible for the score, appear in neither objection. The reviewer has the number that
says the score is too good, reasons about it correctly, and searches the bottom of the importance
ranking instead of the top. Nothing in its prompt asks it to explain *which column* produced an
implausible score. That is the most actionable thing Phase 3 has produced.

Third, **the opaque arm is now the configuration in which the reviewer is measurable at all**, since
it is the only one that reliably puts a leaky matrix in front of it. Any reviewer ablation should
run there; the descriptive arm answers a question about the profiler.

The floor: 389 tests pass in 68.8s full, 325 in 7.9s fast, ruff clean. `uv run ds-agents run
--dataset toy` is green -- leak dropped, `publishable: yes`, ended `exhausted` on the long-standing
`metric_mismatch` pattern. The ablation cost $0.343 for 20 runs, ~23s each.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above, then continue Phase 3 in plan mode: run the
**Haiku-versus-Sonnet reviewer ablation** on `claims_timing --naming opaque`, n>=10 a side, using
the `--reviewer-model` flag that has existed since Phase 1 and has never been exercised. This is the
first result that makes the comparison interesting, and the opaque arm is the only configuration
that reliably hands the reviewer a leaky matrix, so it is the only one where the arm measures the
reviewer rather than the profiler. Before running, decide whether to also test the prompt hypothesis
above -- that the reviewer never asks which column caused an implausible score -- since a one-line
prompt change plus a rerun would separate "Haiku cannot do this" from "we never asked it to", and
those have very different implications for the writeup. Budget roughly $0.70 for the model arm alone
(Sonnet reviewer runs cost more than the $0.017 Haiku ones).

## Open questions

- **Is the reviewer's miss a capability limit or a prompt limit?** It correctly identifies that a
  claimed 0.9886 is not credible and then names the two lowest-importance columns. Nothing in
  `reviewer._user_message` asks "which column explains this score". Testing the prompt change is
  cheap and it is a confound under any Haiku-versus-Sonnet number: if the prompt is the binding
  constraint, the model arm measures nothing. Decide the order before spending on the model arm.
- **A legitimate objection with no available remedy is still not distinguishable from a false
  alarm.** Unchanged from last session and now seen in both arms (1 exhausted descriptive, 2
  opaque). The `metric_mismatch` objections are correct on the evidence but often target a candidate
  the modeler rejected, so nothing can satisfy them and the run burns the loop cap. Options
  unchanged: scope an objection to a candidate, teach the modeler to drop a candidate in response,
  or add an `unactionable` disposition.
- **`customer_id` as a false positive is now measured, and it is every run.** `profiler_false_alarm`
  exists now and says the honest identifier is flagged in ~10 of 10 runs in both arms, on all three
  fixtures. It is a stable ~1.0 per run added to every false-alarm number the project will publish.
  Options unchanged (an `acceptable_flags` set, count id columns as planted, or publish and explain)
  but there is now a number attached to not deciding.
- **The reviewer's non-column objections still do not reach a results row.** `profiler_*` closed the
  upstream half of this; the reviewer half is open. Nine objections were raised in the opaque arm
  and `results_row()` records only their count, so "raised nine objections, all about the wrong
  thing" is not expressible in the committed data and had to be read out of a report artifact by
  hand.
- **Should `reissued_ids` get the same treatment?** The name effect replicated on a second,
  structurally different trap would be a much stronger claim, and it is ~$0.70 and 20 minutes. It
  was deliberately scoped out this session.
- **Duplicate-rows-across-split is still unbuilt**, for the same structural reason: the reviewer
  never sees the split, the profile, or a row, and `results_row()` scores leakage as a set
  comparison over columns.
- **A refused model call loses its token accounting.** `AnthropicModel.generate` raises
  `ModelRefusal` before building the `Completion`, so a billed call reads $0.00. Touches `llm.py`,
  `_run.py` and all six nodes.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **`--results` is a stopgap and should be absorbed by Phase 4's harness**, not extended. It appends
  `results_row()` behind `publishable()` and knows nothing about subsets, baselines or datasets. It
  exists so this session's numbers are committed rather than prose; do not grow it into a harness.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN` in column order).** It
  leaks nothing today, but if a fixture ever puts its traps in a fixed position and a model learns
  that position across runs, this becomes a cue. A seeded shuffle would fix it at the cost of making
  a single run harder to read by eye.
- **`materialize` now does untranslated I/O** (`newline=""` on both sides). Without it Python
  normalises CRLF to LF and silently rewrites every line ending in the file, which would break the
  byte-identity guarantee for a CSV generated on another platform. No committed fixture uses CRLF;
  the test does.
- **`_fixture_state` derives its rename map from `naming` rather than taking both.** Flagged in
  review: as two independent arguments, `naming="opaque"` with an empty map was constructible, and
  that state scores a silent zero that looks exactly like a profiler success.
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it.** Unchanged. `materialize`'s three
  `ValueError` paths are tested but not caught in `cmd_run`, so a malformed fixture surfaces as a
  traceback rather than the exit-code-2 pattern used beside it.
- **The trap fixtures' `mutual_info_with_target` is documentation, not a difficulty dial.** This
  session settles it: association was held constant and the names moved the result by 8x.
- **A stub-model survival test is what stands between a fixture and silent uselessness.**
  `tests/test_trap_survival.py` now runs every assertion in both naming arms and additionally pins
  that the arms produce *identical* permutation importances. Any new fixture needs this.
- **`_toy_state` is still a wrapper** around `_fixture_state(load_fixture("toy"))`, kept because two
  test modules import it.
- **A per-item rule on a response schema is a whole-response rule.** Still worth checking `intake`
  and `modeler`'s LLM-facing schemas.
- **The split manifest is still embedded in snippet text**, and the 1 MiB read cap makes it urgent
  at Phase 4 sizes. Fix is `nodes/feature_eng.py`, `nodes/modeler.py` and their tests.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.**
- **The reviewer-off arm still runs the reviewer node** as a zero-cost no-op.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
  Restrict to the best-by-CV candidate at Phase 4 sizes.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`.
- `ArtifactStore` copies the dataset per run and chmods it 0444. Wasteful for a benchmark set.
- **`.mcp.json` hardcodes the toy dataset on argv.** Still wrong for Phase 5's generalist arm.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
