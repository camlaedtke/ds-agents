# Next session

## Start here
**The remediation bottleneck is diagnosed and it was not the loop cap.** A pre-registered sweep at
`loop_cap` 1/3/5, n=10 each on `claims_timing --naming opaque --reviewer-prompt which_column`, moves
remediation 0, 1, 1 out of 10 while cost per run quadruples ($0.0141 -> $0.0553). 20 rows at
`evals/results/2026-08-28_loop-cap-sweep.jsonl`; the cap=3 arm is the haiku/which_column cell of the
reviewer ablation, reused rather than re-run. Record it as the Phase 5 loop-cap ablation and as a
null result on remediation.

**Two causes instead, both independent of the cap**, found from three diagnostic runs ($0.10) and
confirmed by the sweep. First, **routing**: `feature_eng` force-drops an objected column before its
own model is consulted, but only for objections whose `target_node` is `feature_eng`. The reviewer
under `which_column` overwhelmingly raises `implausible_importance`, which is an objection about a
*score*, and `REVIEWER_SYSTEM` sends anything about "how the run was evaluated" to `modeler` -- a
node with no column lever at all, since every candidate is fit on the one transform `feature_eng`
already froze. Second, **closure**: across 3 diagnostic runs the reviewer never dispositioned an
objection `resolved`. The clearest run dropped both traps, watched the claimed roc_auc fall
0.986 -> 0.823, wrote that this "is consistent with removing leakage", and held the objection open
because the columns "were never validated as non-leaking, only removed" -- an unfalsifiable
standard. **So `exhausted` is not evidence the trap shipped.** `leakage_remediated` is the field
that answers that and it is the one to quote.

**The one number that carries the session.** Across all 27 rows now carrying `route_sequence`:
0 of the 21 runs that never routed to `feature_eng` remediated, against 3 of the 6 that did.
Reaching `feature_eng` is necessary and not sufficient.

**The Sonnet cells were topped to n=7 each** ($0.613, `2026-08-28_reviewer-ablation.jsonl`, now 34
rows). The prompt effect holds and strengthens -- sonnet/base is 0 of 7. The model effect separated
from it and **is not about detection**: under `which_column` Haiku names a trap in 9 of 10 runs and
Sonnet in 5 of 7, so Haiku detects *more*, but Sonnet remediates 3 of 7 against Haiku's 1 of 10 and
exhausts 1 of 7 against 8 of 10 -- because Sonnet addresses its objection to `feature_eng`.
Directional at this n; the pre-registered rule is >=5/10. n=7 rather than the pre-registered n=8
because the sweep overran its estimate; n was cut on cost alone before any Sonnet row was read.

**What shipped in code.** `results_row()` gained `objections_by_target_node`,
`objected_columns_unremediated`, `route_sequence` and `new_objections_per_pass`, all derived from
`objections`/`review_passes`/`final_features` -- no node, tool or `PipelineState` change. `--loop-cap`
became a real CLI flag; it had been on the frozen `RunConfig` and on every results row since the
first one with no way to set it.

The floor: 427 tests pass in 71.5s full, ruff clean. `uv run ds-agents run --dataset toy` is green
live -- leak dropped, verdict `pass`, `publishable: yes`. Session spend ~$1.45 against a $1.50 cap.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above, then continue Phase 3 in plan mode on the
**routing fix**, which is the last unticked Phase 3 item. The claim to beat: 0 of 21 runs that never
routed to `feature_eng` remediated. Make a column-scoped objection reach the node that can act on
it, and measure it against the committed haiku/which_column cell (n=10, 1/10 remediated, 8/10
exhausted) with everything else held identical. Design the intervention first -- the options are not
equivalent and one of them changes what the eval measures:

1. **Route by category, not by the reviewer's choice.** `implausible_importance` naming a column is
   a `feature_eng` problem whatever the reviewer calls it, so the router (or `_forced_drops`) could
   ignore `target_node` for column-scoped categories. Cheapest, and it takes a decision away from
   the model under test -- which is either the right call or quiet cheating, and that argument
   belongs in DECISIONS.md before the code.
2. **Fix the prompt instead**, the way `which_column` fixed detection: one rule saying a column-
   scoped objection goes to `feature_eng`. Keeps the routing decision inside the system under test
   and is a second prompt condition, so it needs to be recorded on `RunConfig` like the first one.
3. **Give `modeler` a real lever**, so a `modeler`-targeted objection is not a dead end. Largest
   change and it is Phase 4/5 scope.

Then, separately, the **closure** half: the reviewer needs a termination condition it can observe.
Do not bundle the two into one arm -- each needs its own before/after against the same cell.

## Open questions

- **Is routing by category a fix or is it cheating?** The whole point of the reviewer is that it is
  a model under test. Deciding `target_node` on its behalf makes the pipeline work and makes one
  fewer thing measurable, and the naming-ablation precedent says record the condition rather than
  argue about it. Settle this in DECISIONS.md before writing the code.
- **What can the reviewer observe that closes its own objection?** It currently demands proof of
  legitimacy, which no node can supply. Candidates: the column is absent from `final_features`
  (mechanical, and the reviewer already sees it); the claimed score fell to within the plausible
  band. Both are checkable from what the prompt already contains, so this may be a prompt fix rather
  than a schema one -- which would make it a third recorded `reviewer_prompt` value.
- **Sonnet remediates more while detecting less.** Solid enough to plan against, not solid enough to
  publish at n=7. Worth ~$0.65 to take both `which_column` cells to n=10 once the routing fix has
  landed, so it measures the fixed pipeline rather than the broken one.
- **`customer_id` as a false positive is measured and it is every run.** `profiler_false_alarm` is a
  stable ~1.0-1.3 per run in every cell of every ablation. Options unchanged (an `acceptable_flags`
  set, count id columns as planted, or publish and explain).
- **Should `reissued_ids` get the naming treatment?** The name effect replicated on a second,
  structurally different trap would be a much stronger claim. ~$0.70 and 20 minutes. Scoped out
  three times now.
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

- **Two live toy runs this session behaved differently**: one passed at a single reviewer pass, the
  other took three. The non-termination described above reaches the toy fixture too, so the toy
  run's cost varies 2-3x between invocations. Not a break -- the gate is the verdict, and it passes.
- **The "reviewer claimed block with no open objection" error was not chased.** It appeared twice in
  the committed haiku/which_column cell and in none of this session's 27 new rows. `route_sequence`
  now makes it visible as a run whose only entry is `reporter`, so it is cheaper to find next time.
- **`objections_by_target_node` and friends cannot be back-filled** onto the 27 rows written at
  commit `2d5c1fc`. Any table crossing that boundary has to say so.
- **Cost-per-remediated-leak is now computable and is a better headline than cost-per-catch.**
  haiku/which_column is $0.30 per remediated run; sonnet/which_column is $0.21. The stronger model is
  cheaper per unit of the thing that actually matters, which is a good line for the README.
- **The prompt arm roughly doubles wall time and cost.** If `which_column` becomes the default, every
  downstream cost estimate in PLAN.md is low by ~2x.
- **The hint-injection ablation has a drawn boundary to cross.** `WHICH_COLUMN_RULE` points only at a
  field the reviewer already receives and names no trap type; the parked hint-injection arm
  deliberately tells the reviewer which *categories* of failure exist. Label it as such when it runs.
- **`--results` is a stopgap and should be absorbed by Phase 4's harness**, not extended. It appends
  `results_row()` behind `publishable()` and knows nothing about subsets, baselines or datasets, and
  has no commit field, so the SHA lives only in `evals/results/LOG.md` by hand.
- **`cmd_run` has no per-run `try/except`**, so an unhandled API error ends a `--repeat` cell early.
  Already-appended rows survive; recovery is re-invoking with `--repeat <remaining>`. Retry policy
  belongs in Phase 4's harness.
- **The loop-cap default `3` is written in three places** (`RunConfig`, `_fixture_state`, argparse).
  Two tests pin the chain, but one shared constant would be better.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN` in column order).** It leaks
  nothing today, but a fixture with traps in a fixed position could become a learnable cue.
- **A stub named anything but `"stub"` slips past `PLACEHOLDER_MODEL_NAMES`.** Only constructible in
  a test.
- **`materialize` does untranslated I/O** (`newline=""` on both sides) to preserve byte identity for
  a CSV generated on another platform.
- **`_fixture_state` derives its rename map from `naming` rather than taking both**, so the opaque
  arm cannot be claimed without the rename being applied.
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it.** `materialize`'s three
  `ValueError` paths are tested but not caught, so a malformed fixture is a traceback rather than
  the exit-code-2 pattern beside it.
- **The trap fixtures' `mutual_info_with_target` is documentation, not a difficulty dial.**
- **A stub-model survival test is what stands between a fixture and silent uselessness.**
  `tests/test_trap_survival.py` runs every assertion in both naming arms. Any new fixture needs this.
- **`_toy_state` is still a wrapper** around `_fixture_state(load_fixture("toy"))`.
- **A per-item rule on a response schema is a whole-response rule.** Still worth checking `intake`
  and `modeler`'s LLM-facing schemas.
- **The split manifest is still embedded in snippet text**, and the 1 MiB read cap makes it urgent at
  Phase 4 sizes. Fix is `nodes/feature_eng.py`, `nodes/modeler.py` and their tests.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.**
- **The reviewer-off arm still runs the reviewer node** as a zero-cost no-op. `results_row()` reports
  `None` rather than 0.0 for the `reviewer_*` fields, and now for `objected_columns_unremediated`.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
  Restrict to the best-by-CV candidate at Phase 4 sizes.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`.
- `ArtifactStore` copies the dataset per run and chmods it 0444. Wasteful for a benchmark set.
- **`.mcp.json` hardcodes the toy dataset on argv.** Still wrong for Phase 5's generalist arm.
- **`.claude/skills/run-eval/SKILL.md` documents `ds-agents eval --subset ci` and `eval-diff`,
  neither of which exists** (`cmd_eval` prints "lands in Phase 4" and returns 2). Fix the skill or
  build the harness; a skill that names commands that do not exist will mislead a future session.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
