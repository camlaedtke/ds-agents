# Next session

## Start here
**Phase 3 is closed.** Both remaining items landed, and the headline is not the one the session set
out to get.

The closure arm was **built, pre-registered, and deliberately not run**. `objection_closure` is a
`Literal["off","on"]` on the frozen `RunConfig` -- its own axis, not a third `reviewer_prompt`
value, so the two prompt rules stay independently attributable -- and `CLOSURE_RULE` points only at
`final_features`, a field the reviewer already sees. Before spending anything, re-reading the
committed control rows showed **the premise the arm was designed against does not hold**: NEXT.md
and PLAN.md both said the reviewer never dispositions an objection `resolved`, but 8 of 10 routing
rows had already closed one, and all 4 `exhausted` runs raised a *new* objection on their *final*
pass rather than refusing to close. That is whack-a-mole against the cap, not an unfalsifiable
standard. So the control cell was pre-registered as a **decision gate** with a stopping rule instead
of a baseline. The gate closed at exactly its line -- `objections_resolved > 0` in 6/10 -- and two
other pre-registered numbers make the verdict stronger than that 6 alone: `objections_falsely_
resolved` is **0 in every row** (the reviewer never closes dishonestly *without* being told the
criterion), and `leakage_remediated` is **9/10**, with the only failure being the unrelated
zero-objection `block` bug. There is no headroom for a prompt rule to buy.
`evals/results/2026-08-28_objection-closure.jsonl`, n=10, $0.2938.

**The unpredicted result is the sticky-drop fix, and it is the largest remediation effect measured
so far.** `_forced_drops` recomputed from `open_objections`, which closes on `resolved` as well as
`withdrawn`, so a resolved objection stopped forcing its drop and the next return to `feature_eng`
put the leaked column back. `PipelineState.binding_objections` is now a second named question --
"what must stay out of the matrix" -- releasing only on `withdrawn`, with **exactly one caller in
the graph**; the router and the reviewer keep asking `open_objections`, or nothing ever terminates.
That fix alone, with the closure axis `off` and byte-identical, moved `leakage_remediated`
**5/10 to 9/10** against the routing cell -- larger than the routing arm's own 1/10 to 5/10 -- and
**6 of 6 runs that resolved an objection remediated**, 3 of them via the two-return shape where the
old bug fired. It has no pre-registration of its own. That is the first item below.

The floor: 473 tests pass (up from 445; 28 added), ruff clean.
`uv run ds-agents run --dataset toy` green live -- leak dropped, verdict `pass`, `publishable: yes`.
Session spend ~$0.32 all in, against a $0.80 cap; the gate saved the arm's $0.27.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above. Phase 3 is done; the question is whether
to spend one more session hardening its headline before moving to Phase 4's harness.

The case for doing it: the sticky-drop fix is currently the project's largest single remediation
effect (5/10 -> 9/10) and it is the *only* major result with **no pre-registration of its own** --
it was found while fixing a bug flagged in passing, and the comparison is n=10 against n=10 on one
fixture where model nondeterminism alone can move a count by a few. Everything else in the results
tables was pre-registered before it ran. Pre-register a confirmation cell, decide in advance what
would falsify it, and run it (~$0.30). The mechanism to state in the pre-registration is specific
and checkable: the effect should appear **only** in runs that resolve an objection and return to
`feature_eng` twice, so `objections_resolved > 0` crossed with a `route_sequence` containing two
`feature_eng` entries is the predicted carrier, and a rise concentrated anywhere else would falsify
the mechanism even if the headline number replicated.

The case against: Phase 3 is closed, PLAN.md budgets 15-25 sessions total and Phase 4 is untouched.

Either way, do **not** re-run the closure arm -- the gate closed on a pre-registered stopping rule
and reopening it without new evidence is exactly what pre-registration exists to prevent.

## Open questions

- **The sticky-drop effect has no pre-registration.** See "First prompt". This is the single
  weakest link in the results tables and it is currently the strongest number in them.
- **The zero-objection `block` bug is now the leading cause of failure**, and it is the *only*
  cause left in the closure control: 1 of 10 rows there, 2 of 10 in the routing cell, plus a live
  toy reproduction. The reviewer claims `block` with nothing open, the router correctly refuses it,
  and the run goes straight to the reporter having dropped nothing (`route_sequence: ["reporter"]`,
  `errored: true`). It was a parking-lot curiosity three sessions ago. It is now the bottleneck.
- **`exhausted` is now uninformative and should probably be said so in the README.** All 4
  `exhausted` runs in the closure control **still remediated**. The verdict no longer distinguishes
  "the fix did not land" from "the reviewer was still talking when the cap bound".
- **Late detection against the cap is the remaining structural defect.** Every `exhausted` run
  raises a new objection on its final pass, which routes straight to the reporter. Worth one cheap
  re-check at `loop_cap=4` under the current code (~$0.30), now that a pass actually does something
  and the sticky fix has landed -- the old loop-cap null was measured on a pipeline where no pass
  was actionable.
- **Should `by_category` become the default?** Unchanged, and now stronger: with the sticky fix it
  is 9/10 remediated. Every downstream cost estimate in PLAN.md still assumes the old default.
- **Does the reviewer ever use `withdrawn` unaided?** Measured for the first time this session:
  3 of 10 runs did. That matters because `withdrawn` is now the only disposition that restores a
  column, so it is the sole route back from a `by_category` false positive. 3/10 is enough that the
  escape hatch is not purely theoretical and not enough to call it reliable.
- **Sonnet under `by_category` + the sticky fix.** Unchanged and still worth ~$0.65. The prediction
  is still that the model effect shrinks toward zero, and it is now a stronger prediction because
  the Haiku baseline is 9/10.
- **`customer_id` as a false positive is measured and it is every run.** Unchanged.
- **Should `reissued_ids` get the naming treatment?** Unchanged. Scoped out five times now.
- **Duplicate-rows-across-split is still unbuilt**, for the same structural reason.
- **A refused model call loses its token accounting.** Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **7 of 84 committed rows are code-boundary-crossed on the sticky-drop fix** -- they meet the
  necessary condition for the old behaviour to have fired (a closure, plus `feature_eng` in
  `route_sequence[1:]`). 4 are in `2026-08-28_objection-routing.jsonl`, 3 in the loop-cap sweep.
  They cannot be decided further, because no row written before this session splits `resolved` from
  `withdrawn`. The 20 naming-ablation rows and the 27 rows at `2d5c1fc` carry no `route_sequence`
  and are **unscreenable, not clean**. Any table crossing that boundary has to say so.
- **`objection_closure`, `objections_resolved`, `objections_withdrawn` and
  `objections_falsely_resolved` cannot be back-filled** onto any row written before this session.
- **A metric that conflates two opposite claims cannot gate anything.** `objections_open_at_end`
  merged "the fix landed" with "I was wrong", which is why the sticky-drop screen above is stuck at
  "at risk" and why the gate needed new columns before it could be evaluated. Worth checking whether
  any other published column has the same defect.
- **`CLOSURE_RULE` deliberately omits a bullet** saying `withdrawn` is the only disposition that
  restores a column. True, but a pipeline mechanic rather than an observable field, and the rule's
  whole claim is that it points only at fields the reviewer is shown. If a future cell shows the
  reviewer never withdraws, adding it is the next arm and it will have evidence behind it.
- **`test_no_appended_rule_names_a_fixture_column` is the answer-injection guard** and it iterates
  every registered fixture, so a new fixture cannot quietly turn an existing prompt rule into a
  cheat sheet. It did not exist for `WHICH_COLUMN_RULE` until this session.
- **`_fixture_state` is now keyword-only** after `fixture`, which closes the transposition hazard
  the parking lot carried for three sessions. `_toy_state` is still a wrapper around it.
- **The loop-cap default `3` is written in three places**, and `objection_routing` and
  `objection_closure` now have their defaults written in three each. The one-shared-constant
  argument is stronger every session.
- **`--results` is a stopgap and should be absorbed by Phase 4's harness**, not extended.
- **`cmd_run` has no per-run `try/except`**, so an unhandled API error ends a `--repeat` cell early.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN`).**
- **A stub named anything but `"stub"` slips past `PLACEHOLDER_MODEL_NAMES`.**
- **`materialize` does untranslated I/O** to preserve byte identity across platforms.
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it**; `materialize`'s three
  `ValueError` paths are tested but not caught.
- **The trap fixtures' `mutual_info_with_target` is documentation, not a difficulty dial.**
- **A per-item rule on a response schema is a whole-response rule.** Still worth checking `intake`
  and `modeler`'s LLM-facing schemas.
- **The split manifest is still embedded in snippet text**, and the 1 MiB read cap makes it urgent
  at Phase 4 sizes.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.**
- **The reviewer-off arm still runs the reviewer node** as a zero-cost no-op.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`.
- `ArtifactStore` copies the dataset per run and chmods it 0444.
- **`.mcp.json` hardcodes the toy dataset on argv.** Still wrong for Phase 5's generalist arm.
- **`.claude/skills/run-eval/SKILL.md` documents `ds-agents eval --subset ci` and `eval-diff`,
  neither of which exists.** A skill naming commands that do not exist will mislead a future session.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
