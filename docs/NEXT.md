# Next session

## Start here
**Phase 3's headline was wrong, and finding that out cost $0.54.** The sticky-drop effect --
`leakage_remediated` 5/10 -> 9/10, the largest number in this project -- **did not replicate against
a same-commit control.** The paired cell came back at control 7/10 against sticky 6/10, a difference
of **-1/10** where +4/10 was pre-registered, and the sticky arm did not reproduce its own 9/10
either. `evals/results/2026-08-28_forced-drop-release.jsonl`, n=10 per arm, $0.5420 against a $0.85
cap. **That line is retired: do not quote 5/10 -> 9/10 anywhere, including the Phase 5 writeup.**

**What replaced it is worth more.** Four cells now exist at the same nominal configuration, and
**two running identical behaviour returned 9/10 and 6/10.** Model nondeterminism alone moves a
10-run count on `claims_timing` by about 3 -- which is most of the effect the original comparison
reported, and it puts error bars on **every** 10-run count in this project, including the routing
arm's pre-registered 1/10 -> 5/10, which is the same size of difference. Nothing about this was
visible from any single cell. It only appears when the same thing is run twice, and until this
session nothing had been.

**The mechanism is separately confirmed, and the fix stays in.**
`objections_falsely_resolved` -- an objection marked `resolved` whose column is back in
`final_features` -- is **3/10 in the control arm and 0/10 in the sticky arm**, matching 0/10 in the
committed closure cell. So resurrection really happens under the old rule and the sticky rule really
prevents it. What is falsified is the link from that event to the outcome: the pipeline recovers on
its own, the reviewer re-objecting to the re-admitted column on a later pass, so resurrection costs
a *loop* rather than a *result*. The control arm ran longer (2.4 loops vs 2.0) and dearer ($0.0292
vs $0.0250), exactly as predicted, and still reached 7/10.

To get the control at all, `forced_drop_release` became a `Literal["withdrawn_only",
"resolved_or_withdrawn"]` on the frozen `RunConfig`, overturning DECISIONS.md 2026-08-28 (fourth
entry). It is the **one condition here whose default is not the old behaviour**, because its off
value reproduces a defect rather than offering a second design; applied in one place, and under the
off value `binding_objections` is provably `open_objections` again. **The axis is now closed: it has
exactly one legitimate use and it has been used.** Do not cross it with anything.

The floor: **487 tests pass** (up from 473; 14 added), ruff clean.
`uv run ds-agents run --dataset toy` green live -- leak dropped, verdict `pass`, `publishable: yes`.
Session spend ~$0.62 all in, against a $0.85 cap.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above. **Phase 3 is closed for real and Phase 4 is
the next session**: `evals/datasets/manifest.yaml` with 10-15 OpenML/Kaggle datasets and cited
published baselines, then `harness.py`.

Two things this session hands Phase 4 that change how to build it. **The harness must run more than
one seed per condition, or it cannot report an effect** -- a single 10-run cell on one fixture is now
demonstrably unable to resolve a 4/10 difference, and that is a harness design constraint, not a
caveat to add later. And **`.claude/skills/run-eval/SKILL.md` documents `ds-agents eval --subset ci`
and `eval-diff`, neither of which exists**; a session following that skill will be misled, so
aligning it with what `harness.py` actually builds belongs in the same session, not after it.

Before starting, decide whether the first Phase 4 task is the manifest or the harness. The manifest
is blocked on an open question from session 0 (which OpenML suite has citable baselines) that can be
researched without spending anything.

## Open questions

- **The zero-objection `block` bug is the leading cause of failure and is now also a measurement
  hazard.** It fired **3 times in the sticky arm and 0 times in the control arm** of this cell,
  against a base rate of 1-2 in 10 -- an asymmetric draw that ate the sticky arm's numerator and is
  the single biggest contributor to the primary's failed direction. The reviewer claims `block` with
  nothing open, the router correctly refuses it, and the run goes straight to the reporter having
  dropped nothing (`route_sequence: ["reporter"]`, `errored: true`). Deliberately not fixed this
  session so the arms stayed comparable. **It should be the first code change of the next session.**
  The live hypothesis, unchecked: the reviewer raises objections that the malformed-objection filter
  drops one at a time, leaving `claim="block"` with an empty list -- `reviewer.py` never checks that
  a `block` claim is accompanied by a surviving objection. Two teed logs from this cell are the
  first evidence that has ever been kept; results rows carry no error text.
- **Every published 10-run count needs a replicate before Phase 5 quotes it.** The routing arm's
  1/10 -> 5/10 is the same size as the effect that just dissolved, and it has never been repeated.
  This is cheap (~$0.30 a cell) and it decides what the writeup is allowed to claim.
- **`exhausted` is now uninformative and should probably be said so in the README.** All 4
  `exhausted` runs in the closure control **still remediated**. The verdict no longer distinguishes
  "the fix did not land" from "the reviewer was still talking when the cap bound". This cell adds a
  second reason: the control arm ended `pass` 6 and `exhausted` 4 while the sticky arm ended `pass`
  4, `block` 3, `exhausted` 3 -- the verdict distribution moved without the outcome moving.
- **Late detection against the cap is the remaining structural defect.** Every `exhausted` run
  raises a new objection on its final pass, which routes straight to the reporter. Worth one cheap
  re-check at `loop_cap=4` under the current code (~$0.30), now that a pass actually does something
  and the sticky fix has landed -- the old loop-cap null was measured on a pipeline where no pass
  was actionable.
- **Should `by_category` become the default?** Weaker than it looked. The 9/10 that motivated it is
  now one draw of four cells reading 9, 7, 6 and 5 out of 10. Across all 40 `by_category` rows the
  reachable-population rate is high (15/16 sticky, 12/18 unsticky) but that pooling crosses a code
  boundary. Every downstream cost estimate in PLAN.md still assumes the old default.
- **Does the reviewer ever use `withdrawn` unaided?** 3 of 10 in the closure cell, 2 of 10 in this
  cell's sticky arm. That matters because `withdrawn` is the only disposition that restores a
  column, so it is the sole route back from a `by_category` false positive. Consistent across two
  cells now, at roughly 1 run in 4 -- enough that the escape hatch is not theoretical, not enough to
  call it reliable.
- **Sonnet under `by_category` + the sticky fix.** Still worth ~$0.65, and the prediction has to be
  restated: the old version rested on "the Haiku baseline is 9/10", which is no longer a number.
  Any model arm now needs a replicated Haiku baseline to be compared against, which roughly doubles
  its real cost.
- **`customer_id` as a false positive is measured and it is every run.** Unchanged.
- **Should `reissued_ids` get the naming treatment?** Unchanged. Scoped out five times now.
- **Duplicate-rows-across-split is still unbuilt**, for the same structural reason.
- **A refused model call loses its token accounting.** Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **`forced_drop_release` is closed to further use.** One legitimate use, now spent. Its default is
  the only one in this repo that is deliberately not the pre-existing behaviour, pinned by
  `test_the_default_release_rule_is_the_fixed_behaviour_and_deliberately_not_the_old_one` -- it
  reads as an inconsistency and "fixing" it would ship the bug.
- **The 2026-08-28 sticky fix changed a prompt string as well as a predicate.** `d5a9a28` reworded
  the forced-drop justification that reaches the feature_eng model's prompt (`"open reviewer
  objection X: ..."` -> `"reviewer objection X, not withdrawn: ..."`), so **any future claim that an
  arm is "byte-identical to the pre-fix tree" is false**. The control arm reproduces the pre-fix
  *release rule*, not the pre-fix *tree*. Found only by reading the fix's own diff.
- **`feature_eng.py`'s comment on that justification string says it "reaches the snippet the model
  reads".** It does not -- it reaches the model's *prompt*, via `already_dropped`, and
  `dropped_features` on the row. The snippet only ever carries `DROP = [...]`. Harmless today,
  wrong for a future reader.
- **`binding_objections`' single-caller invariant is now enforced**, by an AST walk over the package
  (`test_binding_objections_has_exactly_one_caller_in_the_graph`). First test here that reads its own
  source; the pattern is available if another docstring-only invariant needs the same treatment.
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
