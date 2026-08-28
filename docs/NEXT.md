# Next session

## Start here
**The routing half of the remediation fix landed and it worked, at exactly the pre-registered
threshold.** `objection_routing` is a `Literal["as_addressed", "by_category"]` on the frozen
`RunConfig`. `as_addressed` is the default and reproduces every committed row byte for byte -- the
evidence is that all 27 pre-existing router tests pass with zero edits. Under `by_category` a
column-scoped objection gets an *effective* target of `feature_eng` whatever the reviewer wrote.
n=10 live at $0.272: **`leakage_remediated` 5/10 against the control cell's 1/10**, with 8/10 runs
reaching `feature_eng`. `evals/results/2026-08-28_objection-routing.jsonl`. The house rule is
">=5/10 is shown", so this reaches the line rather than clearing it.

**The design question NEXT.md left open is settled and the answer is "record it".**
`Objection.target_node` is never rewritten -- the reviewer's own dispatch choice stays on the record
and `objections_by_target_node` still counts the raw field, so the reviewer's judgement remains
measurable *in the arm that overrides it*, with `objections_rerouted` counting the disagreements.
`test_the_raw_target_node_survives_the_reroute` fails if anyone folds the effective target into that
counter. The condition is applied in exactly one place, `PipelineState.effective_target`, read only
by `open_objections`; `_route_for_block` was rewritten to ask `open_objections(destination)` rather
than compare `target_node` itself, because the router and `feature_eng` had been two independent
answers to "who acts on this" and under the new arm they would have disagreed. See DECISIONS.md
2026-08-28 (third entry).

**Three things that did not go to plan, all recorded rather than smoothed over.** The arm got
*cheaper* ($0.0272/run against a predicted $0.047 and a control of $0.0299) because acting on an
objection ends the loop instead of grinding to `exhausted` -- mean loops fell 2.7 to 2.2 and three
runs ended `pass`, which no control run did. So the pre-registered prediction that `exhausted` would
stay flat was wrong, in the run's favour. The arm also has a real price: **2 of 10 runs dropped
`prior_claims_12m`, a legitimate strong feature**, because `by_category` turns a reviewer false
positive from something inert into a really dropped column. `n_final_features` is on the results row
for that reason and stayed 4-7, so nothing remediated by gutting the matrix.

**Every one of the 5 non-remediating runs failed for a reason that is not routing**, which is the
most useful output of the cell. Three named the surviving trap on the **final** reviewer pass, and a
block at the cap becomes `exhausted` and routes to the reporter -- so **the effective number of
actionable reviewer passes is `loop_cap - 1`, not `loop_cap`**. Two claimed `block` with zero
objections raised (`errored: true`), the parking-lot reviewer bug, now reproduced twice more here
and once on a live toy run.

The floor: 445 tests pass (up from 427; 18 added), ruff clean.
`uv run ds-agents run --dataset toy` green live -- leak dropped, verdict `pass`, `publishable: yes`.
Session spend ~$0.42 all in.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above, then continue Phase 3 in plan mode on the
**closure** half, which is the last unticked Phase 3 item and was deliberately not bundled with the
routing arm. The reviewer needs a termination condition it can observe. The claim to beat is the
committed `by_category` cell (n=10, 5/10 remediated, 4/10 exhausted, 3/10 pass), run with everything
else held identical -- routing is now the default-off condition, so the closure arm must set
`--objection-routing by_category` to build on it rather than measure against a broken pipeline.

Note that routing partly relieved the closure symptom on its own, which was not predicted: three
runs closed their own objection and ended `pass`. That changes the closure question from "the
reviewer never resolves anything" to "the reviewer resolves only when the fix is unambiguous", which
is a smaller and better-posed problem. Re-read the diagnosis before designing the arm.

Two candidate closure conditions, both checkable from what the reviewer's prompt already contains,
so this is likely a third `reviewer_prompt` value rather than a schema change: the objected column
is absent from `final_features` (mechanical, already visible to the reviewer), or the claimed score
has fallen to within the plausible band. Decide which before writing the code, and pre-register.

**Settle one bug first, because it is cheap and it interacts with closure**: a forced drop is not
sticky (see the first open question below). Closure makes resolution *more* common, which makes that
hazard *more* reachable, so fixing it after the closure arm would invalidate the arm.

## Open questions

- **Is a forced drop sticky?** Found while diagnosing this cell and verified directly against the
  node: `_forced_drops` recomputes from `open_objections`, so once the reviewer dispositions an
  objection `resolved`, a *later* return to `feature_eng` for some other objection puts the leaked
  column back in the matrix. With the objection open the snippet reads `DROP =
  ['account_status_code', 'churned', 'customer_id']`; with it resolved, `DROP = ['churned',
  'customer_id']`. `tests/nodes/test_feature_eng.py::test_a_resolved_objection_does_not_force_a_drop`
  pins the current behaviour as intended, and across a single pass it is. It did not cause any
  failure in this cell -- the three surviving-trap runs never resolved anything -- but `by_category`
  produces the two-return shape routinely and closure will produce it more. Either a drop once
  forced stays forced, or resolution must not be allowed to resurrect a column.
- **The last reviewer pass is structurally unactionable.** `loop_cap` permits N passes but only N-1
  can be acted on, because a block at the cap becomes `exhausted` and routes to the reporter. Three
  of this cell's five failures are exactly this. It is arguably correct (the cap has to bind
  somewhere) but it means the loop-cap sweep's null result was measured on a pipeline where *no*
  pass was actionable, so it is worth one cheap re-check at `loop_cap=4` under `by_category` now
  that passes actually do something. ~$0.30.
- **Should `by_category` become the default?** It is strictly better on remediation and cheaper per
  run, and strictly worse on legitimate features dropped. Leaving it off keeps every committed row
  comparable; turning it on makes the pipeline the thing the README describes. Not urgent, but every
  downstream cost estimate in PLAN.md assumes the old default.
- **The "reviewer claimed block with no open objection" error is now reproducible.** Twice in this
  cell (`errored: true`, `route_sequence: ["reporter"]`, nothing dropped, claimed 0.9858) and once
  on a live toy run, where the reviewer raised `implausible_importance` naming only already-dropped
  columns and the node correctly rejected it, leaving `claim=block` with nothing open. No longer a
  parking-lot curiosity: it costs 2 of every 10 runs.
- **Sonnet remediates more while detecting less.** Still n=7 and still directional. Now worth
  re-running under `by_category`, because the Sonnet advantage was *entirely* that it addressed
  objections to `feature_eng` -- which `by_category` hands to Haiku for free. The prediction is that
  the model effect shrinks toward zero, and that is a much stronger finding than the original arm.
  ~$0.65.
- **`customer_id` as a false positive is measured and it is every run.** Unchanged.
- **Should `reissued_ids` get the naming treatment?** Unchanged. Scoped out four times now.
- **Duplicate-rows-across-split is still unbuilt**, for the same structural reason.
- **A refused model call loses its token accounting.** Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **`objection_routing`, `objections_rerouted` and `n_final_features` cannot be back-filled** onto
  any row written before this session. Any table crossing that boundary has to say so; only
  `leakage_remediated` compares cleanly against the control cell.
- **The cost model for this pipeline is now known to be non-monotonic.** Fixing a bug made runs
  cheaper because they stopped exhausting. Any future cost pre-registration should predict a range,
  not a point, and should say which direction a *successful* intervention would move it.
- **`by_category` does not hold two prompts constant.** `feature_eng` and `modeler` both read
  `open_objections(target)`, so the column objection leaves the modeler's prompt and enters
  `feature_eng`'s. Deliberate and tested (`test_a_rerouted_objection_leaves_the_modeler_prompt`),
  but it means the arm is a routing change *plus* two prompt changes and must not be described as a
  pure edge change.
- **`FORCING_OBJECTION_CATEGORIES` is gone**, replaced by importing `COLUMN_SCOPED_CATEGORIES`. The
  two were byte-identical duplicates and are now load-bearing together.
- **The loop-cap default `3` is written in three places** and `objection_routing`'s default is now
  written in three more (`RunConfig`, `_fixture_state`, argparse). The one-shared-constant argument
  is stronger than it was.
- **`_fixture_state` now takes six positional args** and `_run_once` passes them positionally. One
  transposition would silently swap two run conditions. Keyword args would fix it; the `naming`
  derivation invariant documented at `cli.py:45-53` is the thing not to break while doing it.
- **`--results` is a stopgap and should be absorbed by Phase 4's harness**, not extended.
- **`cmd_run` has no per-run `try/except`**, so an unhandled API error ends a `--repeat` cell early.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN`).**
- **A stub named anything but `"stub"` slips past `PLACEHOLDER_MODEL_NAMES`.**
- **`materialize` does untranslated I/O** to preserve byte identity across platforms.
- **`_fixture_state` derives its rename map from `naming` rather than taking both.**
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it**; `materialize`'s three
  `ValueError` paths are tested but not caught.
- **The trap fixtures' `mutual_info_with_target` is documentation, not a difficulty dial.**
- **A stub-model survival test is what stands between a fixture and silent uselessness.**
- **`_toy_state` is still a wrapper** around `_fixture_state(load_fixture("toy"))`.
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
