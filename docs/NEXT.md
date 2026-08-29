# Next session

## Start here
**Phase 4 has a harness, and the bug that was eating cells is fixed.** Three things shipped. The
zero-objection `block` bug -- NEXT.md's mandated first code change, 1-2 runs in 10, the leading
cause of failure -- now triggers a **single retry** of the reviewer's model call, carrying the
reasons its objections did not survive plus the profile's known columns inside the JSON facts
block. The cause was never established (last session's teed logs were not committed and are gone),
so the fix is blind to it: `PipelineState.would_be_open` answers "what is still open once this pass
lands", `router._route_for_block` now calls it instead of deriving its own answer, and the reviewer
asks the same question one node early. All three ways the open set can empty out are handled
identically. `REVIEWER_SYSTEM` is untouched, so **the first model call stays byte-identical to every
committed row.**

**The harness is real and it is replicate-aware by construction.** `ds-agents eval --subset
{toy,ci}` and `ds-agents eval-diff a.jsonl b.jsonl` exist; the run-eval skill documented both since
session 0 and neither did. `plan()` is replicate-major so a binding cost cap truncates every cell
roughly equally rather than starving the last one -- the exact failure that left the Sonnet cells at
n=4 and n=3. `evaldiff.py` holds the refusals: underpowered is checked before anything else and
covers **every row this project has committed**, since none carry a `replicate`. Worth knowing:
pooled Wilson alone already refuses the retired 5/10 vs 9/10 headline ([0.237, 0.763] against
[0.596, 0.982]), so the replicate rule's real job is testing the iid assumption, not narrowing the
interval.

**Three columns were added to the row**: `errors` (node, message, recoverable -- the measurement
hazard NEXT.md named), `commit` on the frozen `RunConfig`, and `default_model`, which was simply
missing while `reviewer_model` was present. None back-fill onto a committed row.

The floor: **577 tests pass** (up from 487; 90 added), ruff clean.
`uv run ds-agents run --dataset toy` green live -- leak dropped, verdict `pass`, `publishable: yes`.
Session spend ~$0.03 all in. This was an infrastructure session; no benchmark cell was run.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. Two candidates, and **the first
one is cheap and settles what the writeup may claim**:

1. **Run the `ci` subset live at 2 replicates, and get the block-retry's first live evidence.**
   `uv run ds-agents eval --subset ci --name ci-baseline --replicates 2 --n 5 --max-cost-usd 0.80`
   (~$0.65). This is the first live exercise of the harness beyond n=1, the first live rows for
   `reissued_ids`, and the first chance to see the `block-retry` prefix in the `errors` column --
   grep it. It also produces the replicated baseline that every remaining ablation needs.
2. **`evals/datasets/manifest.yaml`**, still blocked on the session-0 open question (which OpenML
   suite has citable published baselines). Researchable for free before spending anything.

Do (1) first unless the user wants otherwise: it is the cheaper of the two and it is the only thing
that can tell us whether the block fix works on a real model rather than on a scripted one.

## Open questions

- **The block-retry has never fired on a live run.** Unit tests cover all three trigger paths; live
  evidence is a ~$0.30 coin flip at the old 1-2-in-10 base rate, so it was deliberately not bought
  with infrastructure money. **The `errors` column now makes it greppable** -- the prefix is
  `block-retry`, and the messages distinguish "retry produced N actionable objection(s)" from
  "still produced nothing actionable". First real cell answers this for free.
- **Two consequences of the retry are recorded and unmeasured.** A run the retry rescues now reads
  `errored: true`, because every retry appends a `PipelineError`. And a malformed block on the FINAL
  pass can now end `pass` where it would have ended `exhausted`, so **verdict distributions shift at
  commit `1e5f30a`** and must not be pooled across it.
- **Every published 10-run count still needs a replicate before Phase 5 quotes it.** Unchanged and
  now enforceable: `eval-diff` refuses to compare single-replicate cells at all. The routing arm's
  1/10 -> 5/10 is the same size as the effect that dissolved and has never been repeated.
- **The `ci` subset has never been run live.** Its cells were chosen for sensitivity to breakage,
  not for comparability with any committed cell, and its cost estimates (0.010 / 0.030 / 0.025) are
  guesses that the first real run will correct.
- **`exhausted` is still uninformative**, and the retry gives it a second reason to be: see above.
  Probably worth saying so in the README.
- **Late detection against the cap is the remaining structural defect.** Every `exhausted` run
  raises a new objection on its final pass, which routes straight to the reporter. Worth one cheap
  re-check at `loop_cap=4` (~$0.30) now that a pass actually does something.
- **Should `by_category` become the default?** Unchanged. The 9/10 that motivated it is one draw of
  four cells reading 9, 7, 6 and 5. Every downstream cost estimate in PLAN.md still assumes the old
  default.
- **Does the reviewer ever use `withdrawn` unaided?** Unchanged: roughly 1 run in 4, across two
  cells. It is the only disposition that restores a column, so it is the sole route back from a
  `by_category` false positive.
- **Sonnet under `by_category` + the sticky fix.** Unchanged, and still needs a replicated Haiku
  baseline to be compared against -- which candidate (1) above would produce.
- **`customer_id` as a false positive is measured and it is every run.** Unchanged.
- **Should `reissued_ids` get the naming treatment?** Unchanged. Scoped out six times now.
- **Duplicate-rows-across-split is still unbuilt**, for the same structural reason.
- **A refused model call loses its token accounting.** Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **`errors`, `commit` and `default_model` cannot be back-filled** onto any row written before
  2026-08-29. Only rows were committed and the states are gone. `eval-diff` will therefore always
  group pre-2026-08-29 rows into their own cells, which is correct rather than inconvenient. For
  `commit` specifically, LOG.md's prose is the only provenance some earlier cells have -- the
  reviewer ablation records `2d5c1fc`, the naming ablation records nothing.
- **`harness.py` and `cli.py` import each other inside functions.** `harness` needs `_run_once` and
  `_append_results_row`; `cli` needs `run_eval`. The clean fix is a `runner.py` holding what both
  need, deferred deliberately rather than churning three test modules in a three-deliverable
  session. Take it when it earns its keep.
- **A failed run is charged its cell's ESTIMATE, not its real cost**, because the state that counted
  the tokens never came back. Reported separately as `charged_estimate_usd` so a spend figure never
  silently mixes measured and guessed dollars.
- **The cost cap is invocation-level by choice.** Replicate-major ordering already solves the
  starvation problem a per-cell cap would be for; two budgets means two invocations.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Deliberate -- the
  project has topped up a cell before (the Sonnet cells on 2026-08-28) -- but it means a careless
  re-run can append to a committed results file. Only a stderr note guards it.
- **`--results` on `ds-agents run` still exists** for one-off cells and debugging. Its rows carry no
  `cell` or `replicate`, so they can never enter a powered comparison; the skill says so.
- **`forced_drop_release` is closed to further use** and `Cell` deliberately has no field for it,
  pinned by test. Its default is the only one in this repo that is not the pre-existing behaviour.
- **The 2026-08-28 sticky fix changed a prompt string as well as a predicate**, so no arm is ever
  "byte-identical to the pre-fix tree". Unchanged.
- **`feature_eng.py`'s comment on the justification string says it "reaches the snippet the model
  reads".** It does not -- it reaches the model's *prompt*. Still wrong for a future reader.
- **`binding_objections`' single-caller invariant is enforced by an AST walk.** The pattern is
  available if another docstring-only invariant needs it -- and `would_be_open` now has exactly two
  callers by design, which is the kind of thing that walk could pin.
- **7 of 84 committed rows are code-boundary-crossed on the sticky-drop fix.** Unchanged. The 20
  naming-ablation rows and the 27 rows at `2d5c1fc` are unscreenable, not clean.
- **`objection_closure`, `objections_resolved`, `objections_withdrawn` and
  `objections_falsely_resolved` cannot be back-filled** onto any row before the closure cell.
- **A metric that conflates two opposite claims cannot gate anything.** Worth checking whether any
  other published column has `objections_open_at_end`'s defect.
- **`CLOSURE_RULE` deliberately omits a bullet** saying `withdrawn` is the only disposition that
  restores a column. Unchanged.
- **`test_no_appended_rule_names_a_fixture_column` is the answer-injection guard.** Note it does not
  and should not cover the retry's `known_columns`, which is a dynamic user message derived from
  state rather than an appended prompt rule -- a future session extending that guard would
  otherwise break the retry.
- **The loop-cap default `3` is written in three places**, and so are `objection_routing` and
  `objection_closure`'s defaults. `MAX_CONSECUTIVE_FAILURES` is at least a named constant.
- **`cmd_run` has no per-run `try/except`**, so an unhandled API error still ends a `--repeat` cell
  early. The harness fixed this for `eval` only.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN`).**
- **A stub named anything but `"stub"` slips past `PLACEHOLDER_MODEL_NAMES`.**
- **`materialize` does untranslated I/O** to preserve byte identity across platforms.
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it**; the harness lets `SystemExit`
  propagate on purpose, because a run that never started will not start for run 8 either.
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
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
