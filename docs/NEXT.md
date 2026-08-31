# Next session

## Start here
**The `ci` subset ran live for the first time, and it paid for itself twice over.** 30 rows at
`evals/results/2026-08-31_ci-baseline.jsonl` -- 3 cells x 2 replicates x n=5, $0.7291 against a
$1.00 cap, 0 refused, 0 failed, pre-registered in DECISIONS.md before the runner was called. It
bought the two things it was run for and one it was not.

**The zero-objection `block` retry fired, and this is its first live evidence.** 7 of 30 runs, and
**4 of those 7 retries produced an actionable objection; all 4 remediated.** The other 3 produced
nothing actionable, which is the model saying it meant the block. The pre-registration expected 1-3
and the count came in high for an instructive reason: 5 of the 7 were on `reissued_ids`, which
contributed nothing to the 1-2-in-10 base rate the estimate was extrapolated from. Note the
bookkeeping cost -- every retry appends a `PipelineError`, so a *rescued* run reads `errored: true`,
and **`errored` on this file is not a reliability rate.**

**The run exposed a defect that only a multi-run invocation could show, and it is fixed.**
`_run_once` read `git_commit()` per run; the results file it writes is untracked; so writing row 0
dirtied the tree and runs 1-29 recorded `8a629bf-dirty` against run 0's `8a629bf`. `commit` is an
`eval-diff` condition field, so **`toy-default` fragmented into cells of n=1 and n=9** -- the field
that guarantees rows came from one tree instead guaranteed they could not be pooled. Provenance is
now read **once per invocation**, the same rule `_live_run` already applied to `materialize`, and
`commit` is a required keyword on `_run_once` with no default, because the bug was a default rather
than a call site. Both callers carry a behavioural regression test. **The committed rows are not
back-fixed** -- they record what the run recorded.

Also landed: the three `ci` `est_cost_usd` guesses replaced by measured means (0.015 / 0.030 /
0.029, planning-only, no published number moves), and `commit` + `default_model` added to
ARCHITECTURE.md's two `RunConfig` field lists, where they had been missing since the session that
introduced them.

The floor: **580 tests pass** (up from 577), ruff clean, `uv run ds-agents run --dataset toy` green
live at $0.0133. Session spend ~$0.76 all in.

## First prompt
Read CLAUDE.md, docs/PLAN.md Phase 4, and the "Start here" above. **The highest-value next thing is
free**: `evals/datasets/manifest.yaml`, PLAN.md Phase 4's last unchecked box, blocked since session
0 on one researchable question -- which OpenML suite has citable published baselines. Research it
before spending anything; it is the only Phase 4 item left that does not need a live run, and Phase
5's writeup needs it.

If the user would rather spend money, the two cheap live candidates are:

1. **The `pass`-with-no-model defect** (see open questions). ~$0.30 to characterize at n=10 on
   `claims-opaque-which`, or $0 to fix the verdict derivation and unit-test it -- but a verdict
   change touches a column every committed row carries, so it is its own cell, not a footnote.
2. **`loop_cap=4`**, the late-detection re-check carried from Phase 3. ~$0.30, and both arms must
   run in one invocation (see below).

## Open questions

- **A run produced no model at all and was recorded `review_verdict: "pass"`.**
  `claims-opaque-which` rep=1 idx=2: `by_category` dropped every column across two `feature_eng`
  passes, both candidates failed to fit on an empty matrix, the reviewer had nothing left to object
  to, blocked, and its retry produced nothing. `n_final_features: 0`, `claimed_holdout_score: null`,
  verdict `pass`, and **`publishable()` admitted the row.** `leakage_remediated` is correctly `null`
  and `eval-diff` excluded it, but the verdict is the opposite of what happened. Left open
  deliberately: fixing verdict derivation changes a column every committed row carries.
- **The `ci` baseline is NOT a comparand for future ablations, and the last NEXT.md was wrong to say
  it would be.** `commit` is a condition field and a new arm is usually a new `Literal` on
  `RunConfig`, hence a new tree, which `eval-diff` refuses to pool. **Both arms of any comparison
  must run in one invocation at one commit** -- as the forced-drop-release cell already did. This
  roughly doubles every remaining Phase 5 cost estimate a second time, because each arm now has to
  fund its own control.
- **No CI job and no thresholds, and the reason has changed.** A threshold is now affordable to set,
  but the baseline says an honest one is wide: `leakage_remediated` 8/10 carries [0.490, 0.943], so
  a gate is roughly "fail under 4/10" and catches only outright breakage. There is no `.github/` in
  this repo, no key available to Actions, and a gate that spends real API money per push is a policy
  nobody has written. The cheap deterministic gate (pytest + ruff + toy green) is the better buy.
- **`errored` needs a companion column, or a caveat everywhere it appears.** The retry made it
  ambiguous: a rescued run and a broken run both read `true`. Counting `block-retry` by string
  prefix is the only way to separate them today.
- **`exhausted` is still uninformative** and now has two reasons to be. Unchanged; still worth
  saying in the README.
- **Late detection against the cap is the remaining structural defect.** Unchanged. Worth one cheap
  re-check at `loop_cap=4`, now with a same-invocation control.
- **Should `by_category` become the default?** Unchanged, and this run adds a data point against
  taking it lightly: it is what produced the zero-feature run above.
- **Does the reviewer ever use `withdrawn` unaided?** Unchanged: roughly 1 run in 4.
- **`customer_id` as a false positive is measured and it is every run.** Unchanged.
- **Duplicate-rows-across-split is still unbuilt**, same structural reason.
- **A refused model call loses its token accounting.** Unchanged.
- **LangSmith is wired but never exercised.** Unverified until a key exists.
- Which OpenML suite has citable published baselines. Open since session 0; see First prompt.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot

- **`reissued_ids` is the slow cell**: mean 86.1s a run against 34.7s for `claims` and 21.0s for
  `toy`, at comparable dollar cost. Wall time is not on any results row, so nothing gates on it.
- **The `ci` baseline's `toy-default` row pools two cells that `eval-diff` will not pool** (n=1 at
  `8a629bf`, n=9 at `8a629bf-dirty`). LOG.md says so; anything reading that file programmatically
  must group by the full condition key, never by `dataset_id`.
- **`cli` imports `git_commit` at module scope while `harness` imports it inside `_live_run`** (to
  break the import cycle), so the two provenance tests must monkeypatch two different names. Both
  say so in their docstrings.
- **The n=1 smoke could not have caught the provenance bug**: with one run there is no second read
  to disagree with the first. Worth remembering the next time a write-path proof feels sufficient.
- **`harness.py` and `cli.py` still import each other inside functions.** The clean fix is a
  `runner.py`; deferred again. `_run_once`'s signature grew a parameter this session without pain,
  so it has not started hurting yet.
- **A failed run is charged its cell's ESTIMATE, not its real cost.** Reported separately as
  `charged_estimate_usd`, which was $0.00 for this run -- every dollar measured.
- **The cost cap is invocation-level by choice.** Unchanged.
- **`ds-agents eval` appends to a same-day, same-name file rather than refusing.** Unchanged, and
  now slightly more dangerous: a careless re-run would append post-fix rows carrying a different
  `commit` to this baseline.
- **`--results` on `ds-agents run` still exists** for one-off cells; its rows carry no `cell` or
  `replicate`.
- **`forced_drop_release` is closed to further use.** Unchanged.
- **The 2026-08-28 sticky fix changed a prompt string as well as a predicate.** Unchanged.
- **`feature_eng.py`'s comment on the justification string is still wrong** for a future reader.
- **`binding_objections`' single-caller invariant is enforced by an AST walk**, and
  `test_run_once_has_no_default_commit_to_fall_back_to` is now a second signature-shaped invariant
  in the same spirit.
- **7 of 84 pre-2026-08-31 committed rows are code-boundary-crossed on the sticky-drop fix.**
- **`errors`, `commit` and `default_model` cannot be back-filled** onto any row before 2026-08-29.
- **`objection_closure`, `objections_resolved`, `objections_withdrawn` and
  `objections_falsely_resolved` cannot be back-filled** onto any row before the closure cell.
- **`CLOSURE_RULE` deliberately omits a bullet** saying `withdrawn` is the only restoring
  disposition.
- **`test_no_appended_rule_names_a_fixture_column` is the answer-injection guard** and must not be
  extended to cover the retry's `known_columns`.
- **The loop-cap default `3` is written in three places.**
- **`cmd_run` has no per-run `try/except`**, so an unhandled API error still ends a `--repeat` cell
  early. The harness fixed this for `eval` only.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN`).**
- **A stub named anything but `"stub"` slips past `PLACEHOLDER_MODEL_NAMES`.**
- **`materialize` does untranslated I/O** to preserve byte identity across platforms.
- **`load_fixture` raises `SystemExit` and `cmd_run` catches it**; the harness lets it propagate.
- **The trap fixtures' `mutual_info_with_target` is documentation, not a difficulty dial.**
- **A per-item rule on a response schema is a whole-response rule.** Still worth checking `intake`
  and `modeler`'s LLM-facing schemas.
- **The split manifest is still embedded in snippet text.**
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.**
- **The reviewer-off arm still runs the reviewer node** as a zero-cost no-op.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`.
- `ArtifactStore` copies the dataset per run and chmods it 0444.
- **`.mcp.json` hardcodes the toy dataset on argv.** Still wrong for Phase 5's generalist arm.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
