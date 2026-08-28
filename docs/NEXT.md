# Next session

## Start here
**Phase 3's central question is answered and the answer is that the reviewer's leakage miss was a
prompt limit, not a capability limit.** The Haiku-versus-Sonnet arm was run as a 2x2 against a
second condition, `reviewer_prompt`, because the prompt was a confound under any model number.
27 live rows are committed at `evals/results/2026-08-28_reviewer-ablation.jsonl` ($1.04, all from
commit `2d5c1fc`), on `claims_timing --naming opaque` with Haiku upstream in every cell:

| cell | n | named >=1 planted trap | mean `reviewer_recall` | mean $ |
| ---- | - | ---------------------- | ---------------------- | ------ |
| haiku / base          | 10 | 1/10 | 0.05 | 0.015 |
| haiku / which_column  | 10 | 9/10 | 0.70 | 0.030 |
| sonnet / base         | 4  | 0/4  | 0.00 | 0.078 |
| sonnet / which_column | 3  | 2/3  | 0.67 | 0.093 |

The manipulation is one appended bullet in `REVIEWER_SYSTEM` (`WHICH_COLUMN_RULE` in
`nodes/reviewer.py`): if the claimed score is implausible, read `top_importances` from the *top* and
name the column that explains it. It names no fixture, column or trap type. That one rule moves
Haiku from 1/10 to 9/10; Sonnet under the base prompt is 0/4, no better than Haiku. Under `base`
both models object about the *number* (`metric_mismatch`, `overfit`) and never name a column --
`objections_by_category` makes this legible now. Upstream controls agree across all four cells
(`profiler_recall` 0.17-0.25, false alarm 1.0-1.33, trap survived 25 of 27), so nothing moved above
the reviewer and the cell differences are the reviewer's.

**Four things to know before touching anything.** First, **the bottleneck moved downstream**: 8 of
10 Haiku `which_column` runs end `exhausted`, so a 9/10 catch rate is a 1/10 remediation rate. The
reviewer names the trap, the loop cap expires before `feature_eng` removes it. That is now the most
actionable open problem and it is a different problem than the one this session started with.
Second, **the Sonnet cells are underpowered on purpose** -- Sonnet cost $0.078-0.093 per run against
a budgeted $0.035 (it blocks, so it runs three reviewer passes at ~$0.03 each) and the pre-registered
$1.20 cap bound them to n=4 and n=3. The model main effect is reported as *unresolved*, not absent.
Third, **one run worked end to end for the first time in the project**: sonnet/which_column row 1
named both traps, `feature_eng` dropped them, claimed roc_auc fell to 0.761 -- below the ~0.82
legitimate ceiling -- and the verdict was `pass`. Fourth, `results_row()` now carries
`reviewer_nominated/caught/recall/false_alarm` over all column-scoped categories plus
`objections_by_category`; `leakage_*` deliberately keeps its narrow two-category definition so the
naming-ablation rows stay comparable, and a test pins the divergence.

The floor: 412 tests pass in 66.6s full, ruff clean. `uv run ds-agents run --dataset toy` is green
live -- leak dropped, `publishable: yes`, verdict `pass`, $0.0137.

## First prompt
Read CLAUDE.md, docs/PLAN.md and the "Start here" above, then continue Phase 3 in plan mode on the
**remediation bottleneck**: the reviewer now names the planted trap in 9 of 10 opaque Haiku runs and
the trap is removed in 1 of them, because 8 of 10 runs hit the loop cap first. Diagnose before
building -- pull two or three `exhausted` runs' report artifacts from
`evals/results/2026-08-28_reviewer-ablation.jsonl` and establish *why* the loop does not converge
(is `feature_eng` ignoring a well-formed objection, re-adding the column, or never seeing it?),
since the three candidate fixes point in different directions: raise `loop_cap`, scope an objection
to a candidate, or add an `unactionable` disposition. A `loop_cap` sweep is the cheap first
measurement (1/3/5 at n=10, roughly $0.30 at Haiku prices with `--reviewer-prompt which_column`) and
it doubles as the Phase 5 loop-cap ablation. Also decide whether to spend ~$0.85 topping the Sonnet
cells to n>=10 this session or to defer it to Phase 5.

## Open questions

- **Why does the review loop not converge once the reviewer is right?** New and the most important.
  9/10 caught, 1/10 remediated, 8/10 `exhausted`. Unknown whether `feature_eng` mishandles the
  objection, the modeler re-promotes a candidate carrying the column, or the cap is simply too low
  for a two-trap fixture. Diagnose from artifacts before changing anything.
- **Two Haiku `which_column` runs hit a recoverable router error**, "reviewer claimed block with no
  open objection", seen in no other cell. The reviewer raised something the node then dropped
  (likely a column-scoped objection naming a column not in `known_columns`). Worth reading the
  reviewer's filtering loop against those two runs -- it may be silently discarding correct
  objections in the arm that raises the most of them.
- **Should the Sonnet cells be topped up to n>=10?** ~$0.85. Until then the model main effect is
  unresolved, and the writeup can only say "no evidence Sonnet helps under the base prompt at n=4".
  Never top up after inspecting a cell's outcome fields; decide the n first.
- **A legitimate objection with no available remedy is still not distinguishable from a false
  alarm.** Unchanged from two sessions ago, and now much more consequential: it is a candidate
  explanation for the 8 `exhausted` runs above. Options unchanged -- scope an objection to a
  candidate, teach the modeler to drop a candidate in response, or add an `unactionable`
  disposition.
- **`customer_id` as a false positive is measured and it is every run.** `profiler_false_alarm` is
  a stable ~1.0-1.3 per run in every cell of both ablations. Options unchanged (an `acceptable_flags`
  set, count id columns as planted, or publish and explain).
- **Should `reissued_ids` get the naming treatment?** The name effect replicated on a second,
  structurally different trap would be a much stronger claim. ~$0.70 and 20 minutes. Deliberately
  scoped out twice now.
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

- **The prompt arm roughly doubles wall time and cost** (1.2 to 2.7 mean loops). If `which_column`
  becomes the default prompt, every downstream cost estimate in PLAN.md is low by ~2x.
- **The hint-injection ablation now has a drawn boundary to cross.** `WHICH_COLUMN_RULE` points only
  at a field the reviewer already receives and names no trap type; the parked hint-injection arm is
  the version that deliberately tells the reviewer which *categories* of failure exist. Label it as
  such when it runs -- the distinction is written up in DECISIONS.md 2026-08-28.
- **Cost-per-caught-leak is now computable** and would be a good headline metric: $0.015/run at
  1/10 versus $0.030/run at 9/10 is a 4.5x improvement in dollars per catch.
- **`--results` is a stopgap and should be absorbed by Phase 4's harness**, not extended. It appends
  `results_row()` behind `publishable()` and knows nothing about subsets, baselines or datasets. It
  now also has no commit field, so the SHA lives only in `evals/results/LOG.md` by hand.
- **`cmd_run` has no per-run `try/except`**, so an unhandled API error ends a `--repeat` cell early.
  Already-appended rows survive, so recovery is re-invoking with `--repeat <remaining>`. Retry
  policy belongs in Phase 4's harness, not here.
- **The opaque arm's numbering is dense and positional (`var_01..var_NN` in column order).** It
  leaks nothing today, but a fixture with traps in a fixed position could become a learnable cue. A
  seeded shuffle would fix it at the cost of readability by eye.
- **A stub named anything but `"stub"` slips past `PLACEHOLDER_MODEL_NAMES`.** Only constructible in
  a test (the CLI never names a stub), and `test_the_reviewer_model_binds_to_the_reviewer_node_only`
  now does exactly that on purpose.
- **`materialize` does untranslated I/O** (`newline=""` on both sides) to preserve byte identity for
  a CSV generated on another platform. No committed fixture uses CRLF; the test does.
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
- **The split manifest is still embedded in snippet text**, and the 1 MiB read cap makes it urgent
  at Phase 4 sizes. Fix is `nodes/feature_eng.py`, `nodes/modeler.py` and their tests.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.**
- **The reviewer-off arm still runs the reviewer node** as a zero-cost no-op. `results_row()` now
  reports `None` rather than 0.0 for the `reviewer_*` fields in that arm.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
  Restrict to the best-by-CV candidate at Phase 4 sizes.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`.
- `ArtifactStore` copies the dataset per run and chmods it 0444. Wasteful for a benchmark set.
- **`.mcp.json` hardcodes the toy dataset on argv.** Still wrong for Phase 5's generalist arm.
- **`.claude/skills/run-eval/SKILL.md` documents `ds-agents eval --subset ci` and `eval-diff`,
  neither of which exists** (`cmd_eval` prints "lands in Phase 4" and returns 2). Fix the skill or
  build the harness; a skill that names commands that do not exist will mislead a future session.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
