# Plan

Budget: roughly 15 to 25 Claude Code sessions. Phases 0 to 2 are plumbing; keep prompts tight and
do not let sessions explore.

## Phase 0: Session 0 -- done
Repo scaffolding, `state.py`, the toy fixture, the settings hook, first DECISIONS.md entries. See
DECISIONS.md 2026-08-22.

## Phase 1: Linear skeleton -- done
`intake -> profiler -> feature_eng -> modeler -> reporter` wired end to end behind a `Tools`
Protocol, with the router's seat reserved for Phase 3. See DECISIONS.md 2026-08-22 and 2026-08-26.

## Phase 2: MCP server -- done
Fork-per-snippet sandbox, artifact store, and metric log, swapped in behind the same Protocol and
verified from inside Claude Code itself. See DECISIONS.md 2026-08-27.

## Phase 3: Adversarial reviewer -- done
Reviewer node, loop cap, objection routing and closure, name-transparency and reviewer-prompt
ablations. The remediation bottleneck (objections dispatched to a node with no column lever) was
diagnosed and fixed; the closure arm ran and returned a recorded null. See DECISIONS.md 2026-08-27
and 2026-08-28 (five entries).

## Phase 4: Eval harness -- done
`harness.py`, results JSONL, `eval-diff`, the 13-dataset manifest (generated-only), the re-scorer
and baseline, and `--subset full` run live (52 rows, $1.2165). See DECISIONS.md 2026-08-29 through
2026-09-02.

## Phase 5: Ablations and writeup (open)

Entry state: `--subset full` has run, so tables draw from 52 benchmark rows plus 149 fixture rows.
Two constraints the writeup must not drop: no count in the 52-row file may be quoted as an effect
(n=2 per cell per replicate), and the benchmark rows are `leakage_graded: false` 52/52, so every
leakage headline rests on the fixtures, not the manifest.

- [x] All 13 manifest datasets run end to end, `--subset full`.
- [~] reviewer on/off, Haiku/Sonnet reviewer, single agent vs team, loop cap 1/3. Haiku/Sonnet and
      loop-cap ran (2026-08-28). Reviewer on/off and single-agent-vs-team are untouched.
- [x] README as a short paper, installed 2026-09-21; full write-up moved to docs/RESULTS.md,
      2026-09-23 cleanup.
- [ ] resume bullet with real numbers

Scope note 2026-09-21: reproduction became a phase 5 activity. `claims-repro` confirmed one Result
3 cell and surfaced a second, unresolved question: the profiler's opaque-arm recall has moved
across three dates (2/10, 5/10, 8/10), flagged rather than established. See DECISIONS.md
2026-09-21.

Scope note: `docs/explainers/pipeline-walkthrough.html`, built 2026-09-09, is documentation rather
than measurement and is not on the list above.

## Out of scope unless the core is done and credits remain
- non-tabular data, hyperparameter search, a web UI, TypeScript anything, Kubernetes anything
