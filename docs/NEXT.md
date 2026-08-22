# Next session

## Start here
Phase 0 is done and the floor is green: 33 fast tests in ~0.2s, `ds-agents run --dataset toy` exits
0, hook measured at 0.40s. The draft contract in ARCHITECTURE.md did NOT survive session 0. An
adversarial review (Opus, read-only) returned 8 findings and all were absorbed, so
`src/ds_agents/state.py` now differs substantially from the schema sketched in the original draft —
read the code, not your memory of the draft. The six 2026-08-22 entries in docs/DECISIONS.md explain
every change and why. ARCHITECTURE.md has been rewritten to match and now defers to state.py as
authoritative. Nothing in `nodes/`, `tools/`, or `mcp_server/` exists yet; Phase 1 is untouched.

Half-done: a `reviewer` subagent pass over the Phase 0 diff was launched and had not reported when
the session wrapped. Re-run it (`use the reviewer subagent on the current diff`, comparing against
`HEAD~1` or the session-0 commit) before building on the contract — it was specifically asked to
check `_strip_computed` recursion against unions, dict-valued model fields, and tuples, and to check
`results_row()` for divide-by-zero and None propagation. Those are the likeliest places a bug is
hiding.

## First prompt
Read CLAUDE.md, docs/ARCHITECTURE.md, and src/ds_agents/state.py. Then run the reviewer subagent on
the session-0 commit and fix anything real it finds. After that, start Phase 1: build the `intake`
and `profiler` nodes against `tools/local.py` (a shim with the same signatures the MCP server will
expose), with a unit test per node on the toy fixture. Follow /add-node. Do not wire the reviewer or
the router yet — that is Phase 3 — but do leave `review_iterations` and `review_verdict` untouched by
any node, since the router owns them.

## Open questions
- The router is now a real graph component (it owns `review_iterations` and derives
  `review_verdict`), but PLAN.md does not list it. Does it get built in Phase 1 as a pass-through, or
  does it appear in Phase 3 with the reviewer? Building it early keeps the loop contract honest.
- `verified_holdout_score` requires the harness to re-score outside the graph, which means the
  harness needs to load `feature_code_artifact` and re-apply it. That is new Phase 4 scope not in
  PLAN.md. Worth a line in the plan.
- Still unanswered from session 0: Docker-per-run vs one long-lived container; which OpenML suite has
  citable published baselines; LangChain `with_structured_output` vs the Anthropic API directly. The
  structured-output choice now matters more, because `Objection` has a required `columns` field and a
  model that cannot reliably fill it will look like a reviewer that missed the leak.
- `severity` on Objection is currently written but never read. Either the router blocks only on
  `high`, or it should be deleted rather than costing prompt tokens.

## Parking lot
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric.
- The gap between `claimed_holdout_score` and `verified_holdout_score` may deserve its own table in
  the writeup — "how often does the system overstate itself" is a finding independent of leakage.
- ruff formats Python code blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
  Harmless but surprising; scope the hook to `src/ tests/` if it becomes annoying.

## Owner to-dos (not doable by Claude)
- Apply the permissions allowlist: it is staged outside the repo (session-0 scratchpad). Claude is
  blocked from writing `.claude/settings.json` by design. It also carries a PATH export the hook
  needs, because `uv` lives in `/opt/homebrew/bin` and may be missing from a non-login shell.
