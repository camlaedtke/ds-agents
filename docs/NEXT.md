# Next session

## Start here
**Phase 2 is done apart from one keystroke.** The four tools now cross the MCP protocol.
`mcp_server/server.py` constructs a `LocalTools` -- the same binding of `SandboxPool` and
`ArtifactStore` the graph has used since Phase 1 -- and exposes its four methods as MCP tools, so
the protocol layer decides nothing about how a snippet runs or when a read truncates.
`src/ds_agents/tools/mcp_client.py` satisfies the same `Tools` Protocol from the other side, and
`ds-agents run` now defaults to `--tools mcp` (`--tools local` skips the process boundary for
debugging). Two design choices worth knowing before touching it, both in DECISIONS.md 2026-08-27:
one server process per run, because a shared server would need a fifth tool to open a run and the
tool surface is what the Phase 5 ablation holds constant; and `read_artifact` is capped at 1 MiB
even with no `max_bytes`, in the store rather than in the transport, so the two bindings cannot
disagree about whether an artifact was whole.

The floor: 203 fast tests in 6.7s, 221 in the full suite, `uv run ds-agents run --dataset toy`
exits 0 in ~15.8s at $0.0096. Five live Haiku runs today through MCP plus one through
`--tools local`: `feature_eng` dropped `account_status_code` in all six and claimed roc_auc 0.843
every time, the same number the in-process runs produced last session -- which is the point, since
the transport is supposed to be the only thing that changed. The test named
`test_the_same_run_over_mcp_lands_in_the_same_place` asserts that parity end to end with the stub:
same spec, same features, same chosen model, same claimed score to the digit.

**The one thing left on the Phase 2 checklist.** `.mcp.json` is committed and points Claude Code
at the same server, but `claude mcp list` reports it as "pending approval" -- a project-scoped MCP
server needs one interactive `claude` session to accept, which a non-interactive session cannot
do. Conformance was verified the other way instead: a hand-rolled JSON-RPC client with no SDK on
its side completes the handshake, lists the four tools and gets real results back, so the surface
is standard rather than a dialect our own client happens to speak. Start the next session by
running `claude`, approving the server, and confirming `claude mcp list` says connected.

**A real bug the reviewer caught, worth knowing because it was invisible.** `feature_eng` refused
to act on a truncated split manifest and `modeler` did not. That gap was dead code while an
unbounded read was possible and became a silent wrong number the moment the default cap landed:
the fitting snippet would have trained and scored on a subset of the pinned split while reporting
`claimed_holdout_score` as if it were the holdout. Both guards now exist with a test each. The
second finding was an orphaned server subprocess on a failed connect -- one per failed dataset in
a benchmark, each holding a sandbox worker with pandas resident. Both fixed; the regression test
for the second was confirmed to fail against the old code.

## First prompt
Read CLAUDE.md, docs/ARCHITECTURE.md and docs/PLAN.md. First, close out Phase 2: approve the
project MCP server (`claude` interactively, then `claude mcp list` should show `ds-agents-tools`
connected) and tick the last box. Then start Phase 3 in plan mode -- the adversarial reviewer.
Follow /add-node. The node reads everything the modeler wrote plus the feature code only when
`config.reviewer_sees_code`, and writes `objections`, `reviewer_claim` and nothing else: the
router owns `review_verdict` and `review_iterations`, and `exhausted` is derived, not claimed.
Before writing the node, settle the `ReviewPass` ownership problem in the parking lot below --
`dispositions` and `routed_to` currently have different owners on a field annotated
`operator.add`, so if both nodes append, every iteration writes two records.

## Open questions
- **`customer_id` as a false positive.** Unchanged for three sessions and still unresolved. Every
  toy run flags it, correctly on the merits, and scores it as a false alarm because the manifest
  lists it under `id_columns` rather than `planted_leakage`, so `leakage_precision` reads 0.5
  every time. Options: an `acceptable_flags` set the precision metric forgives; count id columns
  as planted leakage; or publish 0.5 as the honest number and explain it. This changes a published
  column, which is why it keeps not being decided in passing.
- **`NodeEvent` records no duration.** Cost per node is there, wall time per node is not. Worth a
  field before Phase 4 makes wall time a published column. Cheap now, annoying to backfill.
- **The fast-test budget settled at 6.7s** against the hook's stated <10s, and the MCP tests cost
  almost none of it (the in-process client tests are milliseconds; everything that spawns is
  outside `fast`). The three process-lifetime sandbox tests are still ~3.8s of the total. Leaving
  them in `fast` is the current call; revisit only if the hook starts feeling slow.
- **Does `feature_eng` deserve its own reviewer-independent retry?** Still open, still argues
  against itself: measuring the reviewer's effect is cleaner if the baseline is left alone.
- **LangSmith is wired but never exercised.** `@traceable` on `AnthropicModel.generate`, inert
  with no key. Treat "tracing works" as unverified until someone adds the key.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot
- **`ReviewPass` has a split ownership problem, and Phase 3 hits it immediately.** It carries
  `dispositions` (only the reviewer knows them) and `routed_to` (only the router knows it), and
  `review_passes` is `operator.add` -- so if both nodes append, every iteration produces two
  records. Recommended fix when the reviewer lands: the router writes the whole `ReviewPass`, and
  the reviewer hands dispositions across on a new narrow field. Do not add that field before it
  has a writer.
- **The split manifest is still embedded in snippet text, and the 1 MiB read cap made it
  urgent rather than merely wasteful.** Every artifact carries `sandbox_path` in
  `ArtifactMeta.extra` and `ArtifactStore.path_of()` returns it, so a snippet can `open()` the
  manifest instead of having it rendered into its source. Until that lands, a Phase 4 dataset
  around 100k rows produces roughly 7 MB of manifest, which now trips the cap and fails the run
  loudly in `feature_eng` and `modeler` instead of quietly bloating a snippet. Loud is better, but
  the fix is the same fix: `nodes/feature_eng.py`, `nodes/modeler.py` and their tests.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.** What it buys that the
  current sandbox does not is a memory cap and a network block. Neither matters while every
  snippet is templated by us; both matter the moment model-authored feature code runs.
- `permutation_importance` costs `n_source_columns x n_repeats` scoring passes per candidate.
  Trivial on toy; at Phase 4 sizes restrict it to the best-by-CV candidate or drop `n_repeats` to 5.
- `_strip_value` in `state.py` returns on the first `BaseModel` in `get_args`, so a future
  `dict[str, SomeModel]` field would round-trip wrong. No such field exists; fix is to dispatch on
  `get_origin`.
- `ArtifactStore` copies the dataset per run and chmods it 0444. Fine for 200 rows, wasteful for a
  benchmark set.
- `feature_eng` picks columns but does not write code. Revisit once a Docker backend gives
  model-authored feature code a smaller blast radius.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric. A toy run is ~$0.0096.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
  Harmless but surprising; scope the hook to `src/ tests/ mcp_server/` if it becomes annoying.
