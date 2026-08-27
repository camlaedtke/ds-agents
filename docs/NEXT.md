# Next session

## Start here
**Phase 2 is closed.** The last box was "verify Claude Code can connect to the same server", and
it is now verified from inside Claude Code rather than by proxy: the four tools resolve as
`mcp__ds-agents-tools__*` and all four were driven against the toy dataset by an agent that shares
no code with `tools/mcp_client.py`. Worth knowing what that actually established, because it is
more than "it connects". `run_python` reported the repo absent from `sys.path` and no
`ANTHROPIC_API_KEY` in its environment, so the two isolation invariants hold for a client we did
not write. And a 202-byte artifact read with `max_bytes=40` came back `truncated: true`, then
opened whole from inside a snippet at the `sandbox_path` in its own metadata -- the escape hatch
is only worth having if the client that hits the cap can find it, and now that is demonstrated
from the outside. See DECISIONS.md 2026-08-27.

One residual, deliberately not treated as blocking: `claude mcp list` still prints "pending
approval" for the project-scoped entry, so a fresh terminal session prompts once before the tools
appear. That is a per-machine trust prompt about running `uv run mcp-server`, not a property of
the server. Click it if the prompt is in the way; nothing in Phase 3 waits on it.

The floor: 221 tests pass in 25.1s (full suite). Two live Haiku toy runs today, both exit 0 in
~15.9s at $0.0098, both dropping `account_status_code` and `customer_id`, both claiming roc_auc
0.843 and printing `publishable: yes` -- the same number the in-process runs and the first MCP
runs produced, which remains the point.

## First prompt
Read CLAUDE.md, docs/ARCHITECTURE.md and docs/PLAN.md, then start Phase 3 in plan mode: the
adversarial reviewer. Follow /add-node. The node reads everything the modeler wrote plus the
feature code only when `config.reviewer_sees_code`, and writes `objections`, `reviewer_claim` and
nothing else -- the router owns `review_verdict` and `review_iterations`, and `exhausted` is
derived, not claimed.

Settle the `ReviewPass` ownership problem before writing the node; it is the first thing the
reviewer collides with, and it is in the parking lot below with a recommended fix.

## Open questions
- **`customer_id` as a false positive.** Unchanged for four sessions and still unresolved. Every
  toy run flags it, correctly on the merits, and scores it as a false alarm because the manifest
  lists it under `id_columns` rather than `planted_leakage`, so `leakage_precision` reads 0.5
  every time. Options: an `acceptable_flags` set the precision metric forgives; count id columns
  as planted leakage; or publish 0.5 as the honest number and explain it. This changes a published
  column, which is why it keeps not being decided in passing. Phase 3 makes it worse, not better:
  the reviewer will have its own opinion about `customer_id` and there is no rule yet that says
  who is right.
- **The fast-test budget was 6.7s** against the hook's stated <10s at last measurement; the full
  suite is 25.1s. The three process-lifetime sandbox tests are ~3.8s of the fast total. Leaving
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
- **`.mcp.json` hardcodes the toy dataset on argv**, which is right for today (it is the only
  dataset) and wrong for Phase 5, where the generalist arm has to face each benchmark dataset in
  turn. The server already takes `--dataset` and `--dataset-id`, so the fix is whatever launches
  the single-agent arm writing the file per dataset, not a change to the server. Noting it here so
  it is not discovered as a surprise mid-ablation.
- Hint-injection ablation: tell the reviewer which categories of failure exist vs not.
- Cost-per-caught-leak as a headline metric. A toy run is ~$0.0098.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
  Harmless but surprising; scope the hook to `src/ tests/ mcp_server/` if it becomes annoying.
