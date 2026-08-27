# Next session

## Start here
**Phase 2 is about half done: the sandbox and the store are real, the MCP protocol layer is not.**
The session settled the Docker-versus-long-lived-container question by measuring it, and the answer
was neither. `mcp_server/` now holds `_worker.py` (a long-lived worker that imports pandas and
scikit-learn once, then `fork()`s a fresh child per snippet), `sandbox.py` (worker lifetime, the
line-delimited JSON request protocol, the outer deadline), and `store.py` (the artifact store and a
metric log that writes JSONL). `src/ds_agents/tools/local.py` is no longer a standalone shim; it is
a thin binding of those same two objects to the `Tools` Protocol, so the MCP client can bind the
same objects over the wire rather than reimplementing them. Full reasoning in DECISIONS.md
2026-08-27.

The floor: 190 fast tests in 6.4s, 202 in the full suite, `uv run ds-agents run --dataset toy`
exits 0 in ~15s at about $0.009. Per-snippet cost went from 0.898s (a cold interpreter importing
pandas and scikit-learn) to 0.04s including a RandomForest fit, against a one-time 0.87s worker
boot that is now paid once per *process* rather than once per run — which is why it matters much
more at Phase 4 sizes than it does on toy, where the 16s → 15s change is mostly Haiku latency.

**The one finding that changes something already written down.** The Phase 1 shim's docstring
claimed a snippet "cannot see the repo it is being graded in" because its environment was built
from scratch. That was false. An editable install writes a `.pth` file into site-packages that puts
`/ds-agents` and `/ds-agents/src` on `sys.path` at interpreter startup, and a `.pth` is honoured
regardless of the environment — so every snippet run in Phase 1 could `import ds_agents`. Nothing
handed one the live `PipelineState`, so no number published so far is wrong, and the two ablation
arms were equally affected. But "the agent read the grading code" and "the agent reasoned about the
columns" produce the same `leakage_recall`, and that is the distinction the project exists to make.
`_worker._prune_sys_path` now keeps only the standard library and site-packages;
`test_the_repo_is_not_importable_from_a_snippet` is the regression test. See LEARNING.md
`sys-path-is-not-the-environment`.

**Nine live Haiku runs today, and the known behaviour replicated.** The profiler flagged
`account_status_code` every time. `feature_eng` acted on it in 7 of 9 (previously 8 of 9); the two
that did not kept the leak and claimed roc_auc 0.983 against 0.843 for the runs that dropped it.
Same gap, same numbers, no drift attributable to this session — the sandbox change is
deterministic and nothing model-facing was touched.

## First prompt
Read CLAUDE.md, docs/ARCHITECTURE.md ("MCP server (ours)", which now has a "How the sandbox runs a
snippet" subsection), and docs/DECISIONS.md 2026-08-27. Then finish Phase 2: write
`mcp_server/server.py` exposing `run_python`, `read_artifact`, `write_artifact` and `log_metric`
over MCP, wrapping the existing `SandboxPool` and `ArtifactStore` rather than reimplementing them;
add the `mcp-server` console script to pyproject; write `src/ds_agents/tools/mcp_client.py`
satisfying the same `Tools` Protocol; and confirm `uv run ds-agents run --dataset toy` is still
green through the client. Then point Claude Code at the same server and check it connects, which is
the generalist-agent baseline for Phase 5.

Two things to decide as you go. First, whether the server owns one `ArtifactStore` per run keyed by
`run_id` or one process per run — the sandbox worker is already shared process-wide and holds no
run state, but the store does. Second, what `read_artifact` returns over the wire for a large
artifact, since `ArtifactPayload.truncated` exists precisely so a node cannot render a partial
artifact into a prompt as if it were whole.

## Open questions
- **The fast-test budget went from 1.6s to 6.4s.** Still inside the hook's stated <10s, but the new
  sandbox tests cost about 4s of it: a deliberate 1s timeout, a 1s process-group kill, a worker
  reboot, and the one-time 0.87s boot. Those are also the tests most worth running on every edit.
  Leave it, or move the three process-lifetime ones out of `fast` and accept they only run in the
  full suite.
- **`customer_id` as a false positive.** Unchanged from last session and still unresolved. Every
  toy run flags it, correctly on the merits, and scores it as a false alarm because the manifest
  lists it under `id_columns` rather than `planted_leakage`, so `leakage_precision` reads 0.5 every
  time. Options: an `acceptable_flags` set the precision metric forgives; count id columns as
  planted leakage; or publish 0.5 as the honest number and explain it. This changes a published
  column.
- **`NodeEvent` records no duration.** Cost per node is there, wall time per node is not, so "where
  do the 15 seconds go" can only be answered by timing the process from outside. Worth a field
  before Phase 4 makes wall time a published column.
- **Does `feature_eng` deserve its own reviewer-independent retry?** Still open, still argues
  against itself: measuring the reviewer's effect is cleaner if the baseline is left alone.
- **LangSmith is wired but never exercised.** Unchanged. `@traceable` on `AnthropicModel.generate`,
  inert with no key. Treat "tracing works" as unverified until someone adds the key.
- Which OpenML suite has citable published baselines. Open since session 0.
- `ModelResult` has no field for the modeler's `rationale` or a per-candidate `fit_error`.

## Parking lot
- **The split manifest is still embedded in snippet text, but the fix is now unblocked.** Every
  artifact carries `sandbox_path` in `ArtifactMeta.extra`, and `ArtifactStore.path_of()` returns
  it, so a snippet can `open()` the manifest instead of having it rendered into its source. The
  remaining work is in `nodes/feature_eng.py` and `nodes/modeler.py` and their tests, which is why
  it was not done here. Still O(n_rows) until it is: roughly 7 MB of snippet on a 100k-row dataset.
- **Docker is deferred, not rejected, and `SandboxPool` is the seam.** What it buys that the
  current sandbox does not is a memory cap and a network block. Neither matters while every
  snippet is templated by us; both matter the moment model-authored feature code runs.
- **`ReviewPass` has a split ownership problem, and Phase 3 will hit it immediately.** It carries
  `dispositions` (only the reviewer knows them) and `routed_to` (only the router knows it), and
  `review_passes` is `operator.add` — so if both nodes append, every iteration produces two
  records. Recommended fix when the reviewer lands: the router writes the whole `ReviewPass`, and
  the reviewer hands dispositions across on a new narrow field. Do not add that field before it
  has a writer.
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
- Cost-per-caught-leak as a headline metric. A toy run is ~$0.009.
- ruff formats Python blocks inside `docs/*.md`, so the hook rewrites design docs on every edit.
  Harmless but surprising; scope the hook to `src/ tests/ mcp_server/` if it becomes annoying.
