# Architecture

Pressure-tested in session 0 against an adversarial review. `src/ds_agents/state.py` is the
authoritative contract; this file explains the reasoning and the parts that live outside the code.

## Graph

```
intake -> profiler -> feature_eng -> modeler -> reviewer -> reporter
                        ^              ^            |
                        |______________|____________|  (router may send back, max N loops)
```

The reviewer is adversarial: it can block a run and send it back to `feature_eng` or `modeler`
with a structured objection. Two things it does **not** control:

- **It does not increment the loop counter.** The router does. A model under test cannot be
  trusted to enforce its own cap.
- **It does not write the verdict.** It writes `reviewer_claim` (`pass` or `block`). The router
  derives `review_verdict`: a claim of `block` while `review_iterations >= loop_cap` becomes
  `exhausted`. Otherwise a reviewer that simply emits "pass" at the cap would be indistinguishable
  from one that passed on merit, and `exhausted` is the outcome DECISIONS.md calls interesting.

## State

See `src/ds_agents/state.py`. Two rules shape it:

1. **The system under test does not report its own grade.** The modeler writes
   `chosen_model.claimed_holdout_score`. The harness writes `verified_holdout_score`, scored
   outside the graph on a holdout the agents never see. The gap between them is a finding, not an
   error.
2. **Anything the eval counts is a field, never prose.** `Objection.columns` exists because
   `leakage_caught` must be a set comparison against ground truth. Computing it by string-matching
   a model's sentence would score a leakage objection naming the *wrong* column as a catch.

Invariants worth stating explicitly:

- `RunConfig` is frozen. It carries `run_id`, `arm`, `reviewer_enabled`, `reviewer_model`,
  `default_model`, `loop_cap`, `reviewer_sees_code`, `naming`, `reviewer_prompt`,
  `objection_routing`, `objection_closure`, `random_seed` and `dataset_hash`.
  Every results row is self-describing from the state object alone — without it, "reviewer
  disabled" and "reviewer crashed" are the same row, and a descriptive run and an opaque one over
  byte-identical rows are the same row too.
- `split_artifact` is pinned by the profiler before `feature_eng` runs and never rewritten. Without
  a pinned split the `contamination` category is unfalsifiable and no run is reproducible. It
  partitions the rows the AGENTS were given, and is *not* the independent holdout behind
  `verified_holdout_score`, which is withheld before the graph starts and never appears in this
  manifest.
- `objections`, `review_passes`, `errors`, and `node_trace` are append-only, annotated with
  `operator.add` so LangGraph merges rather than replaces. An unannotated list field would be
  overwritten by whichever node wrote last, silently emptying the cost table.
- Objections are never mutated. Resolution is recorded as a `ReviewPass.dispositions` entry, so
  "the reviewer withdrew it" stays distinguishable from "the reviewer never looked again."
- `final_features` records what actually reached the model, so *reviewer caught the leak* and
  *the leak was removed* are two different numbers.

Objection categories (small and enumerable so they are countable): `leakage`, `contamination`,
`overfit`, `metric_mismatch`, `implausible_importance`, `spec_violation`, `other`. Every objection
also carries a free-text `subcategory`, so a reviewer catching something unanticipated is not
forced to mislabel it.

## Node contracts

| node | reads | writes | tools | routing |
|---|---|---|---|---|
| intake | dataset_id, task_description | spec | read_artifact | profiler |
| profiler | spec, dataset_id, config.random_seed | profile, split_artifact | run_python | feature_eng |
| feature_eng | spec, profile, split_artifact, task_description, open objections | feature_code_artifact, feature_summary, final_features, dropped_features | read_artifact, run_python | modeler |
| modeler | spec, feature_code_artifact, split_artifact, final_features, task_description, config.random_seed, config.run_id, open objections | candidates, chosen_model, importance_artifact, top_importances | read_artifact, run_python, log_metric | reviewer |
| reviewer | everything above; feature code only if `config.reviewer_sees_code` | objections, reviewer_claim, reviewer_dispositions | read_artifact | router decides |
| router | reviewer_claim, reviewer_dispositions, review_iterations, config.loop_cap, config.reviewer_enabled, open objections | review_iterations, review_verdict, review_passes | none | feature_eng / modeler / reporter |
| reporter | everything except the ground-truth fields below | report_artifact | write_artifact | END |

No node writes `planted_leakage_columns`, `verified_holdout_score`, or `baseline_score`. Those are
harness-written ground truth; a node that could see them could game them. The reporter additionally
must not *render* them: it receives the whole state, and writing the answer key into a report
artifact would leak it into the store that the Phase 5 single-generalist arm reads.

### One vocabulary: source column names

`final_features`, `dropped_features`, `top_importances`, `Objection.columns`,
`LeakageCandidate.column`, and `planted_leakage_columns` all hold **original** column names, never
one-hot expansions. `results_row()` computes `leakage_remediated` as
`not (planted & set(final_features))`, so storing `region=north` in `final_features` would make
that intersection permanently empty and score every run as remediated with the leak still present.
Expanded matrix names live in exactly two places, neither compared against ground truth:
`FEATURE_ORDER`/`COLUMN_SOURCE` inside `feature_code_artifact`, and `matrix_columns` inside
`importance_artifact`.

### `review_verdict` values, including the awkward one

`pending` (nobody adjudicated -- the reviewer-off arm, or the reviewer crashed), `pass`,
`exhausted` (the reviewer still wanted to block when the loops ran out), and `block`. `block` is
normally transient, but it is reachable as a *terminal* verdict in one case: the reviewer claims
`block` while no objection is open, so there is nothing for feature_eng or modeler to act on and
the router sends the run to the reporter with an error recorded. A results row can therefore end
on `block`, and reading it as "still mid-loop" would be wrong. It always comes with a
`PipelineError` from the router, which is how to tell the two apart.

### Where the loop cap is actually enforced

The router increments `review_iterations` before checking it, then derives the verdict; the graph
edge `route_target()` reads that *written verdict* and sends an `exhausted` run to the reporter.
So the cap is enforced by the two together, not by a clamp inside the router. That is deliberate:
a defensive clamp would let a mis-wired `graph.py` keep looping while quietly reporting a
compliant `review_iterations`, turning a topology bug into a wrong number instead of a crash.

### `ReviewPass` has one writer, and `routed_to` has one authority

The router mints the whole `ReviewPass`, including the parts it did not compute itself. `routed_to`
is known only to the router (it is the sole reader of `reviewer_dispositions`); `dispositions` is
known only to the reviewer. `review_passes` is `operator.add`, so if both nodes appended their own
partial pass, every iteration would land two records instead of one. The reviewer hands its half
across on `reviewer_dispositions` instead: one writer, one reader, deliberately not `operator.add`
-- the reviewer overwrites it wholesale on every pass, because it is a handoff for the invocation in
progress, not a history the way `objections` and `review_passes` are.

`routed_to` gets a single authority for the same reason `review_verdict` does. The router computes
the destination against the *projected* post-pass open set -- `open_objections()` minus whatever
this pass just resolved or withdrew -- and writes the result onto the pass it mints; `route_target`
reads that back rather than re-deriving it, extending the existing "`route_target` reads the written
verdict, never re-derives" rule to `routed_to` as well. This is not belt-and-suspenders: a second,
independent derivation would genuinely disagree with the first. The router computes against the
state as of its own invocation, before this pass's update is merged; the edge function runs after
that merge. A pass that resolves a `feature_eng` objection while raising a new `modeler` one would
have the router see the still-open `feature_eng` objection and the edge see the merged state with
only the new `modeler` objection standing -- two different, both locally correct, destinations. The
whole scheme rests on one invariant: `review_verdict == "block"` implies a `ReviewPass` was minted
on the same invocation, so `route_target` always has a pass with a `routed_to` to read.

**Both handoff fields are rewritten on every reviewer return path** -- a completed pass, a failed
model call, and the disabled no-op all set `reviewer_claim` and `reviewer_dispositions` explicitly,
never leaving either at whatever it held from the previous pass. The router only reads
`reviewer_claim`; it has no way to tell a fresh value from a stale one. Without the rewrite, a
pass-2 crash would leave pass 1's `block` sitting on the field, and the router would count it a
second time. A crashed pass has to read as no pass -- `pending`, not a replay.

**A disabled reviewer is a no-op inside the node, not a graph conditional**, so `graph.py` stays
logic-free either way. That means the reviewer-off arm's `node_trace` still contains a `reviewer`
row -- just a zero-cost one, `model: None`. Anything comparing the reviewer-on and reviewer-off arms
has to read `model` or `cost_usd` off that row, not the presence of `"reviewer"` in the trace, or
the two arms look identical on a check that was only ever asking whether the graph reached the node.

A node's signature is `node(state, *, tools, model) -> dict`. It returns a **narrow dict of the
fields it changed**, never the state object: with `operator.add` on the history fields, returning
state would concatenate the accumulated trace onto itself. `graph.py` binds `tools` and `model`
with `functools.partial`, which is also what keeps the Phase 2 MCP swap confined to that one file.

Two things the table does not show:

- **Intake sees the dataset as an artifact.** It is registered at run start under
  `dataset:<dataset_id>` with columns, dtypes, row count, and per-column cardinality in its
  metadata, so intake can name a target without being given `run_python`.
- **A snippet can create an artifact without calling `write_artifact`.** Anything a snippet drops
  in `DS_ARTIFACTS` is detected and indexed when `run_python` returns, and its id comes back in
  `artifacts_written`. That is how the profiler's split manifest is written without shipping the
  whole row-id list back through stdout. Detection is by content change, not by filename, and the
  file is copied under its artifact id: an artifact is immutable once issued, or `split_artifact`
  being "pinned and never rewritten" means nothing. **Phase 2's server must keep this behaviour.**
- **The profiler refuses split strategies it has not implemented.** `temporal` and `grouped` are
  valid on `TaskSpec`; the snippet handles `random` and `stratified`. Falling back to a random
  split while labelling the manifest `grouped` would put one entity on both sides of the partition
  while the file claims otherwise, which is the exact contamination the manifest exists to make
  falsifiable. It writes no manifest and records an unrecoverable error instead.
- **The profiler is split in two.** The mechanical statistics -- including each column's
  normalized mutual information with the target -- come from a fixed snippet through `run_python`.
  The model is asked only which of those numbers mean "this column encodes the answer" rather than
  "this column is a strong legitimate predictor", which is the part the eval measures.

## MCP server (ours)

Nodes type-hint against the `Tools` Protocol and never import a concrete implementation, so both
arms of the single-agent-vs-team ablation are guaranteed the same surface and the swap to the MCP
client does not reach into `nodes/`.

`mcp_server/store.py` holds the artifact store and the metric log; `mcp_server/sandbox.py` and
`mcp_server/_worker.py` hold the sandbox. `src/ds_agents/tools/local.py` binds those two objects
to the `Tools` Protocol in-process, `mcp_server/server.py` wraps *that same object* in the
protocol, and `src/ds_agents/tools/mcp_client.py` satisfies the Protocol by calling the server. So
the same implementation is reached two ways and "we swapped the shim for the server" cannot quietly
mean "we wrote a second implementation and hoped it matched." `ds-agents run` defaults to
`--tools mcp`; `--tools local` skips the process boundary for debugging.

One server process per run, launched over stdio with the run's root and dataset on argv. The store
is run state and the sandbox worker is not, so a shared server would have to key stores by
`run_id` — which needs either a fifth tool or a `run_id` argument on the other four, and the tool
surface is the thing the single-agent-vs-team ablation holds constant. See DECISIONS.md 2026-08-27.

MCP has one failure channel, and in-process there are two that mean opposite things: a `ToolError`
is a finding about the agent and a node records it, a `SandboxError` is a finding about us and
propagates. `SANDBOX_ERROR_PREFIX` is how the second keeps its type across the wire, so a broken
worker is not filed in the results table as an agent mistake.

### How the sandbox runs a snippet

One worker process per interpreter, launched with an environment built from nothing. It imports
pandas and scikit-learn once, then `fork()`s a fresh child for each snippet; the child redirects
fds 0, 1 and 2, calls `setsid`, replaces `os.environ` wholesale, `chdir`s into the run's work
directory, and `exec`s the code. Every snippet therefore gets a genuinely separate process — no
globals, no imported modules, and no `sys.path` edits survive from one to the next — while paying
the library imports once. Measured: 0.898s per call before, 0.04s after. See DECISIONS.md
2026-08-27 for why this beats both Docker options, and why it does not violate the CLAUDE.md rule
against `exec` in-process.

Four properties are enforced rather than assumed, each with a test in `tests/mcp_server/`:

- **The worker never imports `ds_agents`.** The child inherits the worker's memory, so the
  worker's import list is part of the boundary. `PipelineState` — and `planted_leakage_columns`
  with it — lives in the orchestrator process, which is not an ancestor of any snippet.
- **The repo is pruned off `sys.path`.** An editable install puts the repo root and `src/` on the
  path through a `.pth` file in site-packages, which no amount of environment stripping removes;
  the Phase 1 shim's snippets could `import ds_agents` despite its docstring saying otherwise.
  `_prune_sys_path` keeps only the standard library and site-packages.
- **The protocol channel is not writable by a snippet.** Requests and replies are line-delimited
  JSON on the worker's stdin/stdout. The child redirects its fds before running anything, so a
  snippet cannot forge a reply, and stdin points at `/dev/null` so it cannot eat the next request.
- **A timeout kills the process group.** `setsid` in the child means a snippet that spawned
  helpers cannot leave them running past the deadline. The worker survives the kill and answers
  the next request.

What it still is not: a container. There is no memory cap and no network block, because every
snippet through Phase 2 is templated by us. Both become load-bearing when model-authored feature
code runs, and `SandboxPool` is the seam a Docker backend would land behind.

The worker is shared process-wide rather than created per run, because it holds no run state: the
environment a snippet sees, its working directory, and its output paths all ride on the request.

Tools exposed:

- `run_python(code: str, timeout_s: int) -> {stdout, stderr, exit_code, artifacts_written}`
  Runs in a subprocess inside the sandbox container with the dataset mounted read-only and an
  artifacts dir mounted read-write. No network.
- `read_artifact(id, max_bytes) -> {content, meta, truncated}`
  Capped at `DEFAULT_READ_BYTES` (1 MiB) even when the caller names no limit, in the store rather
  than in the transport, so the two bindings truncate at the same byte. A large artifact is meant
  to be opened from inside a snippet at `meta.extra["sandbox_path"]`, not pulled through the tool.
- `write_artifact(name, content, kind, extra) -> meta`
  Returns the whole `ArtifactMeta` over the wire, not just the id: the id is in it, and an
  external client gets `n_bytes` and `sandbox_path` without a second call. The MCP client takes
  `.id` and satisfies the Protocol, which asks for an id.
- `log_metric(run_id, name, value)` — `run_id` comes from `state.config.run_id`.

Why MCP and not plain Python functions: the agents then have exactly the same tool surface as any
external MCP client, which makes the "single generalist agent vs team" ablation honest, and lets us
point Claude Code at the same server as a comparison. `.mcp.json` at the repo root is that
configuration; a hand-rolled JSON-RPC client with no SDK completes the handshake, lists the four
tools and gets real results back, so the surface is standard rather than a private dialect our own
client happens to speak.

## Observability

LangSmith tracing on by default (env var). Every node appends a `NodeEvent` to `node_trace` with
token counts and cost so results files do not depend on the tracing backend. All timestamps are
timezone-aware UTC; naive ones make committed JSONL inconsistent across machines.

## Eval outcomes per dataset

Produced by `PipelineState.results_row()`. If a number in the published tables cannot be traced to
that method, it is not a real number.

Scores: `claimed_holdout_score`, `verified_holdout_score`, `holdout_claim_gap`, `baseline_score`,
`score_ratio` (direction-aware — a ratio means the opposite thing for RMSE and AUC), `metric`.

Leakage, as a set comparison against ground truth rather than two booleans: `leakage_planted`,
`leakage_flagged`, `leakage_caught`, `leakage_remediated`, `leakage_recall`, `leakage_precision`,
`false_alarm_columns`, `false_alarm`, plus `leakage_flagged_standing` and `false_alarm_standing`.
The `_standing` pair drops columns whose every objection was later resolved or withdrawn: catching
a leak at any point is a genuine catch, but taking back a false alarm is better behaviour than
leaving it standing, and one number cannot say both. A binary pair cannot express "caught the real one and also
flagged three clean columns."

Profiler nominations, scored separately from the reviewer's objections: `profiler_nominated`,
`profiler_caught`, `profiler_recall`, `profiler_false_alarm`. Separate because the two nodes give
different answers to the same run — measured on 2026-08-27, the profiler nominates the planted
column and the reviewer, shown the result, says nothing — and one combined `leakage_caught` would
report a team that catches leaks while hiding which member caught it. All four are null rather than
zero when `profile` is None: a profiler that crashed nominated nothing in a different sense than one
that looked and declined.

Conditions: `arm`, `reviewer_enabled`, `reviewer_model`, `reviewer_sees_code`, `loop_cap`,
`naming`, `reviewer_prompt`, `objection_routing`, `objection_closure`, `random_seed`, straight off
the frozen `RunConfig`.

Loop: `review_verdict`, `review_loops`, `objections_raised`, `objections_by_category`,
`objections_open_at_end`, and -- added 2026-08-28 with the closure axis, because
`objections_open_at_end` conflates two opposite claims about the reviewer -- `objections_resolved`,
`objections_withdrawn` and `objections_falsely_resolved`, the last being an objection marked
`resolved` while one of its columns is still in `final_features`. Plus four fields that say why a
caught leak was not fixed, added 2026-08-28 after 9-of-10 caught turned out to be 1-of-10
remediated:
`objections_by_target_node` (who the reviewer asked to act -- an objection addressed to `modeler`,
which has no column lever, is a correct finding that cannot land), `objected_columns_unremediated`
(column-scoped objected columns still in `final_features`; `None` when no pass completed or the
matrix is empty), `route_sequence` and `new_objections_per_pass` (where the loop actually went, and
whether each pass re-raised the same objection or found a new one). All four are derived from
`objections`, `review_passes` and `final_features` -- no node records its own remediation.

Cost and reliability: `wall_seconds` (real elapsed, not the sum of node events), `cost_usd`,
`errored`. The harness must emit a row for every dataset even on hard failure, or the hardest
datasets disappear and every table biases upward.
