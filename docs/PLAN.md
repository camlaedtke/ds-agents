# Plan

Budget: roughly 15 to 25 Claude Code sessions. Phases 3 and 4 get the deep sessions.
Phases 0 to 2 are plumbing; keep prompts tight and do not let sessions explore.

## Phase 0: Session 0 (this one)
- [x] Pressure-test docs/ARCHITECTURE.md. Done as an adversarial review rather than an interview;
      8 findings, all absorbed. See docs/DECISIONS.md 2026-08-22 entries.
- [x] `uv init`, pyproject with ruff + pytest config, `fast` marker, `ds-agents` CLI entry point.
- [x] `state.py` from the draft schema, with tests that a fixture round-trips.
- [x] `tests/fixtures/toy/`: 200-row synthetic tabular dataset, binary target, one planted leakage
      column (noisy copy of target), one legit strong feature. Generator script committed, seed fixed.
- [x] Confirm the hook in .claude/settings.json fires and is fast (<10s). Measured 0.44s. NOTE:
      the hook needs a PATH export to find uv in a non-login shell; fix is staged, not applied.
- [x] First DECISIONS.md entries.

## Phase 1: Linear skeleton (2 to 3 sessions)
Scope note: the router turned out to need `NodeName` and a real seat in the graph, so it is listed
here rather than appearing for the first time in Phase 3. `verified_holdout_score` requires the
harness to withhold rows before the graph starts and re-apply `feature_code_artifact` to score
them; that is Phase 4 scope that PLAN.md did not previously name.
- [x] `tools/local.py` shim with the four MCP signatures, behind a `Tools` Protocol. Subprocess
      execution, stripped environment, artifact store, both asserted in tests.
- [x] `tools/llm.py`: `StructuredModel` Protocol plus `StubModel`, so the graph runs with no API
      key. The LangChain-vs-Anthropic fork is now decided (direct SDK) and the adapter is written.
- [x] intake node (spec from the dataset artifact's schema) and profiler node (deterministic stats
      via `run_python`, model nominates leakage candidates, pins `split_artifact`), unit tests each
- [x] graph.py wiring for intake -> profiler, `ds-agents run --dataset toy` prints a node trace
- [x] modeler, reporter nodes. Modeler fits a fixed candidate set on the pinned folds and asks the
      model only which to promote; reporter is deterministic markdown and calls no model.
- [x] feature_eng node. Model proposes a drop plan, a templated snippet does the transform. The
      artifact is a *fitted transform as code*, so Phase 4 re-applies the same code path.
- [x] router as a real node on a conditional edge, with the cycle back to feature_eng/modeler
      wired and tested even though nothing takes it until Phase 3.
- [x] LangSmith tracing wired (`@traceable`, inert without a key -- untested end to end, no key
      yet). NodeEvent cost accounting live: a real toy run reports ~$0.009.
- [x] real model client (`AnthropicModel`, direct SDK), per-model pricing, and the guard --
      `PipelineState.publishable()`, which Phase 4's harness must call before writing a row.

Scope note: the guard landed as a method on `PipelineState` rather than inside `harness.py`, so the
CLI and the not-yet-written harness share one implementation. The CLI prints it on every run.
First live Haiku runs done: the profiler flagged the planted leak in 9 of 9 runs, `feature_eng`
acted on it in 8 of 9.

## Phase 2: MCP server (2 to 3 sessions)
Scope note: the Docker-versus-long-lived-container question resolved to neither. The sandbox is a
warm worker process that forks per snippet — same isolation per snippet, one twentieth the cost,
and it does not make `pytest -m fast` depend on a running Docker daemon. Docker is deferred behind
the `SandboxPool` seam until model-authored feature code needs a memory cap and a network block.
See DECISIONS.md 2026-08-27.
- [x] `mcp_server/` sandbox and store: `_worker.py` (fork-per-snippet, stripped env, pruned
      `sys.path`, process-group timeout), `sandbox.py` (worker lifetime, request protocol,
      outer deadline), `store.py` (artifact store, `sandbox_path` in metadata, metric log to
      JSONL). 25 tests. `tools/local.py` rewritten as a thin binding of the same two objects.
- [x] `mcp_server/server.py`: the four tools over the MCP protocol, wrapping the objects above.
      It constructs a `LocalTools` and exposes its four methods, so the protocol layer decides
      nothing; one server process per run, because keying stores by `run_id` would need a fifth
      tool and the tool surface is what the Phase 5 ablation holds constant.
- [x] swap `tools/local.py` for MCP client wrappers; toy run still green. `tools/mcp_client.py`
      satisfies the same Protocol, `ds-agents run` defaults to `--tools mcp`, `--tools local`
      stays for debugging. Three live Haiku runs through MCP: leak dropped 3 of 3, roc_auc 0.843,
      15.4s and $0.0098, all matching the in-process numbers.
- [x] verify Claude Code can connect to the same server (this is the generalist-agent baseline
      later). Done from inside Claude Code itself: the four tools resolve as
      `mcp__ds-agents-tools__*` and all four were exercised against the toy dataset by an agent
      that shares no code with `tools/mcp_client.py`. `run_python` saw the dataset at
      `$DS_DATASET`, the repo off `sys.path` and no API key in the environment; `write_artifact`
      returned an id and a `sandbox_path`; `read_artifact` with `max_bytes=40` returned
      `truncated: true` and the same artifact opened whole from inside a snippet at its
      `sandbox_path`; `log_metric` accepted a number. Residual, and it is a CLI flag rather than a
      capability: `claude mcp list` still prints "pending approval" for the project-scoped entry,
      so a fresh terminal session prompts once. See DECISIONS.md 2026-08-27.

## Phase 3: Adversarial reviewer (4 to 6 sessions, plan mode)
Scope note: the `ReviewPass` split-ownership problem was settled before the node was written, and
it changed who writes what. The router now mints the whole `ReviewPass` and the reviewer hands its
dispositions across on a new narrow field, `reviewer_dispositions`. `routed_to` got a single
authority at the same time: the router computes the destination and `route_target` reads it back.
See DECISIONS.md 2026-08-27.

Scope note: fixtures needed a registry before a second one could be run at all -- `cmd_run` rejected
every `--dataset` but `toy` and hardcoded the CSV path. `src/ds_agents/fixtures.py` types the
manifest and resolves a fixture by name; nothing in `nodes/` imports it, because a node that could
read a manifest could read the answer key.
- [x] reviewer node with structured Objection output and conditional routing
- [x] loop cap, `exhausted` verdict. Both reached live, not just in tests: 2 of 7 live Haiku runs
      ended `exhausted` at `loop_cap=3` on a `metric_mismatch` objection the modeler never
      satisfied.
- [x] feature_eng and modeler consume objections. Wired in Phase 1; this session is the first
      evidence it works end to end, in `tests/test_review_loop.py` (block once, drop the objected
      column, pass) and in the live runs that took the cycle edge back to `modeler` twice.
- [x] reviewer catches the toy leakage; test that it does. **Answered, and the answer is no.** It
      took three fixtures to get a leaky matrix in front of the reviewer at all. In the 3 runs where
      a planted trap did survive upstream and ranked first or second by permutation importance, at a
      claimed roc_auc of 0.954 to 0.986 against a legitimate ceiling near 0.82, the reviewer returned
      `pass` and raised no leakage or contamination objection in 3 of 3. It never mentioned the
      columns. Needs a larger n and a Sonnet arm before publication, but the box is no longer
      untestable. See DECISIONS.md 2026-08-27.
- [~] 2 to 3 more leakage-trap variants. Two built: `claims_timing` (timestamp after label, two trap
      columns) and `reissued_ids` (ID-encoded target, with a genuine unique id alongside as a
      control). Both ship a seeded generator, a committed CSV and a manifest, and both are proved by
      test to reach `final_features` and `top_importances` under `StubModel`. Duplicate-rows-across-
      split is deferred, not skipped: the reviewer is never shown the split or any rows, so it is
      structurally uncatchable today, and `results_row()` scores leakage as a set comparison over
      columns and cannot score a trap with no guilty column.
- [ ] Name transparency as an explicit ablation axis. The traps were tuned to be statistically
      invisible and got caught anyway; renaming them defeated the profiler far more effectively than
      any amount of association tuning did (3 of 3 nominated with descriptive names, 1 of 3 with
      opaque ones, n=3 a side). Difficulty is currently an unrecorded property of what the fixture
      author called a column, which is an uncontrolled variable under every leakage number we plan
      to publish. Needs a manifest field and a harness condition. Now the priority.

## Phase 4: Eval harness (3 to 5 sessions, plan mode)
- [ ] evals/datasets/manifest.yaml with 10 to 15 OpenML / Kaggle playground datasets and
      published baselines, sources cited
- [ ] harness.py, results JSONL, eval-diff command
- [ ] `ci` subset, GitHub Actions job with thresholds
- [ ] LOG.md running

## Phase 5: Ablations and writeup (3 to 4 sessions)
- [ ] reviewer on/off, Haiku/Sonnet reviewer, single agent vs team, loop cap 1/3
- [ ] README as a short paper: thesis, setup, results tables, failure analysis, design section
      lifted from DECISIONS.md
- [ ] resume bullet with real numbers

## Out of scope unless the core is done and credits remain
- non-tabular data, hyperparameter search, a web UI, TypeScript anything, Kubernetes anything
