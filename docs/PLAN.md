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
- [ ] mcp_server/ with run_python sandbox (Docker, subprocess, timeout, no network), artifact
      store, log_metric
- [ ] swap `tools/local.py` for MCP client wrappers; toy run still green
- [ ] verify Claude Code can connect to the same server (this is the generalist-agent baseline later)

## Phase 3: Adversarial reviewer (4 to 6 sessions, plan mode)
- [ ] reviewer node with structured Objection output and conditional routing
- [ ] loop cap, `exhausted` verdict
- [ ] feature_eng and modeler consume objections
- [ ] reviewer catches the toy leakage; test that it does
- [ ] 2 to 3 more leakage-trap variants (timestamp after label, ID-encoded target, duplicate rows across split)

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
