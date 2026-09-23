# ds-agents

Multi-agent data science pipeline (LangGraph) with an adversarial reviewer and an eval harness.
Thesis: how much of a data scientist's tabular workflow can be delegated to a team of agents, and
how do we measure exactly where they fail? Findings are the product. The model is not the product.

## Layout

```
src/ds_agents/
  state.py        # PipelineState (Pydantic). Every node reads and writes this. Nothing else.
  graph.py        # LangGraph wiring only. No logic.
  nodes/          # one file per node: intake, profiler, feature_eng, modeler, reviewer, reporter
  tools/          # MCP client wrappers the nodes call (sandbox, artifacts, metrics)
  harness.py      # runs the pipeline over the benchmark set, writes evals/results/
  evaldiff.py     # compares two results files; refuses to call an underpowered difference an effect
mcp_server/       # our MCP server: run_python sandbox, artifact store, metric log
evals/
  datasets/       # manifest.yaml + loaders. Never edit the datasets or baselines by hand.
  results/        # one JSONL per run, committed
tests/
  fixtures/toy/   # 200-row toy dataset with planted leakage. Full pipeline runs on it in <60s.
docs/             # ARCHITECTURE.md, PLAN.md, DECISIONS.md, decisions-archive.md, NEXT.md,
                  # LEARNING.md, RESULTS.md, explainers/
```

## Commands

- `uv run pytest -m fast` fast unit tests (runs on every edit via hook)
- `uv run pytest` full suite including the toy end-to-end run
- `uv run ds-agents run --dataset toy` single pipeline run, prints node trace
- `uv run ds-agents eval --subset ci` benchmark subset, see /run-eval skill
- `uv run mcp-server` start the sandbox/artifact MCP server locally

## Rules

- All node I/O goes through `PipelineState`. No node reads files or env directly; it calls a tool.
- Agents execute code only through the `run_python` MCP tool. Never `exec` in-process.
- The toy pipeline must pass at the end of every session. If it is broken, fixing it is the next task.
- Do not **hand-edit** `evals/datasets/` or published baseline numbers. They are the ground
  truth, and `.claude/settings.json` denies Edit and Write there, which is what makes
  "nothing in the manifest was typed by a person" a property rather than a promise. Change
  it only by running `ds-agents datasets refresh`, then `ds-agents datasets verify --online`.
- New nodes follow /add-node. Benchmark runs follow /run-eval. End every session with /session-wrap.
- Log architecture choices in `docs/DECISIONS.md`, one paragraph each, when they are made.
- Cost discipline: default agent model is Haiku. Sonnet only for the reviewer, and only when testing that ablation.
- If Cameron says he is confused or asks what something is, use /explain-visual rather than explaining in prose.

## Working mode

The owner of this repo is deliberately delegating most of the implementation and learning the
parts that are load-bearing. Optimize for forward motion; make the learning optional and explicit.

- **Default to proceeding.** When a choice is routine, pick the sensible option, state it in one
  line, and keep going. Ask only when two readings would produce materially different work, or
  when the choice would quietly change what the eval measures.
- **Delegate to subagents by default**, sized to the task. Keep the main thread for judgment,
  design, and explanation.
  - Haiku: running tests and evals (`test-runner`), locating code (`Explore`), mechanical edits,
    log and output triage.
  - Sonnet: diff review (`reviewer`), implementing an already-specced node, writing docs.
  - Opus: architecture and contract design, adversarial critique of the eval design, debugging a
    failure whose cause is not yet obvious, anything in PLAN.md Phase 3 or 4.
  Say which agent and model you used when it is not obvious.
- **Flag concepts, do not lecture.** When the work depends on something the owner has not learned,
  append an entry to `docs/LEARNING.md` and continue. Interrupt only for a `load-bearing` concept
  that a decision is about to rest on, once, in one sentence. `/learn` is where dives happen.
- **Never hide a real problem to keep things smooth.** Low friction means fewer questions, not
  fewer facts. Broken toy pipeline, a failing test, a number that moved: say it plainly.

## Style

- Python 3.12, type hints everywhere, Pydantic v2 for all structured data. ruff for lint and format.
- Tests before wiring. A node without a unit test on a fixed fixture is not done.
- Plain language in docs and commit messages. No marketing tone.
