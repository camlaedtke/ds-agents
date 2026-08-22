---
name: add-node
description: Procedure for adding or substantially changing a LangGraph node in this pipeline (intake, profiler, feature_eng, modeler, reviewer, reporter, or a new one). Use this whenever the task involves creating a node, changing what a node reads from or writes to PipelineState, adding a routing edge, or adding a new agent role, even if the user just says "add a step" or "make the pipeline do X."
---

# Add or change a node

Every node is a pure function of `PipelineState` plus tool calls. Follow these steps in order.
Do not skip the test step to "come back to it later."

## 1. Decide the contract before writing code

Write down, in the chat, before touching files:

- **Reads:** which `PipelineState` fields this node consumes
- **Writes:** which fields it produces or updates (additive only; nodes never delete state)
- **Tools:** which MCP tools it may call (`run_python`, `read_artifact`, `write_artifact`, `log_metric`)
- **Routing:** fixed next node, or conditional? If conditional, what field drives it?
- **Failure:** what happens on tool error or malformed LLM output? (Default: write an error to `state.errors`, route to reporter.)

If the contract needs a new state field, that is a state schema change. Do it first, in `state.py`,
with a default value so old fixtures still load.

## 2. Files to create or edit

1. `src/ds_agents/state.py` if the schema changes. Add a docstring on the new field.
2. `src/ds_agents/nodes/<name>.py` with a single entry point `def <name>(state: PipelineState) -> PipelineState`.
   - Prompt templates live in the same file as module-level constants. Structured output via a Pydantic model; parse with the LLM's structured output mode, never regex.
   - The node must be runnable with a mocked LLM. Take the model client as an injectable dependency.
3. `tests/nodes/test_<name>.py`
   - One test with a mocked LLM response on `tests/fixtures/toy/` state, asserting the writes in the contract.
   - One test for the failure path in the contract.
   - Mark both `@pytest.mark.fast`.
4. `src/ds_agents/graph.py`: register the node and its edges. Nothing else changes here.
5. `docs/ARCHITECTURE.md`: update the node table (reads, writes, tools, routing).

## 3. Verify

```
uv run pytest tests/nodes/test_<name>.py
uv run ds-agents run --dataset toy
```

The toy run must finish and the node must appear in the printed trace. If the node is the reviewer
or anything that loops, confirm the loop cap in `state.review_iterations` is respected on the toy
dataset (which has planted leakage, so the reviewer should object at least once).

## 4. Record

If you made a design choice that a future session might question (loop caps, what counts as
"blocking," which model the node uses), add a paragraph to `docs/DECISIONS.md`.

## Things that go wrong

- Node reads a file path from state and opens it directly. Wrong. Use `read_artifact`.
- Node returns a dict instead of `PipelineState`. LangGraph will accept it and silently drop typed validation.
- Prompt asks the model for JSON "in the following format" as free text. Use structured output.
- Test asserts on the LLM's prose instead of the structured fields. Prose is not a contract.
