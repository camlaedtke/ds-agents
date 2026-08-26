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
2. `src/ds_agents/nodes/<name>.py` with a single entry point:
   `def <name>(state: PipelineState, *, tools: Tools, model: StructuredModel) -> dict[str, Any]`.
   - It returns a **narrow dict of the fields it changed**. Never `return state`. Nothing in the
     type system stops you, and with `operator.add` on `objections`, `review_passes`, `errors`,
     and `node_trace`, returning the state object concatenates the accumulated history onto itself
     and doubles every trace event and every objection.
   - `tools` and `model` are injected, never imported at module level. `graph.py` binds them with
     `functools.partial`. This is what makes the node testable against canned tool results and the
     reviewer-model ablation a config change.
   - Use `NodeRun` from `nodes/_run.py` for the `NodeEvent`, and `run.failure(...)` for error
     returns, so the trace event is appended on the failure path too. A node that only records its
     event on success makes the cost table read low by exactly the runs that failed.
   - Prompt templates live in the same file as module-level constants. Structured output via a Pydantic model; parse with the LLM's structured output mode, never regex.
   - Put the facts the model needs in the user message as a JSON block, not as prose. The tests
     assert on what reached the prompt, and a model cannot name a column it was never shown.
3. `tests/nodes/test_<name>.py`
   - Use `FakeTools` and `ScriptedModel` from `tests/conftest.py`. A node test that shells out to
     a real subprocess is testing pandas.
   - One test asserting the writes in the contract, including `set(update) == {...}` — a node
     writing someone else's field is the failure the contract exists to prevent.
   - One test for the failure path in the contract, asserting the error AND that the trace event
     is still there.
   - One test that the facts the model needs actually reached the prompt.
   - Mark them all `@pytest.mark.fast`.
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
- Node returns the whole `PipelineState` instead of a narrow dict. Doubles every append-only field.
- Node lets a model's output onto the state unvalidated. A leakage candidate naming a column that
  does not exist cannot be scored against ground truth, and counting it fills the false-alarm
  column with the model's typos rather than its judgement. Filter against the known columns.
- Node swallows a tool error into a `None` or an empty list without writing to `state.errors`. A
  statistic that silently reads `null` is indistinguishable from a clean one.
- Prompt asks the model for JSON "in the following format" as free text. Use structured output.
- Test asserts on the LLM's prose instead of the structured fields. Prose is not a contract.
