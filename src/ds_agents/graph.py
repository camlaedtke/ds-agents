"""LangGraph wiring only. No logic.

Phase 1 has the front of the linear skeleton: intake -> profiler -> END. feature_eng, modeler,
reviewer, the router, and reporter arrive in later phases; the shape of this file does not change
when they do, only its edges.

`tools` and `model` are bound here with `functools.partial` so LangGraph sees the one-argument
function it expects while nodes stay pure functions of state plus injected dependencies. That is
also what makes the Phase 2 MCP swap a change to this file alone.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from ds_agents.nodes.intake import intake
from ds_agents.nodes.profiler import profiler
from ds_agents.state import PipelineState, utc_now
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import Tools

NODES = (("intake", intake), ("profiler", profiler))


def build_graph(*, tools: Tools, model: StructuredModel):
    builder = StateGraph(PipelineState)
    for name, node in NODES:
        builder.add_node(name, partial(node, tools=tools, model=model))
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "profiler")
    builder.add_edge("profiler", END)
    return builder.compile()


def run_pipeline(state: PipelineState, *, tools: Tools, model: StructuredModel) -> PipelineState:
    """Run the graph and hand back a `PipelineState`, not a dict.

    `ended_at` is stamped here rather than by the last node, because `wall_seconds` is supposed to
    include graph and tool overhead that node events miss, and because the last node changes every
    time a phase lands.
    """
    final = build_graph(tools=tools, model=model).invoke(state)
    result = final if isinstance(final, PipelineState) else PipelineState.from_dump(dict(final))
    result.ended_at = utc_now()
    return result
