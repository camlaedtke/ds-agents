"""LangGraph wiring only. No logic.

Phase 1 has the whole linear skeleton:

    intake -> profiler -> feature_eng -> modeler -> router -> reporter -> END

The router sits on a conditional edge and the cycle edges back to `feature_eng` and `modeler` are
already wired, even though nothing takes them yet: with no reviewer node, `reviewer_claim` is
`None`, so `route_target` always answers `reporter` and the graph runs straight through. Wiring the
cycle now is what makes the loop contract honest before Phase 3 needs it -- the edges are tested,
and Phase 3 inserts `reviewer` between `modeler` and `router` without rewiring anything here.

`tools` and `model` are bound with `functools.partial` so LangGraph sees the one-argument function
it expects while nodes stay pure functions of state plus injected dependencies. That is also what
makes the Phase 2 MCP swap a change to this file alone.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from ds_agents.nodes.feature_eng import feature_eng
from ds_agents.nodes.intake import intake
from ds_agents.nodes.modeler import modeler
from ds_agents.nodes.profiler import profiler
from ds_agents.nodes.reporter import reporter
from ds_agents.nodes.router import route_target, router
from ds_agents.state import PipelineState, utc_now
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import Tools

NODES = (
    ("intake", intake),
    ("profiler", profiler),
    ("feature_eng", feature_eng),
    ("modeler", modeler),
    ("router", router),
    ("reporter", reporter),
)

# Straight-line edges. `modeler -> router` becomes `modeler -> reviewer -> router` in Phase 3.
EDGES = (
    ("intake", "profiler"),
    ("profiler", "feature_eng"),
    ("feature_eng", "modeler"),
    ("modeler", "router"),
)

# Where `route_target` may send a run. Every value here must be a node above, or LangGraph raises
# at compile time rather than at the first block -- which is the failure we want, since a bad
# destination would otherwise surface only on a dataset the reviewer actually objects to.
ROUTES = {"feature_eng": "feature_eng", "modeler": "modeler", "reporter": "reporter"}


def build_graph(*, tools: Tools, model: StructuredModel):
    builder = StateGraph(PipelineState)
    for name, node in NODES:
        builder.add_node(name, partial(node, tools=tools, model=model))
    builder.add_edge(START, "intake")
    for source, destination in EDGES:
        builder.add_edge(source, destination)
    builder.add_conditional_edges("router", route_target, ROUTES)
    builder.add_edge("reporter", END)
    return builder.compile()


def run_pipeline(state: PipelineState, *, tools: Tools, model: StructuredModel) -> PipelineState:
    """Run the graph and hand back a `PipelineState`, not a dict.

    `ended_at` is stamped here rather than by the last node, because `wall_seconds` is supposed to
    include graph and tool overhead that node events miss, and because the last node changes every
    time a phase lands.
    """
    # LangGraph's default recursion_limit is 25 and knows nothing about `loop_cap`. At
    # loop_cap=6 the cycle overruns it and the run dies with GraphRecursionError and NO results
    # row -- which biases every table upward by deleting exactly the runs that looped most. Six
    # nodes per pass plus headroom, floored so a loop_cap of 0 still gets a sane budget.
    limit = max(25, (len(NODES) + 1) * (state.config.loop_cap + 2))
    final = build_graph(tools=tools, model=model).invoke(state, config={"recursion_limit": limit})
    result = final if isinstance(final, PipelineState) else PipelineState.from_dump(dict(final))
    result.ended_at = utc_now()
    return result
