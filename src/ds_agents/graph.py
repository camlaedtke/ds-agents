"""LangGraph wiring only. No logic.

The whole skeleton, cycle included:

    intake -> profiler -> feature_eng -> modeler -> reviewer -> router -> reporter -> END
                               ^                                    |
                               +---------------- feature_eng -------+
                                          `-- modeler --'

Every straight-line edge above is really `halt_or(destination)`: a node that returned an error with
`recoverable=False` sends the run to `reporter` and no further node runs. The reporter is written
for that path, so a dataset that cannot be run still produces a row -- one carrying `halted_at`.

The router sits on a conditional edge; the cycle edges back to `feature_eng` and `modeler` fire
whenever `route_target` reads a `block` verdict off the router's latest `ReviewPass`.

`tools` and `model` are bound with `functools.partial` so LangGraph sees the one-argument function
it expects while nodes stay pure functions of state plus injected dependencies. `reviewer_model`
is bound to the `reviewer` node ONLY -- everything else still gets `model` -- which is what makes
the Haiku/Sonnet reviewer ablation a config change rather than an edit across nodes. A disabled
reviewer is not a graph-level decision: `reviewer` no-ops internally (see its module docstring),
so this file stays logic-free either way.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from ds_agents.nodes.feature_eng import feature_eng
from ds_agents.nodes.intake import intake
from ds_agents.nodes.modeler import modeler
from ds_agents.nodes.profiler import profiler
from ds_agents.nodes.reporter import reporter
from ds_agents.nodes.reviewer import reviewer
from ds_agents.nodes.router import halt_or, route_target, router
from ds_agents.state import PipelineState, utc_now
from ds_agents.tools.llm import StructuredModel
from ds_agents.tools.protocol import Tools

NODES = (
    ("intake", intake),
    ("profiler", profiler),
    ("feature_eng", feature_eng),
    ("modeler", modeler),
    ("reviewer", reviewer),
    ("router", router),
    ("reporter", reporter),
)

# Straight-line edges -- conditional in LangGraph's sense but not in this pipeline's, because the
# only thing that diverts one is an unrecoverable error. See `halt_or` in `nodes/router.py`.
EDGES = (
    ("intake", "profiler"),
    ("profiler", "feature_eng"),
    ("feature_eng", "modeler"),
    ("modeler", "reviewer"),
    ("reviewer", "router"),
)

# Where `route_target` may send a run. Every value here must be a node above, or LangGraph raises
# at compile time rather than at the first block -- which is the failure we want, since a bad
# destination would otherwise surface only on a dataset the reviewer actually objects to.
ROUTES = {"feature_eng": "feature_eng", "modeler": "modeler", "reporter": "reporter"}


def build_graph(
    *, tools: Tools, model: StructuredModel, reviewer_model: StructuredModel | None = None
):
    builder = StateGraph(PipelineState)
    for name, node in NODES:
        node_model = reviewer_model if name == "reviewer" and reviewer_model is not None else model
        builder.add_node(name, partial(node, tools=tools, model=node_model))
    builder.add_edge(START, "intake")
    for source, destination in EDGES:
        builder.add_conditional_edges(
            source, halt_or(destination), {destination: destination, "reporter": "reporter"}
        )
    builder.add_conditional_edges("router", route_target, ROUTES)
    builder.add_edge("reporter", END)
    return builder.compile()


def run_pipeline(
    state: PipelineState,
    *,
    tools: Tools,
    model: StructuredModel,
    reviewer_model: StructuredModel | None = None,
) -> PipelineState:
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
    graph = build_graph(tools=tools, model=model, reviewer_model=reviewer_model)
    final = graph.invoke(state, config={"recursion_limit": limit})
    result = final if isinstance(final, PipelineState) else PipelineState.from_dump(dict(final))
    result.ended_at = utc_now()
    return result
