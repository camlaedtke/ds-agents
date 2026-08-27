"""One file per node. Every node is a pure function of `PipelineState` plus tool calls.

A node signature is `node(state, *, tools, model) -> dict`. It returns a NARROW dict of new
values, never the state object: with `operator.add` on the history fields, returning the whole
state would concatenate the accumulated trace onto itself and double every event and objection.
`graph.py` binds `tools` and `model` with `functools.partial` so LangGraph sees a one-argument
function.

`router` and `reporter` accept `model` and never call it. The uniform signature is what lets
`graph.py` bind every node the same way; a test on each asserts the model went untouched, so a
node quietly acquiring a model call shows up as a failure rather than as a line on a bill.
"""

from ds_agents.nodes.feature_eng import feature_eng
from ds_agents.nodes.intake import intake
from ds_agents.nodes.modeler import modeler
from ds_agents.nodes.profiler import profiler
from ds_agents.nodes.reporter import reporter
from ds_agents.nodes.reviewer import reviewer
from ds_agents.nodes.router import route_target, router

__all__ = [
    "feature_eng",
    "intake",
    "modeler",
    "profiler",
    "reporter",
    "reviewer",
    "route_target",
    "router",
]
