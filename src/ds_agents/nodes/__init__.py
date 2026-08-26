"""One file per node. Every node is a pure function of `PipelineState` plus tool calls.

A node signature is `node(state, *, tools, model) -> dict`. It returns a NARROW dict of new
values, never the state object: with `operator.add` on the history fields, returning the whole
state would concatenate the accumulated trace onto itself and double every event and objection.
`graph.py` binds `tools` and `model` with `functools.partial` so LangGraph sees a one-argument
function.
"""

from ds_agents.nodes.intake import intake
from ds_agents.nodes.profiler import profiler

__all__ = ["intake", "profiler"]
