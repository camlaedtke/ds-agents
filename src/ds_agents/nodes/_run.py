"""Per-node bookkeeping shared by every node.

Exists so that the `NodeEvent` on a node's error path is identical to the one on its happy path.
A node that only appended its event on success would make the cost table read low by exactly the
runs that failed, which is the direction that flatters the system.
"""

from typing import Any

from ds_agents.state import NodeEvent, NodeName, PipelineError, utc_now
from ds_agents.tools.llm import Completion


class NodeRun:
    """Accumulates usage across a node's model calls and mints its `NodeEvent`."""

    def __init__(self, node: NodeName) -> None:
        self.node = node
        self.started = utc_now()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.model: str | None = None

    def record[T](self, completion: Completion[T]) -> T:
        """Book the usage and hand back the parsed value."""
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.cost_usd += completion.cost_usd
        self.model = completion.model
        return completion.value

    def event(self) -> NodeEvent:
        return NodeEvent(
            node=self.node,
            started=self.started,
            ended=utc_now(),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            model=self.model,
        )

    def failure(self, message: str, recoverable: bool = True) -> dict[str, Any]:
        """The standard error return: an error, an event, and no half-written state fields."""
        return {
            "errors": [PipelineError(node=self.node, message=message, recoverable=recoverable)],
            "node_trace": [self.event()],
        }
