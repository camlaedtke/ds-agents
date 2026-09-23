"""Tool surface the nodes call.

`local.py` is an in-process implementation with exactly the signatures the MCP server also
exposes (`tools/mcp_client.py`). Nodes import the Protocol, never a concrete implementation, so
the transport is a one-line change in `graph.py` and the toy run is the regression test for it.
"""

from ds_agents.tools.local import LocalTools, dataset_artifact_id
from ds_agents.tools.protocol import (
    ArtifactMeta,
    ArtifactPayload,
    RunResult,
    ToolError,
    Tools,
)

__all__ = [
    "ArtifactMeta",
    "ArtifactPayload",
    "LocalTools",
    "RunResult",
    "ToolError",
    "Tools",
    "dataset_artifact_id",
]
