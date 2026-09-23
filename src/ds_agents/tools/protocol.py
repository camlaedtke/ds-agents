"""The tool surface, as a Protocol.

These four signatures are the contract in docs/ARCHITECTURE.md under "MCP server (ours)",
satisfied in-process (`local.py`) and over MCP (`tools/mcp_client.py`). Nodes type-hint against
`Tools` and never import a concrete implementation, so the two arms of the single-agent-vs-team
ablation are guaranteed the same surface.

Everything crossing this boundary is JSON-shaped on purpose. A shim that handed a node a DataFrame
would work fine in-process and be impossible to reproduce over MCP.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from ds_agents.state import ArtifactId, Contract


class ToolError(RuntimeError):
    """A tool refused or failed. Nodes catch this and write a `PipelineError`; they never crash
    the graph, because a dataset that hard-fails still has to produce an eval row."""


# MCP has exactly one failure channel: an error result carrying a string. That would collapse the
# one distinction the in-process binding is careful about -- a tool that refused (`ToolError`, the
# snippet or the request was bad) against a sandbox that could not run at all (`SandboxError`, the
# machine is broken). Those belong in different columns: the first is a finding about the agent,
# the second is a finding about us. The server prefixes the second, the client strips the prefix
# and re-raises the right type, and `tests/tools/test_mcp_client.py` asserts the round trip.
SANDBOX_ERROR_PREFIX = "sandbox-failure: "

# What `read_artifact` returns when the caller names no limit. Without it a node could pull an
# arbitrarily large artifact through a JSON response and render it into a prompt; with it the
# payload comes back `truncated=True`, which `feature_eng` and `modeler` already refuse to use.
# The cap lives here, not in the transport, so the in-process and the MCP bindings truncate at the
# same byte and a node cannot behave differently depending on how it was wired. The escape hatch
# for a genuinely large artifact is `ArtifactMeta.extra["sandbox_path"]`: open it in a snippet
# rather than asking for a bigger cap.
DEFAULT_READ_BYTES = 1 << 20


class RunResult(Contract):
    """What `run_python` gives back. Deliberately not a Python object: stdout is a string, and a
    node that wants structure has to make its snippet print JSON."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_s: float = Field(default=0.0, ge=0.0)
    artifacts_written: list[ArtifactId] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class ArtifactMeta(Contract):
    id: ArtifactId
    name: str
    kind: str = Field(default="text", description="text | table | json | model | report")
    n_bytes: int = Field(default=0, ge=0)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific detail recorded at write time. For a table: columns, n_rows, "
        "dtypes. This is how intake sees the schema without being handed the whole file.",
    )


class ArtifactPayload(Contract):
    meta: ArtifactMeta
    content: str = ""
    truncated: bool = Field(
        default=False,
        description="True when `max_bytes` cut the content. A node that renders a truncated "
        "artifact into a prompt as if it were whole would report on rows it never saw.",
    )


@runtime_checkable
class Tools(Protocol):
    """The whole tool surface. If a node needs something not here, that is an architecture
    change, not a helper function."""

    def run_python(self, code: str, timeout_s: int = 60) -> RunResult:
        """Execute `code` in the sandbox. Never `exec` in-process: see CLAUDE.md."""
        ...

    def read_artifact(
        self, artifact_id: ArtifactId, max_bytes: int | None = None
    ) -> ArtifactPayload: ...

    def write_artifact(
        self,
        name: str,
        content: str,
        kind: str = "text",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactId: ...

    def log_metric(self, run_id: str, name: str, value: float) -> None: ...
