"""The tool surface, as a Protocol.

These four signatures are the contract in docs/ARCHITECTURE.md under "MCP server (ours)". Phase 1
satisfies them in-process (`local.py`); Phase 2 satisfies them over MCP. Nodes type-hint against
`Tools` and never import a concrete implementation, so the two arms of the single-agent-vs-team
ablation are guaranteed the same surface.

Everything crossing this boundary is JSON-shaped on purpose. A shim that handed a node a DataFrame
would work fine in Phase 1 and be impossible to reproduce over MCP in Phase 2.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from ds_agents.state import ArtifactId, Contract


class ToolError(RuntimeError):
    """A tool refused or failed. Nodes catch this and write a `PipelineError`; they never crash
    the graph, because a dataset that hard-fails still has to produce an eval row."""


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
