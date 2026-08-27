"""The four tools over the MCP protocol.

This is a skin, not an implementation. It constructs a `LocalTools` -- the same in-process binding
of `SandboxPool` and `ArtifactStore` that the graph used through Phase 1 -- and exposes its four
methods as MCP tools. Nothing here decides anything about how a snippet runs, what an artifact id
looks like, or when a read is truncated. That is the whole point: "we swapped the shim for the
server" has to mean the transport changed, not that a second implementation appeared and matched
the first by inspection. If a rule needs changing it changes in `store.py` or `sandbox.py` and
both bindings move together.

**One server process per run.** The store is run state -- it owns the artifact index and the run's
directories -- while the sandbox worker is deliberately process-wide and holds none. The
alternative was one long-lived server keyed by `run_id`, which needs a fifth tool (or a run_id
argument on the other four) to open a run, and that changes the agent-facing tool surface. Since
the surface is exactly what the single-agent-vs-team ablation holds constant, it does not get an
extra tool for our convenience. The cost of a process per run is one 0.87s worker boot per
dataset, which a Phase 4 benchmark pays once per dataset against minutes of fitting.

**Two failure channels squeezed into one.** MCP reports failure as an error result carrying a
string, so a snippet that failed and a sandbox that could not run would arrive identically. They
are different findings -- one is about the agent, one is about us -- so `SandboxError` crosses
with `SANDBOX_ERROR_PREFIX` and `tools/mcp_client.py` raises it back as itself.

Run it with `uv run mcp-server --root <dir> [--dataset <csv> --dataset-id <name>]`, which is also
the command an external MCP client (Claude Code, for the Phase 5 generalist baseline) is pointed
at. Requests are line-delimited JSON-RPC over stdin/stdout, so nothing this module writes may go
to stdout: `print` here corrupts the protocol channel the same way a snippet writing to fd 1
would corrupt the sandbox's.
"""

import argparse
import contextlib
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as WireToolError

from ds_agents.state import ArtifactId
from ds_agents.tools.local import LocalTools
from ds_agents.tools.protocol import (
    SANDBOX_ERROR_PREFIX,
    ArtifactMeta,
    ArtifactPayload,
    RunResult,
    ToolError,
)
from mcp_server.sandbox import SandboxError

SERVER_NAME = "ds-agents-tools"

INSTRUCTIONS = """\
The tool surface for one data science run. `run_python` executes a snippet in a sandbox that has
pandas, numpy and scikit-learn imported and nothing else of ours: it cannot see this repository,
this run's PipelineState, or any API key. The dataset is mounted read-only at $DS_DATASET and
$DS_ARTIFACTS is a writable directory -- a file left there is frozen under a new artifact id and
returned in `artifacts_written`. Artifacts are immutable once issued. `read_artifact` is capped
and reports `truncated`; a large artifact is meant to be opened from inside a snippet at
`meta.extra["sandbox_path"]`, not pulled through the tool.
"""


@contextlib.contextmanager
def _translated() -> Iterator[None]:
    """Map our two failure types onto MCP's one.

    Both become an expected error result rather than a crash, so the client gets the message
    instead of a generic "error executing tool"; the prefix is what keeps them distinguishable.
    Anything else really is a crash and is left to the SDK, which withholds its text -- correct,
    because an unexpected exception here is a bug in this file, not a finding about a run.
    """
    try:
        yield
    except SandboxError as exc:
        raise WireToolError(f"{SANDBOX_ERROR_PREFIX}{exc}") from exc
    except ToolError as exc:
        raise WireToolError(str(exc)) from exc


def build_server(tools: LocalTools) -> MCPServer:
    """Wrap one run's `LocalTools` in the protocol. Separate from `main` so a test can connect to
    the server object in-process and exercise the real serialisation without spawning anything."""
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(
        description="Execute a Python snippet in the sandbox and return its stdout, stderr, exit "
        "code and any artifacts it wrote. Never raises on a snippet that fails: check `exit_code` "
        "and `timed_out`."
    )
    def run_python(code: str, timeout_s: int = 60) -> RunResult:
        with _translated():
            return tools.run_python(code, timeout_s=timeout_s)

    @server.tool(
        description="Read an artifact by id. `max_bytes` caps the content; the response says "
        "whether it was truncated. Do not reason about a truncated artifact as if it were whole."
    )
    def read_artifact(artifact_id: ArtifactId, max_bytes: int | None = None) -> ArtifactPayload:
        with _translated():
            return tools.read_artifact(artifact_id, max_bytes=max_bytes)

    @server.tool(
        description="Store text as a new immutable artifact and return its metadata, including "
        "the id and the path a snippet can open it at."
    )
    def write_artifact(
        name: str,
        content: str,
        kind: str = "text",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactMeta:
        # Returns the metadata rather than the bare id, which the `Tools` Protocol asks for: the
        # id is in it, and an external client gets `n_bytes` and `sandbox_path` without a second
        # call. `tools/mcp_client.py` takes `.id` and satisfies the Protocol unchanged.
        with _translated():
            return tools.store.meta(tools.write_artifact(name, content, kind=kind, extra=extra))

    @server.tool(description="Record one number against this run id in the run's metric log.")
    def log_metric(run_id: str, name: str, value: float) -> None:
        with _translated():
            tools.log_metric(run_id, name, value)

    return server


def build_tools(
    root: Path | None = None, dataset: Path | None = None, dataset_id: str = ""
) -> LocalTools:
    """One run's tools. `root` defaults to a fresh tempdir so a client that only wants to poke at
    the sandbox does not have to invent a directory first."""
    return LocalTools(
        Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="ds-agents-run-")),
        dataset_path=Path(dataset) if dataset else None,
        dataset_id=dataset_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="mcp-server", description="ds-agents tool server (MCP)")
    parser.add_argument("--root", default=None, help="where this run's artifacts land")
    parser.add_argument("--dataset", default=None, help="csv to register as the run's dataset")
    parser.add_argument("--dataset-id", default="", help="id the dataset artifact is keyed by")
    args = parser.parse_args()

    tools = build_tools(args.root, args.dataset, args.dataset_id)
    # stderr, not stdout: stdout is the protocol channel.
    print(f"{SERVER_NAME}: run root {tools.root}", file=sys.stderr)
    try:
        build_server(tools).run(transport="stdio")
    finally:
        tools.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
