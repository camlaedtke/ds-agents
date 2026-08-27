"""The `Tools` surface over MCP, so nodes call the server instead of calling Python.

`LocalTools` and `MCPTools` are the same implementation reached two ways: the server wraps a
`LocalTools`, and this class talks to the server. Swapping them changes the transport and nothing
else, which is what makes "the agents have the same tool surface as any external MCP client" a
fact about the code rather than an intention.

Two things this file exists to absorb.

**The SDK is async and the graph is not.** Nodes are plain functions and LangGraph calls them
synchronously, so the client runs its event loop on a private thread and each tool call blocks on
a future. The session is opened and closed by one long-lived coroutine on that loop rather than by
the calling thread, because the SDK's connection is an anyio task group and a task group has to be
exited by the task that entered it.

**Failure has to keep its shape.** The wire has one error channel; `SANDBOX_ERROR_PREFIX` is how a
broken sandbox stays distinguishable from a refused tool call on the way back. See
`mcp_server/server.py`.
"""

import asyncio
import concurrent.futures
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.client import Client

from ds_agents.state import ArtifactId
from ds_agents.tools.protocol import (
    SANDBOX_ERROR_PREFIX,
    ArtifactMeta,
    ArtifactPayload,
    RunResult,
    ToolError,
)
from mcp_server.sandbox import SandboxError
from mcp_server.store import dataset_artifact_id

__all__ = ["MCPTools", "stdio_params", "dataset_artifact_id"]

# How long past a snippet's own deadline the client waits for the server to answer. The sandbox
# already enforces the snippet timeout and its own reply deadline; this is the outermost backstop,
# and it exists so a wedged server fails a run instead of hanging a benchmark forever.
_REPLY_SLACK_S = 30.0
_CONNECT_TIMEOUT_S = 60.0
# How long a disconnect is given before the loop is torn down anyway. Long enough for the SDK to
# close its streams and reap the server; short enough that a wedged server cannot hold a benchmark.
_CLOSE_TIMEOUT_S = 10.0


def stdio_params(
    root: Path, dataset_path: Path | None = None, dataset_id: str = ""
) -> StdioServerParameters:
    """Launch the server as a subprocess of this process.

    `sys.executable -m mcp_server.server` rather than the `mcp-server` console script: the script
    is only on PATH inside an activated venv, and a benchmark run started by anything other than
    `uv run` would then fail at connect time for a reason that has nothing to do with the pipeline.
    """
    args = ["-m", "mcp_server.server", "--root", str(root)]
    if dataset_path is not None:
        args += ["--dataset", str(dataset_path), "--dataset-id", dataset_id]
    return StdioServerParameters(command=sys.executable, args=args)


class MCPTools:
    """Satisfies `Tools` by calling an MCP server. One instance per run; `close()` ends it.

    `target` is anything the SDK client connects to: `stdio_params(...)` for the real thing, or an
    `MCPServer` object for a test that wants the protocol without a subprocess.
    """

    def __init__(self, target: Any, connect_timeout_s: float = _CONNECT_TIMEOUT_S) -> None:
        self._target = target
        self._client: Client | None = None
        self._closing: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-tools", daemon=True
        )
        self._thread.start()
        ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._session = asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
        try:
            ready.result(timeout=connect_timeout_s)
        except Exception:
            # `close`, not `_shutdown_loop`: a connect that timed out has left a coroutine parked
            # inside the SDK, and the server subprocess it spawned is only reaped by unwinding
            # that. Stopping the loop under it orphans the process, which a benchmark would do
            # once per failed dataset.
            self.close()
            raise

    # ---- lifetime ---------------------------------------------------------------------------

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        """Own the session for its whole life, on the loop thread.

        Opening here and closing here is not tidiness: `Client.__aenter__` starts an anyio task
        group, and exiting it from a different task -- which is what a `close()` called on the
        calling thread would do -- raises instead of disconnecting.
        """
        self._task = asyncio.current_task()
        try:
            async with Client(self._target, raise_exceptions=False) as client:
                self._closing = asyncio.Event()
                self._client = client
                ready.set_result(None)
                await self._closing.wait()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller through `ready`
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._client = None

    def close(self) -> None:
        """End the session, then the loop. Safe to call twice, and after a failed connect."""
        if not self._loop.is_closed() and not self._session.done():
            if self._closing is not None:
                self._loop.call_soon_threadsafe(self._closing.set)
            elif self._task is not None:
                # Never connected, so there is no session to end politely -- only a coroutine
                # parked inside the SDK's connect. Cancelling it is what unwinds the transport
                # and kills the server subprocess.
                self._loop.call_soon_threadsafe(self._task.cancel)
            try:
                self._session.result(timeout=_CLOSE_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                # Not worth failing a finished run over: the numbers are already in the state
                # object by the time anything calls this. Worth saying out loud, because a server
                # that will not disconnect is how orphaned processes start.
                print(
                    f"the tool server did not disconnect within {_CLOSE_TIMEOUT_S}s",
                    file=sys.stderr,
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"the tool server session ended with {exc!r}", file=sys.stderr)
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)
        self._loop.close()

    def __enter__(self) -> "MCPTools":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- one call ---------------------------------------------------------------------------

    def _call(self, name: str, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise ToolError(f"the tool server is not connected; cannot call {name!r}")
        future = asyncio.run_coroutine_threadsafe(
            client.call_tool(name, arguments, read_timeout_seconds=timeout_s), self._loop
        )
        try:
            result = future.result(timeout=timeout_s + _REPLY_SLACK_S)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise SandboxError(f"the tool server did not answer {name!r} in {timeout_s}s") from exc
        if result.is_error:
            raise _wire_error(result)
        content = result.structured_content
        if content is None:
            raise ToolError(f"{name!r} returned no structured content")
        return content

    # ---- Tools ------------------------------------------------------------------------------

    def run_python(self, code: str, timeout_s: int = 60) -> RunResult:
        return RunResult.model_validate(
            self._call("run_python", {"code": code, "timeout_s": timeout_s}, float(timeout_s))
        )

    def read_artifact(
        self, artifact_id: ArtifactId, max_bytes: int | None = None
    ) -> ArtifactPayload:
        return ArtifactPayload.model_validate(
            self._call("read_artifact", {"artifact_id": artifact_id, "max_bytes": max_bytes}, 30.0)
        )

    def write_artifact(
        self,
        name: str,
        content: str,
        kind: str = "text",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactId:
        payload = self._call(
            "write_artifact",
            {"name": name, "content": content, "kind": kind, "extra": extra},
            30.0,
        )
        return ArtifactMeta.model_validate(payload).id

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        self._call("log_metric", {"run_id": run_id, "name": name, "value": value}, 30.0)


def _wire_error(result: Any) -> Exception:
    """Rebuild the exception the server raised.

    A sandbox failure has to come back as a `SandboxError`, because a node treats it as "this
    machine is broken" and lets it propagate, while a `ToolError` is written into the run as a
    recoverable `PipelineError`. Collapsing the two would file our outage under the agent's
    mistakes.
    """
    message = "\n".join(
        block.text for block in (result.content or []) if getattr(block, "text", None)
    )
    # `in`, not `startswith`: the SDK prepends "Error executing tool <name>: " to an expected
    # tool error, so the marker is inside the message rather than at the front of it.
    _prefix, sep, detail = message.partition(SANDBOX_ERROR_PREFIX)
    if sep:
        return SandboxError(detail)
    return ToolError(message or "the tool server reported an error with no message")
