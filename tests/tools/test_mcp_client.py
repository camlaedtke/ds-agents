"""The MCP transport, asserted as a transport.

The rules these tools obey -- immutable artifacts, a pruned `sys.path`, a killed process group --
are tested against the sandbox and the store in `tests/mcp_server/`, and the server wraps those
same objects, so repeating them here would test the same code twice. What is only true of the
protocol layer, and is therefore what this file is for: the four calls cross a JSON boundary and
come back as the same Pydantic contracts, and a failure keeps its type on the way back.

Most of these connect to the server object in-process. That is not a shortcut around the protocol
-- the request is serialised, validated against the tool's schema, dispatched and answered exactly
as it would be over a pipe -- it just skips spawning a Python. `test_the_server_runs_as_a_real
_subprocess` is the one that proves the spawn works, and it is deliberately not `fast`.
"""

import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest
from mcp import StdioServerParameters

from ds_agents.cli import _select_tools
from ds_agents.tools.local import LocalTools
from ds_agents.tools.mcp_client import MCPTools, stdio_params
from ds_agents.tools.protocol import DEFAULT_READ_BYTES, ToolError, Tools
from mcp_server.sandbox import SandboxError
from mcp_server.server import build_server
from mcp_server.store import dataset_artifact_id

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "toy" / "toy.csv"


@pytest.fixture
def local(tmp_path: Path) -> LocalTools:
    tools = LocalTools(tmp_path, dataset_path=TOY, dataset_id="toy")
    yield tools
    tools.close()


@pytest.fixture
def client(local: LocalTools) -> MCPTools:
    """A client wired to a server wrapping `local`, so a test can check both sides of one call."""
    tools = MCPTools(build_server(local))
    yield tools
    tools.close()


@pytest.mark.fast
def test_the_client_satisfies_the_tools_protocol(client: MCPTools):
    """If this fails, `graph.py` cannot bind it and the Phase 2 swap is not a swap."""
    assert isinstance(client, Tools)


@pytest.mark.fast
def test_artifacts_round_trip_through_the_protocol(client: MCPTools):
    artifact_id = client.write_artifact("notes.txt", "the leak is account_status_code", kind="text")
    payload = client.read_artifact(artifact_id)

    assert payload.content == "the leak is account_status_code"
    assert payload.meta.id == artifact_id
    assert payload.meta.kind == "text"
    assert not payload.truncated
    # `sandbox_path` is what makes a large artifact openable from a snippet instead of pulled
    # through the wire, so it has to survive serialisation.
    assert Path(payload.meta.extra["sandbox_path"]).exists()


@pytest.mark.fast
def test_the_dataset_artifact_carries_its_schema_across(client: MCPTools):
    """Intake names the target off `extra`, and it is the one node with no `run_python`."""
    payload = client.read_artifact(dataset_artifact_id("toy"), max_bytes=200)

    assert payload.truncated
    assert len(payload.content) == 200
    assert "churned" in payload.meta.extra["columns"]
    assert payload.meta.extra["n_rows"] == 200


@pytest.mark.fast
def test_a_read_with_no_limit_is_still_capped(local: LocalTools, client: MCPTools):
    """An uncapped read would let a node pull an arbitrary number of bytes into a prompt. Both
    bindings truncate at the same byte, so a node cannot behave differently by transport."""
    artifact_id = client.write_artifact("big.txt", "x" * (DEFAULT_READ_BYTES + 10))

    over_the_wire = client.read_artifact(artifact_id)
    in_process = local.read_artifact(artifact_id)

    assert over_the_wire.truncated and in_process.truncated
    assert len(over_the_wire.content) == len(in_process.content) == DEFAULT_READ_BYTES
    assert over_the_wire.meta.n_bytes == DEFAULT_READ_BYTES + 10


@pytest.mark.fast
def test_a_missing_artifact_is_a_tool_error_not_a_crash(client: MCPTools):
    """A node catches `ToolError` and writes a recoverable `PipelineError`. If this arrived as
    something else the run would die instead of producing an eval row."""
    with pytest.raises(ToolError, match="no artifact"):
        client.read_artifact("art-999-nope")


@pytest.mark.fast
def test_a_sandbox_failure_does_not_arrive_as_a_tool_error(client: MCPTools, monkeypatch):
    """MCP has one error channel and these are two different findings: a refused tool call is
    about the agent, a dead sandbox is about us. Collapsing them files our outage as the agent's
    mistake, which is why the prefix exists."""

    def die(*_args, **_kwargs):
        raise SandboxError("worker did not become ready")

    monkeypatch.setattr("mcp_server.sandbox.SandboxPool.execute", die)

    with pytest.raises(SandboxError, match="worker did not become ready"):
        client.run_python("print(1)")


@pytest.mark.fast
def test_metrics_reach_the_run_log(local: LocalTools, client: MCPTools):
    client.log_metric("run-7", "leakage_recall", 1.0)

    assert local.metrics == [("run-7", "leakage_recall", 1.0)]
    assert "leakage_recall" in (local.root / "metrics.jsonl").read_text()


def test_a_snippet_runs_and_its_artifacts_come_back(client: MCPTools):
    """Not `fast`: the first `run_python` in a session boots the sandbox worker."""
    result = client.run_python(
        "import os, json, pathlib, pandas as pd\n"
        "frame = pd.read_csv(os.environ['DS_DATASET'])\n"
        "print(json.dumps({'rows': len(frame)}))\n"
        "pathlib.Path(os.environ['DS_ARTIFACTS'], 'counts.json').write_text('{\"n\": 1}')\n"
    )

    assert result.ok, result.stderr
    assert '"rows": 200' in result.stdout
    assert len(result.artifacts_written) == 1
    assert result.duration_s > 0


def test_a_failing_snippet_is_a_result_not_an_exception(client: MCPTools):
    """`run_python` reports the agent's mistakes; it does not raise them."""
    result = client.run_python("raise ValueError('boom')")

    assert not result.ok
    assert "boom" in result.stderr


def test_the_server_runs_as_a_real_subprocess(tmp_path: Path):
    """The in-process tests above share the server object; this one proves the launch command in
    `stdio_params` works, which is the same command an external MCP client is given."""
    with MCPTools(stdio_params(tmp_path, TOY, "toy")) as client:
        payload = client.read_artifact(dataset_artifact_id("toy"), max_bytes=50)
        assert payload.truncated
        assert payload.meta.extra["n_rows"] == 200

        result = client.run_python("import ds_agents", timeout_s=30)
        # Same property the sandbox tests assert directly, restated once here because the
        # subprocess server is the configuration an outside agent actually gets.
        assert not result.ok
        assert "ModuleNotFoundError" in result.stderr


@pytest.mark.fast
def test_a_server_that_cannot_launch_leaves_no_thread_behind():
    """`MCPTools` owns a thread and an event loop. A connect failure that leaked them would
    accumulate one per dataset in a benchmark and only show up as a machine slowly dying."""
    before = threading.active_count()

    with pytest.raises(Exception):  # noqa: B017 - the SDK's error type is not part of our contract
        MCPTools(StdioServerParameters(command="/nonexistent/python", args=[]), connect_timeout_s=5)

    assert threading.active_count() == before


def test_a_server_that_never_answers_is_not_left_running():
    """Not `fast`: it waits out a connect timeout on purpose.

    The failure this guards is invisible at the call site -- `__init__` raises either way, and the
    abandoned server is reparented to init rather than showing up as a child. A Phase 4 harness
    builds one client per dataset, so one orphan per failed connect is one orphan per dataset,
    each holding a sandbox worker with pandas and scikit-learn resident.
    """
    # Unique per run: a stale orphan from an earlier, broken build would otherwise fail a fixed
    # one, and the first thing anyone would do is disbelieve the test.
    marker = f"ds-agents-hanging-server-probe-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    params = StdioServerParameters(
        command=sys.executable, args=["-c", f"import time; time.sleep(120)  # {marker}"]
    )

    with pytest.raises(Exception):  # noqa: B017 - the SDK's timeout type is not our contract
        MCPTools(params, connect_timeout_s=2)

    listing = subprocess.run(["ps", "-ax", "-o", "args"], capture_output=True, text=True).stdout
    survivors = [line for line in listing.splitlines() if marker in line and "ps -ax" not in line]
    assert survivors == [], f"the server was abandoned rather than terminated: {survivors}"


@pytest.mark.fast
def test_closing_twice_is_not_an_error(local: LocalTools):
    """The CLI closes in a `finally` and a test closes in a fixture, so the same client gets
    closed twice whenever a run raises."""
    tools = MCPTools(build_server(local))
    tools.close()
    tools.close()


@pytest.mark.fast
def test_a_call_after_close_says_so_instead_of_hanging(local: LocalTools):
    tools = MCPTools(build_server(local))
    tools.close()

    with pytest.raises(ToolError, match="not connected"):
        tools.write_artifact("late.txt", "too late")


# ---- the CLI's transport switch ------------------------------------------------------------
# `_select_tools` lives in cli.py but is tested here rather than in test_cli.py, which is entirely
# `fast`: choosing the mcp branch spawns a server, and that does not belong in a per-edit hook.


@pytest.mark.fast
def test_the_cli_local_branch_returns_the_in_process_binding(tmp_path: Path):
    tools = _select_tools("local", tmp_path, TOY, "toy")

    assert isinstance(tools, LocalTools)
    tools.close()


def test_the_cli_mcp_branch_returns_a_connected_client(tmp_path: Path):
    """`--tools mcp` is the default, so this is the wiring every published run goes through."""
    tools = _select_tools("mcp", tmp_path, TOY, "toy")

    try:
        assert isinstance(tools, MCPTools)
        assert tools.read_artifact(dataset_artifact_id("toy")).meta.extra["n_rows"] == 200
    finally:
        tools.close()


@pytest.mark.fast
def test_a_tool_server_that_will_not_start_exits_instead_of_traceback(tmp_path: Path, monkeypatch):
    """A run that cannot reach its tools never started. A traceback here would read as a pipeline
    bug, and on a benchmark it is the difference between "this dataset failed" and "the harness
    is misconfigured"."""
    monkeypatch.setattr(
        "ds_agents.cli.stdio_params",
        lambda *_args, **_kwargs: StdioServerParameters(command="/nonexistent/python", args=[]),
    )

    with pytest.raises(SystemExit, match="could not start the tool server"):
        _select_tools("mcp", tmp_path, TOY, "toy")
