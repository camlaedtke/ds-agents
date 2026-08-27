"""The sandbox worker and pool.

`tests/tools/test_local.py` checks the surface a node sees. This file checks the properties the
surface rests on, because the fork-from-a-warm-worker design buys its speed by giving every
snippet a parent that is already running. Everything that parent can reach, agent code can reach.
So the isolation assertions here are not defence-in-depth; they are the design's only defence.
"""

import os
import time
from pathlib import Path

import pytest

from mcp_server.sandbox import SandboxError, SandboxPool, SandboxWorker, fork_available

pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(not fork_available(), reason="the sandbox requires os.fork"),
]


@pytest.fixture
def pool(tmp_path: Path) -> SandboxPool:
    return SandboxPool(work_dir=tmp_path, base_env={"DS_MARKER": "present"})


# ---- isolation ------------------------------------------------------------------------------


def test_the_repo_is_not_importable_from_a_snippet(pool: SandboxPool):
    """The one that matters most, and the one the Phase 1 subprocess shim got wrong.

    An editable install puts the repo root and `src/` on `sys.path` through a `.pth` file, which
    no amount of environment stripping removes -- the old shim's snippets could `import ds_agents`
    despite its docstring saying otherwise. Agent code that can read the grader's source is not
    the same experiment as agent code that reasoned, and nothing in a results table would show
    the difference.
    """
    result = pool.execute(
        "for name in ('ds_agents', 'mcp_server'):\n"
        "    try:\n"
        "        __import__(name)\n"
        "        print(name, 'IMPORTABLE')\n"
        "    except ImportError:\n"
        "        print(name, 'blocked')\n"
    )

    assert result.ok, result.stderr
    assert result.stdout.split() == ["ds_agents", "blocked", "mcp_server", "blocked"]


def test_the_worker_never_imported_ds_agents(pool: SandboxPool):
    """The child inherits the worker's memory, so the worker's import list is part of the
    sandbox boundary, not an implementation detail."""
    result = pool.execute("import sys; print([m for m in sys.modules if 'ds_agents' in m])")

    assert result.ok, result.stderr
    assert result.stdout.strip() == "[]"


def test_pandas_and_sklearn_are_already_imported(pool: SandboxPool):
    """The entire justification for the design. If the preload stops arriving, the sandbox still
    works and silently costs 0.9s a call again."""
    result = pool.execute("import sys; print('pandas' in sys.modules, 'sklearn' in sys.modules)")

    assert result.stdout.strip() == "True True"


def test_the_environment_is_built_not_inherited(pool: SandboxPool, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-visible")

    result = pool.execute("import os; print(sorted(os.environ))")

    assert "ANTHROPIC_API_KEY" not in result.stdout
    assert "DS_MARKER" in result.stdout


def test_globals_do_not_survive_between_snippets(pool: SandboxPool):
    """Each snippet is its own fork. A long-lived interpreter reused across calls would let the
    modeler's snippet see whatever the profiler's left behind, and a run would stop being
    reproducible from its inputs."""
    pool.execute("LEFTOVER = 1")
    result = pool.execute("print('LEFTOVER' in dir())")

    assert result.stdout.strip() == "False"


def test_a_snippet_cannot_write_to_the_protocol_channel(pool: SandboxPool):
    """The child's fd 1 is redirected before anything else runs. If it were not, a snippet
    printing a JSON object shaped like a reply could forge its own exit code."""
    result = pool.execute('print(\'{"exit_code": 0, "timed_out": false}\')\nraise SystemExit(3)')

    assert result.exit_code == 3
    assert '"exit_code"' in result.stdout


def test_a_snippet_reading_stdin_does_not_eat_the_next_request(pool: SandboxPool):
    """The child inherits the request pipe; stdin is redirected to /dev/null so it cannot."""
    first = pool.execute("import sys; print(repr(sys.stdin.read()))")
    second = pool.execute("print('still here')")

    assert first.ok, first.stderr
    assert second.stdout.strip() == "still here"


# ---- failure handling -----------------------------------------------------------------------


def test_a_crash_is_reported_not_raised(pool: SandboxPool):
    result = pool.execute("def f():\n    raise ValueError('boom')\nf()")

    assert not result.ok
    assert result.exit_code == 1
    assert "ValueError: boom" in result.stderr


def test_the_traceback_shows_the_snippet_not_the_sandbox(pool: SandboxPool):
    """A node explains a failure with `stderr[-500:]`. Frames from `_worker.py` at the top would
    describe the plumbing instead of the snippet."""
    result = pool.execute("raise ValueError('boom')")

    assert "_worker.py" not in result.stderr
    assert "in _child" not in result.stderr


def test_a_timeout_is_flagged_and_the_worker_survives(pool: SandboxPool):
    """Recovery is the assertion with teeth. A timeout that killed the worker would make the
    next node in the graph fail for an unrelated reason."""
    timed_out = pool.execute("import time\nwhile True:\n    time.sleep(0.05)", timeout_s=1)
    after = pool.execute("print('alive')")

    assert timed_out.timed_out
    assert timed_out.exit_code == 124
    assert after.stdout.strip() == "alive"


def test_a_timeout_kills_the_whole_process_group(pool: SandboxPool, tmp_path: Path):
    """The child calls `setsid` so a snippet that spawned helpers cannot leave them running past
    the deadline, holding a benchmark's CPU while its row is already written.

    Asserted on the grandchild's liveness rather than on a file it would eventually write: a file
    check that runs before the helper's own sleep elapses passes whether or not anything was
    killed, which is a test that can only pass.
    """
    pid_file = tmp_path / "helper.pid"
    helper = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    )
    result = pool.execute(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {helper!r}])\n"
        "time.sleep(30)\n",
        timeout_s=1,
    )

    assert result.timed_out
    helper_pid = int(_wait_for(pid_file).strip())
    for _ in range(50):
        if not _alive(helper_pid):
            break
        time.sleep(0.02)
    assert not _alive(helper_pid), f"grandchild {helper_pid} outlived the timeout"


def _wait_for(path: Path, timeout_s: float = 2.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and (text := path.read_text()):
            return text
        time.sleep(0.02)
    raise AssertionError(f"{path} never appeared; the helper never started")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_a_segfaulting_snippet_is_an_outcome_not_an_exception(pool: SandboxPool):
    result = pool.execute("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)")

    assert not result.ok
    assert result.exit_code < 0


# ---- worker lifetime ------------------------------------------------------------------------


def test_the_worker_is_shared_across_pools(tmp_path: Path):
    """Two runs, one warm worker. Booting per run cost 0.87s a unit test and would cost that per
    dataset in a benchmark."""
    first = SandboxPool(work_dir=tmp_path / "a")
    second = SandboxPool(work_dir=tmp_path / "b")

    assert first.worker is second.worker


def test_each_pool_keeps_its_own_environment_and_cwd(tmp_path: Path):
    """The corollary of sharing a worker: nothing about a run may live in it."""
    first = SandboxPool(work_dir=tmp_path / "a", base_env={"DS_RUN": "one"})
    second = SandboxPool(work_dir=tmp_path / "b", base_env={"DS_RUN": "two"})

    show = "import os; print(os.environ['DS_RUN'], os.getcwd())"
    out_a, out_b = first.execute(show).stdout, second.execute(show).stdout

    assert out_a.split()[0] == "one"
    assert out_b.split()[0] == "two"
    assert os.path.realpath(out_a.split()[1]) == os.path.realpath(tmp_path / "a")
    assert os.path.realpath(out_b.split()[1]) == os.path.realpath(tmp_path / "b")


def test_a_dead_worker_is_restarted_on_the_next_call(tmp_path: Path):
    pool = SandboxPool(work_dir=tmp_path, worker=SandboxWorker())
    pool.execute("print('first')")
    pool.worker.close()

    assert pool.execute("print('second')").stdout.strip() == "second"


def test_a_worker_that_cannot_boot_raises_sandbox_error(tmp_path: Path):
    """`SandboxError`, not `ToolError`. A machine with no working sandbox must not produce eval
    rows that read as "the agent wrote bad code"."""
    pool = SandboxPool(work_dir=tmp_path, worker=SandboxWorker(python="/nonexistent/python"))

    with pytest.raises(SandboxError):
        pool.execute("print('never')")
