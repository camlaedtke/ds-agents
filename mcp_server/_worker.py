"""The warm sandbox worker. Runs as a bare script, never as an importable part of this project.

One of these is launched per run with a stripped environment. It pays the pandas/scikit-learn
import cost once, then forks a fresh child per snippet, which inherits those imports through
copy-on-write and starts near-instantly. Every snippet still gets its own process: globals do not
survive between calls, a crash cannot take the worker with it, and a timeout is a `SIGKILL` to the
child's process group.

Three properties this file exists to guarantee, all of them tested in tests/mcp_server/:

- **It never imports `ds_agents`.** The forked child inherits this process's memory, so anything
  reachable here is reachable by agent code. `PipelineState` -- and therefore
  `planted_leakage_columns` -- lives in the orchestrator process, which is not an ancestor of any
  snippet. `_prune_sys_path` then removes the repo from the import path outright, so agent code
  cannot even reach the source it is being graded by.
- **The protocol channel is not writable by a snippet.** Requests and responses are
  line-delimited JSON on this process's stdin/stdout. The child redirects fds 0, 1 and 2 before
  it runs anything, so a snippet that prints cannot forge a response.
- **The environment the child sees is built, not inherited.** `os.environ` is cleared and
  replaced with exactly what the request carries.
"""

import contextlib
import json
import os
import signal
import sys
import time
import traceback

_WORKER_FILE = os.path.abspath(__file__)
# Long enough that a well-behaved snippet never hits it, short enough that a wedged child cannot
# hold a benchmark run open indefinitely if a caller passes a nonsense timeout.
_POLL_S = 0.002


def _prune_sys_path() -> None:
    """Keep only the standard library and site-packages.

    Without this the sandbox can `import ds_agents`: an editable install puts the repo root and
    `src/` on `sys.path` through a `.pth` file, which no amount of environment stripping removes.
    That would not hand a snippet the live state object, but it would hand it the source of the
    grader, and "the agent read the scoring code" is not a distinction we could recover from a
    results table afterwards.
    """
    import sysconfig

    allowed = tuple(
        os.path.abspath(path)
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if (path := sysconfig.get_paths().get(key))
    )
    sys.path[:] = [
        entry for entry in sys.path if entry and os.path.abspath(entry).startswith(allowed)
    ]


_prune_sys_path()

# The whole point of the worker. Imported here, once, before any fork.
import numpy  # noqa: E402, F401
import pandas  # noqa: E402, F401
import sklearn.ensemble  # noqa: E402, F401
import sklearn.inspection  # noqa: E402, F401
import sklearn.linear_model  # noqa: E402, F401
import sklearn.metrics  # noqa: E402, F401
import sklearn.model_selection  # noqa: E402, F401

PRELOADED = [
    "numpy",
    "pandas",
    "sklearn.ensemble",
    "sklearn.inspection",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.model_selection",
]


def _format_snippet_traceback() -> str:
    """The snippet's traceback without this file's frames.

    A raw `print_exc()` from inside `_child` starts with the worker's own `exec` line, so a node
    doing `stderr[-500:]` to explain a failure would show the sandbox's plumbing instead of the
    snippet's error.
    """
    kind, value, tb = sys.exc_info()
    frames = [f for f in traceback.extract_tb(tb) if os.path.abspath(f.filename) != _WORKER_FILE]
    parts = ["Traceback (most recent call last):\n"]
    parts += traceback.format_list(frames)
    parts += traceback.format_exception_only(kind, value)
    return "".join(parts)


def _child(request: dict) -> None:
    """Runs in the forked child. Never returns; always `os._exit`."""
    code = 1
    try:
        # fds first, before anything can raise with the protocol pipe still on fd 1. stdin is
        # redirected too: the child inherits the request pipe, and a snippet calling input()
        # would otherwise consume the next request.
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        for path, fd in ((request["stdout_path"], 1), (request["stderr_path"], 2)):
            handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(handle, fd)
        sys.stdout = os.fdopen(1, "w", buffering=1)
        sys.stderr = os.fdopen(2, "w", buffering=1)
        # Own process group, so a timeout kills anything the snippet spawned as well.
        os.setsid()
        os.environ.clear()
        os.environ.update(request["env"])
        os.chdir(request["cwd"])
        namespace = {"__name__": "__main__", "__file__": request.get("filename", "<snippet>")}
        exec(compile(request["code"], request.get("filename", "<snippet>"), "exec"), namespace)
        code = 0
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        with contextlib.suppress(Exception):
            sys.stderr.write(_format_snippet_traceback())
        code = 1
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os._exit(code)


def _execute(request: dict) -> dict:
    timeout_s = float(request["timeout_s"])
    sys.stdout.flush()
    sys.stderr.flush()
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        _child(request)
    deadline = started + timeout_s
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return {
                "exit_code": os.waitstatus_to_exitcode(status),
                "timed_out": False,
                "duration_s": time.monotonic() - started,
            }
        if time.monotonic() >= deadline:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # setsid() had not run yet, or the child is already reaped.
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {
                "exit_code": 124,
                "timed_out": True,
                "duration_s": time.monotonic() - started,
            }
        time.sleep(_POLL_S)


def main() -> int:
    sys.stdout.write(json.dumps({"ready": True, "preloaded": PRELOADED}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = _execute(json.loads(line))
        except Exception as exc:  # noqa: BLE001 - a worker that dies here strands the caller
            response = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
