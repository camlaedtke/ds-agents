"""Parent side of the sandbox: one warm worker per process, a fresh forked child per snippet.

Why a warm worker at all: every `run_python` call used to pay a fresh pandas + scikit-learn
import, measured at 0.90s against 0.013s for a bare interpreter. A toy run makes four such calls
and takes about 16s wall, so imports alone were a quarter of it, and a Phase 4 benchmark
multiplies that by every dataset. Forking from a pre-imported worker takes the per-call cost to
~0.04s including a RandomForest fit. See docs/DECISIONS.md 2026-08-27.

Why the worker is shared across runs rather than one per run: it holds no run state. The
environment a snippet sees, its working directory, and its output paths all ride on the request,
so two runs sharing a worker are as isolated from each other as two snippets in one run -- which
is to say completely, because each gets its own forked process. Booting one per `SandboxPool`
instead cost 0.87s per unit test and would cost that again per dataset in a benchmark.

The isolation properties live in `_worker.py`. This module owns process lifetime, the request
protocol, and the outer deadline.
"""

import atexit
import contextlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_WORKER = Path(__file__).with_name("_worker.py")
# The worker enforces the snippet timeout itself by killing the child. This is the parent's
# backstop for the worker never answering at all -- a fork failure, an OOM kill, a wedged pipe.
# Without it a caller blocks on readline forever and a benchmark run hangs instead of erroring.
_REPLY_SLACK_S = 15.0
_BOOT_TIMEOUT_S = 120.0


class SandboxError(RuntimeError):
    """The sandbox itself failed, as opposed to the snippet failing inside it."""


@dataclass(frozen=True)
class ExecOutcome:
    """What one snippet did. Deliberately not a `RunResult`: artifact detection belongs to the
    store, so the MCP server and the in-process adapter share one implementation of it."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class SandboxWorker:
    """The long-lived process. Booted lazily, restarted if it dies, one per interpreter.

    Its own environment is stripped for the same reason the child's is: no ancestor of a snippet
    should ever have held an API key, and the fork inherits this process's memory.
    """

    def __init__(self, python: str = sys.executable) -> None:
        self.python = python
        self.preloaded: list[str] = []
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running:
            return
        env = {
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            # Not a preference: without it a user site-packages directory survives
            # `_worker._prune_sys_path`, which `sysconfig` reports as a legitimate purelib.
            "PYTHONNOUSERSITE": "1",
        }
        try:
            proc = subprocess.Popen(  # noqa: S603 - launching the sandbox is this module's job
                [self.python, str(_WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )
        except OSError as exc:
            raise SandboxError(f"could not launch the sandbox worker: {exc}") from exc
        self._proc = proc
        # The worker also exits on its own when this process dies and the request pipe closes;
        # atexit is for the ordinary case, so a test session does not accumulate workers.
        atexit.register(self.close)
        line = self._readline(_BOOT_TIMEOUT_S)
        if line is None:
            self.close()
            raise SandboxError("sandbox worker did not become ready")
        try:
            hello = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise SandboxError(f"sandbox worker sent garbage on boot: {line!r}") from exc
        if not hello.get("ready"):
            self.close()
            raise SandboxError(f"sandbox worker refused to start: {hello!r}")
        self.preloaded = list(hello.get("preloaded", []))

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin and not proc.stdin.closed:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream and not stream.closed:
                stream.close()

    def roundtrip(self, request: dict, deadline_s: float) -> dict:
        """Send one request, wait for its reply. Serialised: the worker answers in order, so two
        callers interleaving writes would hand each other's replies back."""
        with self._lock:
            self.start()
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise SandboxError("sandbox worker is not running")
            try:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                self.close()
                raise SandboxError("sandbox worker died before the snippet was sent") from exc
            line = self._readline(deadline_s)
            if line is None:
                # Either the worker exited or it is wedged. Both are unrecoverable for this call,
                # and both must leave the worker closed so the next call reboots rather than
                # reading a stale reply off a desynchronised pipe.
                stderr = _drain(proc)
                self.close()
                detail = f": {stderr.strip()[-500:]}" if stderr.strip() else ""
                raise SandboxError(f"sandbox worker stopped responding{detail}")
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise SandboxError(f"sandbox worker sent garbage: {line!r}") from exc

    def _readline(self, timeout_s: float) -> str | None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.5))
            if ready:
                line = proc.stdout.readline()
                # EOF: the worker exited. Distinguishable from "nothing yet" only by the empty
                # string, which is why this is not a bare readline with a signal alarm.
                return line or None
            if proc.poll() is not None:
                return None


_SHARED: SandboxWorker | None = None
_SHARED_LOCK = threading.Lock()


def shared_worker() -> SandboxWorker:
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = SandboxWorker()
        return _SHARED


@dataclass
class SandboxPool:
    """One run's view of the sandbox: its working directory and the environment its snippets see.

    `base_env` is built from nothing rather than inherited, so a snippet's environment contains
    exactly what this run put there.
    """

    work_dir: Path
    base_env: dict[str, str] = field(default_factory=dict)
    worker: SandboxWorker | None = None
    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._io_dir = self.work_dir / ".sandbox_io"
        self._io_dir.mkdir(parents=True, exist_ok=True)
        if self.worker is None:
            self.worker = shared_worker()
        # PATH and HOME are part of what a snippet sees, not of how the worker is launched. The
        # Phase 1 shim gave snippets both, and a snippet whose HOME is unset writes dotfiles into
        # whatever directory it happens to be standing in.
        self.base_env = {
            "PATH": os.defpath,
            "HOME": str(self.work_dir),
            "PYTHONHASHSEED": "0",
            **self.base_env,
        }

    def _worker(self) -> SandboxWorker:
        return self.worker if self.worker is not None else shared_worker()

    def start(self) -> None:
        self._worker().start()

    def close(self) -> None:
        """Drop this run's claim on the worker.

        A no-op when the worker is shared: another run may still be using it, and the boot cost is
        the whole reason it is shared. `atexit` closes it for real.
        """

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def execute(
        self,
        code: str,
        timeout_s: float = 60.0,
        env_overlay: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> ExecOutcome:
        """Run `code` in a fresh forked child. Raises `SandboxError` only when the sandbox itself
        failed; a snippet that crashes or times out comes back as a non-ok `ExecOutcome`."""
        self._counter += 1
        stem = self._io_dir / f"snippet_{self._counter:03d}"
        out_path, err_path = stem.with_suffix(".out"), stem.with_suffix(".err")
        script = stem.with_suffix(".py")
        # Written for debugging only. The worker compiles the string it is handed, so editing
        # this file after the fact changes nothing that ran.
        script.write_text(code)
        request = {
            "code": code,
            "filename": str(script),
            "timeout_s": float(timeout_s),
            "cwd": str(cwd or self.work_dir),
            "env": {**self.base_env, **(env_overlay or {})},
            "stdout_path": str(out_path),
            "stderr_path": str(err_path),
        }
        started = time.monotonic()
        reply = self._worker().roundtrip(request, float(timeout_s) + _REPLY_SLACK_S)
        if "error" in reply:
            raise SandboxError(f"sandbox worker failed to run the snippet: {reply['error']}")
        return ExecOutcome(
            stdout=_read(out_path),
            stderr=_read(err_path),
            exit_code=int(reply["exit_code"]),
            timed_out=bool(reply["timed_out"]),
            duration_s=float(reply.get("duration_s", time.monotonic() - started)),
        )


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        # The child was killed before it opened its output files.
        return ""


def _drain(proc: subprocess.Popen) -> str:
    if proc.stderr is None:
        return ""
    try:
        os.set_blocking(proc.stderr.fileno(), False)
        return proc.stderr.read() or ""
    except (OSError, ValueError):
        return ""


def fork_available() -> bool:
    """False on a platform without `fork`, which this design requires."""
    return hasattr(os, "fork")
