"""In-process adapter over the sandbox and the artifact store.

This used to be a standalone shim that ran each snippet with `subprocess.run`. It is now a thin
binding of `mcp_server.sandbox.SandboxPool` and `mcp_server.store.ArtifactStore` to the `Tools`
Protocol, which is the point: when the MCP client arrives in the next session it binds the *same*
two objects over the protocol, so "we swapped the shim for the server" cannot quietly mean "we
wrote a second implementation and hoped it matched."

What it still is not: a container. There is no memory cap and no network block. What it does keep,
and now enforces more strictly than the subprocess version did, is the isolation the thesis rests
on -- snippets run in a forked child of a worker that has never imported `ds_agents`, with the
repo pruned off `sys.path` and an environment built from nothing. See `mcp_server/_worker.py`.
"""

from pathlib import Path
from typing import Any

from ds_agents.state import ArtifactId
from ds_agents.tools.protocol import ArtifactPayload, RunResult
from mcp_server.sandbox import SandboxPool
from mcp_server.store import ArtifactStore, MetricLog, dataset_artifact_id

__all__ = ["LocalTools", "dataset_artifact_id"]


class LocalTools:
    """Satisfies `Tools`. One instance per run; `root` holds the artifacts for that run.

    The sandbox worker boots lazily on the first `run_python`, so a test that only reads and
    writes artifacts does not pay the ~1.5s import. Close it when the run ends, or let `atexit`
    do it.
    """

    def __init__(self, root: Path, dataset_path: Path | None = None, dataset_id: str = "") -> None:
        self.root = Path(root)
        self.store = ArtifactStore(self.root)
        self.work_dir = self.root / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metric_log = MetricLog(self.root / "metrics.jsonl")
        self.dataset_path: Path | None = None
        if dataset_path is not None:
            self.dataset_path = self.register_dataset(Path(dataset_path), dataset_id)
        self.sandbox = SandboxPool(
            work_dir=self.work_dir,
            base_env={
                "DS_DATASET": str(self.dataset_path) if self.dataset_path else "",
                "DS_ARTIFACTS": str(self.store.artifacts_dir),
            },
        )

    # ---- convenience passthroughs used by the CLI and the tests -----------------------------

    @property
    def artifacts_dir(self) -> Path:
        return self.store.artifacts_dir

    @property
    def data_dir(self) -> Path:
        return self.store.data_dir

    @property
    def metrics(self) -> list[tuple[str, str, float]]:
        return self.metric_log.records

    def register_dataset(self, path: Path, dataset_id: str) -> Path:
        return self.store.register_dataset(path, dataset_id)

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self) -> "LocalTools":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- Tools ------------------------------------------------------------------------------

    def run_python(self, code: str, timeout_s: int = 60) -> RunResult:
        before = self.store.snapshot()
        # A `SandboxError` deliberately propagates instead of becoming a `ToolError`: a sandbox
        # that cannot run at all is not a snippet failure, and a node that recorded it as one
        # would put "the model wrote bad code" in the results row for a broken machine.
        outcome = self.sandbox.execute(code, timeout_s=timeout_s)
        return RunResult(
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            exit_code=outcome.exit_code,
            timed_out=outcome.timed_out,
            duration_s=outcome.duration_s,
            artifacts_written=self.store.scan(before),
        )

    def read_artifact(
        self, artifact_id: ArtifactId, max_bytes: int | None = None
    ) -> ArtifactPayload:
        return self.store.read(artifact_id, max_bytes=max_bytes)

    def write_artifact(
        self,
        name: str,
        content: str,
        kind: str = "text",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactId:
        return self.store.write(name, content, kind=kind, extra=extra)

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        self.metric_log.log(run_id, name, value)
