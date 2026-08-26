"""In-process implementation of the tool surface, for Phase 1.

This is a shim, not a sandbox. It has the signatures the MCP server will expose and none of the
isolation: no container, no memory cap, no network block. What it does keep is the two properties
the thesis actually rests on:

- **Code runs in a subprocess, never `exec`.** An in-process `exec` would put `PipelineState` --
  including `planted_leakage_columns` -- inside the agent's reach, and a leakage_recall of 1.0
  would no longer distinguish reasoning from reading the answer key.
- **The environment is stripped.** The snippet gets a minimal env with the dataset and artifact
  paths and nothing else, so agent code cannot read API keys or the repo it is being graded in.

Phase 2 replaces this file with an MCP client. Nothing in `nodes/` should need to change.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ds_agents.state import ArtifactId
from ds_agents.tools.protocol import ArtifactMeta, ArtifactPayload, RunResult, ToolError

_SLUG = re.compile(r"[^a-z0-9]+")


def dataset_artifact_id(dataset_id: str) -> ArtifactId:
    """The artifact id a dataset is registered under at run start.

    docs/ARCHITECTURE.md gives intake only `read_artifact`, but intake has to name `spec.target`,
    which means seeing the columns. Registering the dataset as an artifact is the smaller of the
    two fixes; the other was giving intake `run_python`, which would let the first node in the
    graph execute arbitrary code before anything has been profiled.
    """
    return f"dataset:{dataset_id}"


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-") or "artifact"


class LocalTools:
    """Satisfies `Tools`. One instance per run; `root` holds the artifacts for that run."""

    def __init__(self, root: Path, dataset_path: Path | None = None, dataset_id: str = "") -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.work_dir = self.root / "work"
        self.data_dir = self.root / "data"
        for directory in (self.artifacts_dir, self.work_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._index: dict[ArtifactId, ArtifactMeta] = {}
        self._paths: dict[ArtifactId, Path] = {}
        self._counter = 0
        self.metrics: list[tuple[str, str, float]] = []
        self.dataset_path: Path | None = None
        if dataset_path is not None:
            self.dataset_path = self.register_dataset(Path(dataset_path), dataset_id)

    # ---- run-start registration -------------------------------------------------------------

    def register_dataset(self, path: Path, dataset_id: str) -> Path:
        """Copy the dataset into the run's read-only mount and index it as an artifact.

        The copy is what makes "read-only" true for the shim: agent code that writes to the
        dataset corrupts its own scratch copy, not `tests/fixtures/`.
        """
        landed = self.data_dir / path.name
        shutil.copyfile(path, landed)
        landed.chmod(0o444)
        artifact_id = dataset_artifact_id(dataset_id)
        frame = pd.read_csv(landed)
        self._index[artifact_id] = ArtifactMeta(
            id=artifact_id,
            name=path.name,
            kind="table",
            n_bytes=landed.stat().st_size,
            extra={
                "columns": list(frame.columns),
                "n_rows": int(len(frame)),
                "dtypes": {c: str(d) for c, d in frame.dtypes.items()},
                # n_unique at registration so intake can tell a target from an id column without
                # spending a `run_python` call before anything has been profiled.
                "n_unique": {c: int(frame[c].nunique(dropna=True)) for c in frame.columns},
                "sandbox_path": str(landed),
            },
        )
        self._paths[artifact_id] = landed
        return landed

    # ---- Tools ------------------------------------------------------------------------------

    def run_python(self, code: str, timeout_s: int = 60) -> RunResult:
        self._counter += 1
        script = self.work_dir / f"snippet_{self._counter:03d}.py"
        script.write_text(code)
        before = self._artifact_state()
        env = {
            "PATH": os.defpath,
            "HOME": str(self.work_dir),
            "PYTHONHASHSEED": "0",
            "DS_DATASET": str(self.dataset_path) if self.dataset_path else "",
            "DS_ARTIFACTS": str(self.artifacts_dir),
        }
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(  # noqa: S603 - the point of this call is running the snippet
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                cwd=self.work_dir,
                env=env,
                timeout=timeout_s,
                check=False,
            )
            stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            exit_code = 124
        duration = time.monotonic() - started
        after = self._artifact_state()
        # New files AND changed ones. A path-set diff alone misses a snippet that overwrites a
        # name it wrote before -- `split_manifest.json` is a fixed name -- which would report
        # nothing written while silently mutating the content behind an already-issued id.
        changed = sorted(path for path, stamp in after.items() if before.get(path) != stamp)
        written = [self._index_file(path) for path in changed]
        return RunResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_s=duration,
            artifacts_written=written,
        )

    def read_artifact(
        self, artifact_id: ArtifactId, max_bytes: int | None = None
    ) -> ArtifactPayload:
        meta = self._index.get(artifact_id)
        if meta is None:
            raise ToolError(f"no artifact {artifact_id!r}")
        text = self._paths[artifact_id].read_text()
        truncated = max_bytes is not None and len(text) > max_bytes
        return ArtifactPayload(
            meta=meta, content=text[:max_bytes] if truncated else text, truncated=truncated
        )

    def write_artifact(
        self,
        name: str,
        content: str,
        kind: str = "text",
        extra: dict[str, Any] | None = None,
    ) -> ArtifactId:
        self._counter += 1
        artifact_id = f"art-{self._counter:03d}-{_slug(name)}"
        path = self.artifacts_dir / f"{artifact_id}-{name}"
        path.write_text(content)
        self._index[artifact_id] = ArtifactMeta(
            id=artifact_id,
            name=name,
            kind=kind,
            n_bytes=path.stat().st_size,
            extra=extra or {},
        )
        self._paths[artifact_id] = path
        return artifact_id

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        self.metrics.append((run_id, name, float(value)))

    # ---- internals --------------------------------------------------------------------------

    def _artifact_state(self) -> dict[Path, tuple[int, int]]:
        """Path -> (mtime, size), so an overwrite is as visible as a new file."""
        return {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in self.artifacts_dir.iterdir()
            if path.is_file()
        }

    def _index_file(self, path: Path) -> ArtifactId:
        """Register a file a snippet dropped into DS_ARTIFACTS itself.

        The file is COPIED under its artifact id rather than indexed where it lies. An artifact
        has to be immutable once issued: `split_artifact` is pinned before feature_eng runs and
        never rewritten, and that invariant is worth nothing if a later snippet can change the
        bytes behind the id by writing the same filename again.
        """
        self._counter += 1
        artifact_id = f"art-{self._counter:03d}-{_slug(path.stem)}"
        frozen = self.artifacts_dir / f"{artifact_id}-{path.name}"
        shutil.copyfile(path, frozen)
        extra: dict[str, Any] = {}
        if path.suffix == ".json":
            try:
                payload = json.loads(frozen.read_text())
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                extra = {"keys": sorted(payload)}
        self._index[artifact_id] = ArtifactMeta(
            id=artifact_id,
            name=path.name,
            kind="json" if path.suffix == ".json" else "text",
            n_bytes=frozen.stat().st_size,
            extra=extra,
        )
        self._paths[artifact_id] = frozen
        return artifact_id
