"""The artifact store and the metric log, extracted so one implementation serves both callers.

`tools/local.py` uses these in-process today and the MCP server will use the same objects behind
the protocol next session. That is the point of the extraction: if the store's rules lived in the
shim, the Phase 2 swap would silently be a *reimplementation*, and the single-agent-vs-team
ablation would be comparing two stores that only look alike.

Two rules the store enforces, both load-bearing rather than tidiness:

- **An artifact is immutable once issued.** A file a snippet drops in `DS_ARTIFACTS` is copied
  under its artifact id, never indexed where it lies. `split_artifact` is pinned by the profiler
  and "never rewritten" means nothing if a later snippet can change the bytes behind the id by
  reusing a filename.
- **New files and changed files both count as writes.** Detection is by content stamp, not by
  path. `split_manifest.json` is a fixed name; a path-set diff would report nothing written while
  the content underneath an already-issued id changed.

This module runs in the orchestrator process, not in the sandbox. `_worker.py` imports none of it.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ds_agents.state import ArtifactId
from ds_agents.tools.protocol import (
    DEFAULT_READ_BYTES,
    ArtifactMeta,
    ArtifactPayload,
    ToolError,
)

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


class ArtifactStore:
    """The run's artifact index. One instance per run; `root` holds its files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.data_dir = self.root / "data"
        for directory in (self.artifacts_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._index: dict[ArtifactId, ArtifactMeta] = {}
        self._paths: dict[ArtifactId, Path] = {}
        self._counter = 0
        self.dataset_path: Path | None = None

    # ---- run-start registration -------------------------------------------------------------

    def register_dataset(self, path: Path, dataset_id: str) -> Path:
        """Copy the dataset into the run's read-only mount and index it as an artifact.

        The copy is what makes "read-only" true: agent code that writes to the dataset corrupts
        its own scratch copy, not `tests/fixtures/`.
        """
        path = Path(path)
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
        self.dataset_path = landed
        return landed

    # ---- the two artifact tools -------------------------------------------------------------

    def read(self, artifact_id: ArtifactId, max_bytes: int | None = None) -> ArtifactPayload:
        """Read an artifact, capped.

        A caller that names no `max_bytes` still gets one: see `DEFAULT_READ_BYTES`. "No limit"
        is not an option on purpose -- the two callers that read artifacts render them into a
        prompt or into snippet source, and neither has a size it could survive.
        """
        meta = self._index.get(artifact_id)
        if meta is None:
            raise ToolError(f"no artifact {artifact_id!r}")
        text = self._paths[artifact_id].read_text()
        limit = DEFAULT_READ_BYTES if max_bytes is None else max_bytes
        truncated = len(text) > limit
        return ArtifactPayload(
            meta=meta, content=text[:limit] if truncated else text, truncated=truncated
        )

    def write(
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
            extra={**(extra or {}), "sandbox_path": str(path)},
        )
        self._paths[artifact_id] = path
        return artifact_id

    def path_of(self, artifact_id: ArtifactId) -> Path:
        """Where a snippet can `open()` the artifact.

        The alternative -- rendering an artifact into snippet source, which is what the split
        manifest does today -- is O(n_rows) in the snippet text: about 7 MB on a 100k-row Phase 4
        dataset. `sandbox_path` in `ArtifactMeta.extra` is the same value, for callers that
        already hold the metadata.
        """
        if artifact_id not in self._paths:
            raise ToolError(f"no artifact {artifact_id!r}")
        return self._paths[artifact_id]

    def meta(self, artifact_id: ArtifactId) -> ArtifactMeta:
        meta = self._index.get(artifact_id)
        if meta is None:
            raise ToolError(f"no artifact {artifact_id!r}")
        return meta

    # ---- snippet-written artifacts ----------------------------------------------------------

    def snapshot(self) -> dict[Path, tuple[int, int]]:
        """Path -> (mtime, size), so an overwrite is as visible as a new file."""
        return {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in self.artifacts_dir.iterdir()
            if path.is_file()
        }

    def scan(self, before: dict[Path, tuple[int, int]]) -> list[ArtifactId]:
        """Index every file in the artifacts dir that appeared or changed since `before`."""
        after = self.snapshot()
        changed = sorted(path for path, stamp in after.items() if before.get(path) != stamp)
        return [self._index_file(path) for path in changed]

    def _index_file(self, path: Path) -> ArtifactId:
        """Freeze a file a snippet dropped into DS_ARTIFACTS under a fresh, immutable id."""
        self._counter += 1
        artifact_id = f"art-{self._counter:03d}-{_slug(path.stem)}"
        frozen = self.artifacts_dir / f"{artifact_id}-{path.name}"
        shutil.copyfile(path, frozen)
        extra: dict[str, Any] = {"sandbox_path": str(frozen)}
        if path.suffix == ".json":
            try:
                payload = json.loads(frozen.read_text())
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                extra["keys"] = sorted(payload)
        self._index[artifact_id] = ArtifactMeta(
            id=artifact_id,
            name=path.name,
            kind="json" if path.suffix == ".json" else "text",
            n_bytes=frozen.stat().st_size,
            extra=extra,
        )
        self._paths[artifact_id] = frozen
        return artifact_id


class MetricLog:
    """`log_metric` behind a file, so a crashed run still leaves its metrics behind.

    Phase 1 kept these in a list on the tools object, which is unreadable the moment the process
    that owns it exits -- exactly the runs a benchmark most wants to look at.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[tuple[str, str, float]] = []

    def log(self, run_id: str, name: str, value: float) -> None:
        record = (run_id, name, float(value))
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(
                    json.dumps({"run_id": run_id, "name": name, "value": float(value)}) + "\n"
                )
