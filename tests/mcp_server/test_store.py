"""The artifact store and the metric log.

`tests/tools/test_local.py` exercises most of this through the `Tools` surface. What is asserted
here is what the extraction added, and the two invariants the Phase 2 MCP server must not quietly
reimplement differently: an artifact is immutable once issued, and a write is detected by content
rather than by filename.
"""

import json
from pathlib import Path

import pytest

from ds_agents.tools.protocol import ToolError
from mcp_server.store import ArtifactStore, MetricLog, dataset_artifact_id

pytestmark = pytest.mark.fast

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "toy" / "toy.csv"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path)
    store.register_dataset(TOY, "toy")
    return store


def test_every_artifact_carries_a_sandbox_path(store: ArtifactStore):
    """The parking-lot fix from Phase 1. The split manifest is currently rendered into snippet
    source, which is O(n_rows) -- about 7 MB on a 100k-row Phase 4 dataset. A snippet can only
    `open()` it instead if the path is in the metadata."""
    artifact_id = store.write("notes.txt", "hello")

    meta = store.meta(artifact_id)
    assert Path(meta.extra["sandbox_path"]).read_text() == "hello"
    assert store.path_of(artifact_id) == Path(meta.extra["sandbox_path"])


def test_the_dataset_artifact_carries_the_schema_intake_needs(store: ArtifactStore):
    """intake has `read_artifact` and nothing else, by design: giving the first node in the graph
    `run_python` would let it execute code before anything has been profiled."""
    meta = store.meta(dataset_artifact_id("toy"))

    assert meta.kind == "table"
    assert "customer_id" in meta.extra["columns"]
    assert meta.extra["n_rows"] == 200
    assert meta.extra["n_unique"]["customer_id"] == 200


def test_a_snippet_written_file_is_frozen_under_its_own_id(store: ArtifactStore):
    """Indexed where it lies, a later snippet reusing the filename would change the bytes behind
    an already-issued id. `split_artifact` is pinned by the profiler and never rewritten, and that
    guarantee is only as good as this copy."""
    dropped = store.artifacts_dir / "split_manifest.json"

    before = store.snapshot()
    dropped.write_text(json.dumps({"train": [0, 1]}))
    (first,) = store.scan(before)

    before = store.snapshot()
    dropped.write_text(json.dumps({"train": [9]}))
    (second,) = store.scan(before)

    assert first != second
    assert json.loads(store.read(first).content) == {"train": [0, 1]}
    assert json.loads(store.read(second).content) == {"train": [9]}


def test_an_unchanged_artifacts_dir_reports_no_writes(store: ArtifactStore):
    assert store.scan(store.snapshot()) == []


def test_reading_an_unknown_artifact_raises_tool_error(store: ArtifactStore):
    with pytest.raises(ToolError):
        store.read("art-999-nope")
    with pytest.raises(ToolError):
        store.path_of("art-999-nope")


def test_truncation_is_declared(store: ArtifactStore):
    """A node that rendered a truncated artifact into a prompt as if it were whole would report
    on rows it never saw."""
    payload = store.read(dataset_artifact_id("toy"), max_bytes=120)

    assert payload.truncated
    assert len(payload.content) == 120


def test_the_registered_dataset_is_a_read_only_copy(store: ArtifactStore, tmp_path: Path):
    landed = store.path_of(dataset_artifact_id("toy"))

    assert landed != TOY
    assert landed.stat().st_mode & 0o222 == 0
    assert landed.read_bytes() == TOY.read_bytes()


def test_metrics_survive_the_process_that_wrote_them(tmp_path: Path):
    """Phase 1 kept metrics on the tools object, which is unreadable the moment the process
    exits -- exactly the runs a benchmark most wants to look at."""
    path = tmp_path / "metrics.jsonl"
    log = MetricLog(path)

    log.log("run-1", "claimed_holdout_score", 0.983)
    log.log("run-1", "leakage_candidates", 2)

    assert log.records == [
        ("run-1", "claimed_holdout_score", 0.983),
        ("run-1", "leakage_candidates", 2.0),
    ]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0] == {"run_id": "run-1", "name": "claimed_holdout_score", "value": 0.983}
    assert rows[1]["value"] == 2.0


def test_a_metric_log_without_a_path_still_records(tmp_path: Path):
    log = MetricLog()
    log.log("run-1", "x", 1)
    assert log.records == [("run-1", "x", 1.0)]
