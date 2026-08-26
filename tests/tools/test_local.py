"""The Phase 1 tool shim.

The test that carries weight here is the environment one. `LocalTools` is not a sandbox -- no
container, no network block -- so the two properties it does guarantee have to be asserted, or
Phase 2 will inherit a shim nobody checked: code runs in a subprocess, and that subprocess cannot
see the environment it is being graded in.
"""

from pathlib import Path

import pytest

from ds_agents.tools.local import LocalTools, dataset_artifact_id
from ds_agents.tools.protocol import ToolError

pytestmark = pytest.mark.fast

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "toy" / "toy.csv"


@pytest.fixture
def tools(tmp_path: Path) -> LocalTools:
    return LocalTools(tmp_path, dataset_path=TOY, dataset_id="toy")


def test_snippet_cannot_read_the_parent_environment(tools: LocalTools, monkeypatch):
    """If this fails, agent code can read an API key, and on a real dataset it could read the
    repo that holds `planted_leakage_columns`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-visible")

    result = tools.run_python("import os; print(sorted(os.environ))")

    assert result.ok
    assert "ANTHROPIC_API_KEY" not in result.stdout
    assert "DS_DATASET" in result.stdout


def test_snippet_runs_out_of_process(tools: LocalTools):
    """An in-process `exec` would put PipelineState inside the agent's reach."""
    result = tools.run_python(
        "import os, sys; print(os.getpid()); print('ds_agents' in sys.modules)"
    )

    child_pid, imported = result.stdout.split()
    assert int(child_pid) != __import__("os").getpid()
    assert imported == "False"


def test_nonzero_exit_is_reported_not_raised(tools: LocalTools):
    result = tools.run_python("raise ValueError('boom')")

    assert not result.ok
    assert result.exit_code != 0
    assert "boom" in result.stderr


def test_timeout_is_flagged(tools: LocalTools):
    result = tools.run_python("import time; time.sleep(5)", timeout_s=1)

    assert result.timed_out
    assert not result.ok


def test_files_dropped_in_the_artifacts_dir_are_indexed(tools: LocalTools):
    code = (
        "import json, os\n"
        "with open(os.path.join(os.environ['DS_ARTIFACTS'], 'out.json'), 'w') as fh:\n"
        "    json.dump({'rows': 3}, fh)\n"
    )
    result = tools.run_python(code)

    (artifact_id,) = result.artifacts_written
    payload = tools.read_artifact(artifact_id)
    assert payload.meta.kind == "json"
    assert payload.meta.extra["keys"] == ["rows"]
    assert '"rows": 3' in payload.content


def test_dataset_is_registered_with_the_schema_intake_needs(tools: LocalTools):
    payload = tools.read_artifact(dataset_artifact_id("toy"))

    assert payload.meta.kind == "table"
    assert payload.meta.extra["n_rows"] == 200
    assert "account_status_code" in payload.meta.extra["columns"]
    # Intake tells a target from an id column by cardinality, so this has to be there before any
    # run_python call has happened.
    assert payload.meta.extra["n_unique"]["churned"] == 2
    assert payload.meta.extra["n_unique"]["customer_id"] == 200


def test_the_repo_copy_of_the_dataset_is_not_what_the_snippet_touches(tools: LocalTools):
    """'read-only mount' has to mean the fixture on disk is safe from agent code."""
    before = TOY.read_bytes()
    tools.run_python("import os; open(os.environ['DS_DATASET'], 'a').write('junk')")

    assert TOY.read_bytes() == before


def test_truncation_is_declared(tools: LocalTools):
    payload = tools.read_artifact(dataset_artifact_id("toy"), max_bytes=120)

    assert payload.truncated is True
    assert len(payload.content) == 120


def test_unknown_artifact_raises_tool_error(tools: LocalTools):
    with pytest.raises(ToolError):
        tools.read_artifact("art-nope")


def test_write_then_read_round_trips(tools: LocalTools):
    artifact_id = tools.write_artifact("notes.md", "one legit strong feature", kind="report")

    payload = tools.read_artifact(artifact_id)
    assert payload.content == "one legit strong feature"
    assert payload.meta.kind == "report"
    assert payload.truncated is False


def test_log_metric_records_the_run_id(tools: LocalTools):
    tools.log_metric("run-1", "leakage_candidates", 1)

    assert tools.metrics == [("run-1", "leakage_candidates", 1.0)]


def test_overwriting_an_artifact_file_issues_a_new_id_and_leaves_the_old_one_alone(
    tools: LocalTools,
):
    """`split_manifest.json` is a fixed name, and `split_artifact` is supposed to be pinned and
    never rewritten. If a second write could change the bytes behind an issued id, that invariant
    would be worth nothing and the run would still report `artifacts_written: []`."""
    write = (
        "import json, os\n"
        "with open(os.path.join(os.environ['DS_ARTIFACTS'], 'm.json'), 'w') as fh:\n"
        "    json.dump({{'v': {value}}}, fh)\n"
    )
    first = tools.run_python(write.format(value=1))
    second = tools.run_python(write.format(value=2))

    (first_id,), (second_id,) = first.artifacts_written, second.artifacts_written
    assert first_id != second_id
    assert '"v": 1' in tools.read_artifact(first_id).content
    assert '"v": 2' in tools.read_artifact(second_id).content
