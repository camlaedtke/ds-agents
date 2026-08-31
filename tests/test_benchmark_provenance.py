"""The online half: does `evals/datasets/manifest.yaml` still agree with where it came from?

Opt-in, because `uv run pytest` is promised hermetic and green offline and because these hit two
third-party services. Run with:

    DS_AGENTS_NETWORK_TESTS=1 uv run pytest tests/test_benchmark_provenance.py

A failure here is a FINDING, not a flake. An md5 that stops matching means OpenML re-versioned a
dataset under us, and every number this repo has published about that dataset is then about a file
that no longer exists at that id.

Deliberately NOT gated by putting `-m "not network"` in `addopts`: pytest's `-m` is a single
option, so the edit hook's `pytest -m fast` would REPLACE that filter rather than intersect with
it, and these would start running on every keystroke. An env var cannot be overridden by accident.
"""

import json
import os
import urllib.request

import pytest
import yaml

from ds_agents.benchmark import load_manifest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("DS_AGENTS_NETWORK_TESTS") != "1",
        reason="hits openml.org and raw.githubusercontent.com; set DS_AGENTS_NETWORK_TESTS=1",
    ),
]

MANIFEST = load_manifest()
AMLB_RAW = "https://raw.githubusercontent.com/openml/automlbenchmark/{ref}/{path}"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


@pytest.mark.parametrize("entry", MANIFEST.datasets, ids=lambda e: e.dataset_id)
def test_entry_matches_the_live_openml_data_description(entry) -> None:
    """Every pinned upstream field, against the API it was read from."""
    description = _get_json(MANIFEST.source.openml_data_api.format(data_id=entry.openml_data_id))[
        "data_set_description"
    ]
    assert int(description["id"]) == entry.openml_data_id
    assert description["name"] == entry.openml_name
    assert int(description["version"]) == entry.openml_version
    assert description["format"] == entry.openml_format
    assert description["md5_checksum"] == entry.md5_checksum
    assert description["url"] == entry.openml_url


@pytest.mark.parametrize("entry", MANIFEST.datasets, ids=lambda e: e.dataset_id)
def test_entry_task_still_points_at_the_recorded_data_id(entry) -> None:
    """The task -> data mapping is the one thing a name change would silently break."""
    task = _get_json(f"https://www.openml.org/api/v1/json/task/{entry.openml_task_id}")["task"]
    source = next(i for i in task["input"] if i["name"] == "source_data")["data_set"]
    assert int(source["data_set_id"]) == entry.openml_data_id
    assert source["target_feature"] == entry.target


def test_every_entry_is_a_member_of_its_declared_amlb_list_at_the_pinned_ref() -> None:
    """The provenance claim the citation rests on, checked against immutable bytes."""
    membership: dict[str, set[int]] = {}
    for list_name, path in MANIFEST.source.amlb_lists.items():
        url = AMLB_RAW.format(ref=MANIFEST.source.amlb_ref, path=path)
        with urllib.request.urlopen(url, timeout=60) as response:
            entries = yaml.safe_load(response.read().decode("utf-8"))
        membership[list_name] = {int(e["openml_task_id"]) for e in entries}

    for entry in MANIFEST.datasets:
        assert entry.openml_task_id in membership[entry.amlb_list], entry.dataset_id
