"""`evals/datasets/manifest.yaml` is ground truth, and it is generated rather than written.

These tests are the offline half of that claim: they check that the committed file has the shape,
the provenance fields and the internal consistency the registry promises, without touching the
network. `tests/test_benchmark_provenance.py` is the online half, which checks the same file
against the APIs it came from.

The two tests worth reading before the rest are `test_no_manifest_field_can_reach_baseline_score`
and `test_nothing_in_nodes_imports_a_dataset_registry`. Both encode rules that are otherwise only
prose in a docstring, and both describe failures that would produce a plausible-looking number
rather than an error.
"""

import ast
import re
from pathlib import Path
from typing import get_args

import pytest
import yaml

from ds_agents import benchmark, fixtures
from ds_agents.benchmark import (
    SELECTION_RULE,
    DatasetManifest,
    load_dataset,
    load_manifest,
)
from ds_agents.state import Metric, TaskType

pytestmark = pytest.mark.fast

SRC = Path(__file__).resolve().parents[1] / "src" / "ds_agents"
MANIFEST = load_manifest()


class TestShape:
    def test_manifest_parses_under_extra_forbid(self) -> None:
        raw = yaml.safe_load(benchmark.MANIFEST_PATH.read_text())
        assert DatasetManifest.model_validate(raw) == MANIFEST

    def test_dataset_count_is_inside_the_recorded_target_range(self) -> None:
        low, high = SELECTION_RULE.target_count
        assert low <= len(MANIFEST.datasets) <= high

    def test_dataset_ids_are_unique_snake_case_and_sorted(self) -> None:
        ids = [entry.dataset_id for entry in MANIFEST.datasets]
        assert len(set(ids)) == len(ids)
        assert ids == sorted(ids)
        for dataset_id in ids:
            assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", dataset_id), dataset_id

    def test_openml_data_ids_and_task_ids_are_each_unique_and_both_recorded(self) -> None:
        """Both, always, and never one derived from the other.

        They are different namespaces that happen to coincide for credit-g (data 31, task 31).
        A reader who saw only that entry would learn the wrong rule and a later dataset would
        silently be fetched from the wrong id.
        """
        data_ids = [entry.openml_data_id for entry in MANIFEST.datasets]
        task_ids = [entry.openml_task_id for entry in MANIFEST.datasets]
        assert len(set(data_ids)) == len(data_ids)
        assert len(set(task_ids)) == len(task_ids)
        assert all(isinstance(i, int) and i > 0 for i in data_ids + task_ids)
        differing = [e for e in MANIFEST.datasets if e.openml_data_id != e.openml_task_id]
        assert differing, "if these ever all coincide, the pair has stopped being evidence"

    def test_dataset_ids_do_not_collide_with_fixture_names(self) -> None:
        """`--dataset` has one namespace, so the two registries must stay disjoint."""
        assert set(benchmark.available()) & set(fixtures.available()) == set()

    def test_unknown_dataset_id_raises_listing_the_known_ones(self) -> None:
        with pytest.raises(SystemExit) as caught:
            load_dataset("not_a_dataset")
        message = str(caught.value)
        assert "not_a_dataset" in message
        assert MANIFEST.datasets[0].dataset_id in message


class TestTheSelectionRule:
    def test_the_manifest_selection_block_equals_the_module_constant(self) -> None:
        """The generated file and the constant that generated it cannot drift apart."""
        assert MANIFEST.selection == SELECTION_RULE

    def test_every_entry_satisfies_the_recorded_rule(self) -> None:
        for entry in MANIFEST.datasets:
            assert entry.task_type in SELECTION_RULE.task_types, entry.dataset_id
            assert entry.metric == SELECTION_RULE.metric, entry.dataset_id
            assert entry.n_rows <= SELECTION_RULE.max_rows, entry.dataset_id
            assert entry.n_features <= SELECTION_RULE.max_features, entry.dataset_id
            assert entry.n_rows * entry.n_features <= SELECTION_RULE.max_cells, entry.dataset_id
            assert entry.n_usable_features >= SELECTION_RULE.min_usable_features, entry.dataset_id

    def test_task_type_and_metric_are_values_state_py_can_type(self) -> None:
        """A manifest metric `state.py` cannot express is a dataset that cannot be run."""
        for entry in MANIFEST.datasets:
            assert entry.task_type in get_args(TaskType)
            assert entry.metric in get_args(Metric)

    def test_every_exclusion_records_its_criterion_and_the_measurement(self) -> None:
        """So "why is christine not in here" is answerable from the file, not from a transcript."""
        known = set(SELECTION_RULE.model_dump()) | {"target_count", "task_type"}
        for excluded in MANIFEST.excluded:
            assert excluded.failed, excluded.openml_name
            assert excluded.measured, excluded.openml_name
            if excluded.failed in known or excluded.failed.startswith(("max_", "min_")):
                assert any(isinstance(v, int | float) for v in excluded.measured.values()), (
                    excluded.openml_name
                )

    def test_the_rule_actually_excluded_something_on_every_bound(self) -> None:
        """A bound that never fires is decoration. These three did real work on this set."""
        fired = {excluded.failed for excluded in MANIFEST.excluded}
        assert {"task_type", "max_features", "min_usable_features"} <= fired


class TestProvenanceFields:
    def test_the_source_block_names_the_paper_the_suite_and_a_pinned_commit(self) -> None:
        source = MANIFEST.source
        assert "Gijsbers" in source.citation and "2024" in source.citation
        assert source.paper_url.startswith("https://")
        assert source.openml_suite_id == 271
        # A 40-hex sha, not a tag or a branch: the point of recording it is that the provenance
        # test checks the same bytes next year that it checks today.
        assert re.fullmatch(r"[0-9a-f]{40}", source.amlb_ref)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", source.retrieved)

    def test_every_entry_carries_its_upstream_pins(self) -> None:
        for entry in MANIFEST.datasets:
            assert re.fullmatch(r"[0-9a-f]{32}", entry.md5_checksum), entry.dataset_id
            assert re.fullmatch(r"[0-9a-f]{64}", entry.csv_sha256), entry.dataset_id
            assert entry.openml_url.startswith("https://"), entry.dataset_id
            assert entry.amlb_list in MANIFEST.source.amlb_lists, entry.dataset_id

    def test_every_published_reference_is_citable_and_labelled_with_its_protocol(self) -> None:
        """A number without its protocol is the exact thing this manifest refuses to publish."""
        for entry in MANIFEST.datasets:
            reference = entry.published_reference
            if reference is None:
                continue
            assert reference.openml_run_id > 0, entry.dataset_id
            assert str(reference.openml_run_id) in reference.openml_run_url, entry.dataset_id
            assert 0.0 <= reference.value <= 1.0, entry.dataset_id
            assert "NOT this repo" in reference.protocol, entry.dataset_id

    def test_the_baseline_block_is_a_definition_and_carries_no_number(self) -> None:
        baselines = MANIFEST.baselines
        assert baselines.zero_point and baselines.unit_point
        assert not re.search(r"\d\.\d", baselines.zero_point + baselines.unit_point)


class TestLeakageIsNotClaimed:
    def test_every_entry_is_declared_unlabelled_for_leakage(self) -> None:
        """These are real datasets. Nobody has enumerated their leaks, and the file says so."""
        assert all(entry.leakage_labelled is False for entry in MANIFEST.datasets)

    def test_known_leakage_never_claims_to_be_planted_ground_truth(self) -> None:
        """`planted_leakage_columns` is read as a COMPLETE list.

        Promoting a documented column into it would claim we know every leak in a real dataset,
        and would score every other genuinely-suspicious column the reviewer names as a false
        alarm. `Literal[False]` makes the flip impossible without a schema change.
        """
        for entry in MANIFEST.datasets:
            for leak in entry.known_leakage:
                assert leak.in_planted_leakage_columns is False

    def test_the_documented_leak_survived_into_the_manifest(self) -> None:
        """The project's first non-synthetic, independently documented leak.

        Every leakage finding so far is on a fixture this repo wrote itself, so a reviewer that
        catches our traps may only be catching our habits. This one is somebody else's.
        """
        documented = [e for e in MANIFEST.datasets if e.known_leakage]
        assert documented, "no dataset carries a documented leak any more"
        for entry in documented:
            for leak in entry.known_leakage:
                assert leak.source_url.startswith("https://")
                assert leak.why and leak.source


class TestTheYamlItself:
    def test_a_column_named_no_survives_the_round_trip(self) -> None:
        """YAML 1.1 resolves an unquoted `no` to False and an unquoted date to `datetime.date`.

        A column named `no` is not hypothetical in OpenML data, and a target silently parsed as
        `False` would fail far away from here. The generator quotes every string scalar; this
        checks the parser agrees.
        """
        from ds_agents.benchmark_build import _Dumper

        payload = {"target": "no", "other": "on", "retrieved": "2026-08-31", "flag": True}
        round_tripped = yaml.safe_load(yaml.dump(payload, Dumper=_Dumper))
        assert round_tripped == payload
        assert isinstance(round_tripped["retrieved"], str)

    def test_the_file_says_it_is_generated(self) -> None:
        header = benchmark.MANIFEST_PATH.read_text()[:1200]
        assert "GENERATED FILE" in header
        assert "datasets refresh" in header


class TestBoundaries:
    """Two rules that are otherwise only prose, and whose violation yields a number rather
    than an error."""

    def test_nothing_in_nodes_imports_a_dataset_registry(self) -> None:
        """A node that can read a registry can read the answer key.

        Stated in three docstrings and, until now, enforced by nothing.
        """
        banned = {
            "ds_agents.benchmark",
            "ds_agents.benchmark_build",
            "ds_agents.fixtures",
            "ds_agents.naming",
        }
        for path in sorted((SRC / "nodes").glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name not in banned, f"{path.name} imports {alias.name}"
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module not in banned, f"{path.name} imports {node.module}"
                # A static import is the obvious route and not the only one. `import_module` on a
                # string would sail past the two clauses above while doing exactly the same thing.
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert node.value not in banned, f"{path.name} names {node.value} as a string"

    def test_no_manifest_field_can_reach_baseline_score(self) -> None:
        """The measurement-independence guarantee, as an executable invariant.

        A published score came from OpenML's 10-fold CV run by another flow;
        `verified_holdout_score` will come from this repo's withheld holdout. Wiring one to the
        other would attribute a difference between protocols to a difference between systems --
        and would do it silently, producing a `score_ratio` that looks entirely reasonable.
        """
        for name in ("benchmark.py", "benchmark_build.py"):
            source = (SRC / name).read_text()
            # Checked against the AST, not the text: both modules DISCUSS these fields at length,
            # and they should. What must not exist is code that reads or writes one.
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr not in {
                        "baseline_score",
                        "verified_holdout_score",
                    }, f"{name} touches {node.attr}"
                if isinstance(node, ast.keyword):
                    assert node.arg not in {
                        "baseline_score",
                        "verified_holdout_score",
                    }, f"{name} passes {node.arg}"
                if isinstance(node, ast.Name):
                    assert node.id not in {
                        "baseline_score",
                        "verified_holdout_score",
                        "score_ratio",
                    }, f"{name} binds {node.id}"
                # String-keyed access -- `row["baseline_score"] = x`, or `setattr(s, "...", x)` --
                # reaches the same field without ever appearing as an attribute or a name. The
                # docstrings above DO discuss these fields, so only `ast.Constant` string nodes
                # are checked, never the raw file text.
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert node.value not in {
                        "baseline_score",
                        "verified_holdout_score",
                        "score_ratio",
                    }, f"{name} uses {node.value!r} as a string literal"
