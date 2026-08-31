"""`ds-agents datasets` -- the three verbs, including the paths that return non-zero.

`refresh` is the only writer of `evals/datasets/manifest.yaml` and `verify --online` is the only
thing that checks it, so a silent regression in either is a silent regression in the manifest's
provenance story. Both are mocked here: these tests do no network I/O and never write the real
manifest.
"""

import pytest

from ds_agents import benchmark
from ds_agents.cli import _build_parser, cmd_datasets

pytestmark = pytest.mark.fast


def _args(*argv):
    return _build_parser().parse_args(["datasets", *argv])


class TestList:
    def test_list_prints_every_dataset_with_its_ids(self, capsys) -> None:
        assert cmd_datasets(_args("list")) == 0
        out = capsys.readouterr().out
        manifest = benchmark.load_manifest()
        for entry in manifest.datasets:
            assert entry.dataset_id in out
            assert str(entry.openml_data_id) in out

    def test_list_without_a_manifest_says_how_to_make_one(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(benchmark, "available", lambda: [])
        assert cmd_datasets(_args("list")) == 1
        assert "datasets refresh" in capsys.readouterr().err


class TestRefresh:
    def test_refresh_reports_what_it_wrote(self, monkeypatch, capsys) -> None:
        from ds_agents import benchmark_build

        manifest = benchmark.load_manifest()
        monkeypatch.setattr(benchmark_build, "refresh", lambda *a, **k: manifest)
        assert cmd_datasets(_args("refresh")) == 0
        out = capsys.readouterr().out
        assert str(len(manifest.datasets)) in out
        assert "manifest.yaml" in out

    def test_a_count_outside_the_target_range_warns_rather_than_being_hand_fixed(
        self, monkeypatch, capsys
    ) -> None:
        """The rule decides the count. If it lands short, that is a finding to report, not a gap
        to close by picking an extra dataset -- so the tool says so loudly and still exits 0.
        """
        from ds_agents import benchmark_build

        manifest = benchmark.load_manifest()
        trimmed = manifest.model_copy(update={"datasets": manifest.datasets[:2]})
        monkeypatch.setattr(benchmark_build, "refresh", lambda *a, **k: trimmed)
        assert cmd_datasets(_args("refresh")) == 0
        err = capsys.readouterr().err
        assert "outside the target range" in err
        assert "do not hand-pick" in err


class TestVerify:
    def test_offline_verify_only_checks_the_schema(self, capsys) -> None:
        assert cmd_datasets(_args("verify")) == 0
        assert "offline" in capsys.readouterr().out

    def test_online_verify_passes_when_a_rebuild_matches(self, monkeypatch, capsys) -> None:
        from ds_agents import benchmark_build

        on_disk = benchmark.MANIFEST_PATH.read_text()
        monkeypatch.setattr(benchmark_build, "build", lambda: None)
        monkeypatch.setattr(benchmark_build, "render", lambda _: on_disk)
        assert cmd_datasets(_args("verify", "--online")) == 0
        assert "matches a fresh rebuild" in capsys.readouterr().out

    def test_online_verify_fails_with_a_diff_when_the_file_has_drifted(
        self, monkeypatch, capsys
    ) -> None:
        """The failure that matters: OpenML re-versioned a dataset, or someone edited the file
        past the deny rule. Non-zero exit, and the diff says what moved."""
        from ds_agents import benchmark_build

        monkeypatch.setattr(benchmark_build, "build", lambda: None)
        monkeypatch.setattr(benchmark_build, "render", lambda _: "'schema_version': 999\n")
        assert cmd_datasets(_args("verify", "--online")) == 1
        err = capsys.readouterr().err
        assert "DIFFERS" in err
        assert "999" in err
