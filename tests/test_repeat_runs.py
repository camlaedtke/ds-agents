"""`--repeat` end to end: does the loop actually produce N independent runs?

`cmd_run`'s repeat loop is the mechanism every ablation number now comes out of, and until this
module existed the isolation it claims was asserted in a comment and nowhere else. What matters is
not that it runs N times -- it is that run 2 cannot see run 1. The artifact store is per-run and
mints ids like `art-000`, so a shared root would let one run read another's artifacts by id and the
repetitions would quietly stop being independent samples.

Runs against `StubModel` over `--tools local`: no key, no network, no MCP subprocess. That also
makes this the test for the other half of the contract -- a stub run must be REFUSED a results row,
because a results file that accepted placeholder rows would look exactly like a results file.

Not marked `fast`: each case runs the whole graph, twice.
"""

import argparse
from pathlib import Path

from ds_agents.cli import _build_parser, cmd_run


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    argv = ["run", "--dataset", "toy", "--tools", "local", "--no-live"]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    args = _build_parser().parse_args(argv)
    args.artifacts_dir = str(tmp_path / "artifacts")
    return args


def test_two_repetitions_get_two_isolated_artifact_directories(tmp_path):
    assert cmd_run(_args(tmp_path, repeat=2)) == 0

    roots = sorted(p.name for p in (tmp_path / "artifacts").iterdir() if p.is_dir())
    assert roots == ["input", "run-0", "run-1"] or roots == ["run-0", "run-1"], roots

    # The store copies the dataset into each run's own `data/` and writes artifacts into its own
    # `artifacts/`. Two separate trees is the isolation; one shared tree would not be.
    for name in ("run-0", "run-1"):
        assert (tmp_path / "artifacts" / name / "artifacts").is_dir()


def test_a_single_run_still_uses_the_root_directly(tmp_path):
    """--repeat 1 must be byte-for-byte the old behaviour, or every existing invocation moved."""
    assert cmd_run(_args(tmp_path, repeat=1)) == 0
    assert (tmp_path / "artifacts" / "artifacts").is_dir()
    assert not (tmp_path / "artifacts" / "run-0").exists()


def test_stub_repetitions_are_refused_a_results_row(tmp_path):
    """The gate, exercised through the loop rather than in isolation."""
    results = tmp_path / "results.jsonl"
    args = _args(tmp_path, repeat=2)
    args.results = str(results)

    assert cmd_run(args) == 0
    assert not results.exists(), "a placeholder run was written to a results file"


def test_repeat_below_one_is_refused(tmp_path):
    assert cmd_run(_args(tmp_path, repeat=0)) == 2


def test_an_unknown_dataset_exits_two_before_running_anything(tmp_path):
    args = _args(tmp_path, repeat=2)
    args.dataset = "does_not_exist"
    assert cmd_run(args) == 2
    assert not (tmp_path / "artifacts").exists()


def test_the_opaque_arm_materialises_one_csv_for_all_repetitions(tmp_path):
    """Every run in an arm must see the same bytes. Rewriting the file once per repetition would
    be N chances for them not to, so it happens once per invocation."""
    args = _args(tmp_path, repeat=2)
    args.naming = "opaque"
    args.dataset = "claims_timing"
    assert cmd_run(args) == 0

    materialised = list((tmp_path / "artifacts" / "input").glob("*.csv"))
    assert len(materialised) == 1
    assert materialised[0].read_text().splitlines()[0].startswith("var_01,")
