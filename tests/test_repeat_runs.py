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


def test_a_single_run_gets_its_own_directory_like_every_other_run(tmp_path):
    """Changed deliberately with the re-scorer: `--repeat 1` used to collapse the run root into
    the invocation root, so the layout had two shapes.

    That was tolerable while the invocation root held only `input/`. It stopped being tolerable
    when the carve put `withheld/` beside it and the grader put `rescore/` under the run root:
    "the withheld rows are outside every run root" is a property a test has to be able to check,
    and it cannot check it against a layout that is sometimes one thing and sometimes another.
    Nothing that reads a results row is affected -- no path reaches a row.
    """
    assert cmd_run(_args(tmp_path, repeat=1)) == 0
    assert (tmp_path / "artifacts" / "run-0" / "artifacts").is_dir()
    # No `input/`: under `descriptive` a fixture is mounted from its committed path and
    # nothing is written, which is `materialize`'s control-arm-must-not-drift rule.


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


def test_every_repetition_records_the_same_commit(tmp_path, monkeypatch):
    """`--repeat` reads provenance once, before run 0, not once per run.

    The sibling of `TestProvenanceIsReadOncePerInvocation` in `tests/test_harness.py`, and it
    guards the OTHER caller. `--results` writes an untracked file, so a per-run `git_commit()` here
    records run 0 at `<hash>` and every later run at `<hash>-dirty` -- which is what happened to the
    harness on 2026-08-31 and split one cell in two, `commit` being an `eval-diff` condition field.
    The signature guard in `test_harness.py` stops a caller omitting `commit`; only this stops
    `cmd_run` computing a fresh one inside its own loop.

    A `git_commit` that answers differently on every call is the whole mechanism: if the loop asks
    twice, the two runs disagree, and the assertion below is the only thing that can tell.

    Patched at `ds_agents.cli.git_commit`, not at `ds_agents.provenance.git_commit`, and the
    difference is not incidental: `cli` imports the name at module scope, so it is already bound by
    the time a test runs, while `harness._live_run` imports it inside the function (to break the
    `cli` <-> `harness` import cycle) and therefore does pick up a patch on the source module. The
    two sibling tests patch two different names for that reason.
    """
    answers = iter(["feed1", "feed1-dirty", "feed1-dirty"])
    monkeypatch.setattr("ds_agents.cli.git_commit", lambda *a, **k: next(answers))

    seen: list[str | None] = []
    real_run_state = __import__("ds_agents.cli", fromlist=["_run_state"])._run_state

    def capturing_fixture_state(runnable, **kwargs):
        seen.append(kwargs["commit"])
        return real_run_state(runnable, **kwargs)

    monkeypatch.setattr("ds_agents.cli._run_state", capturing_fixture_state)

    assert cmd_run(_args(tmp_path, repeat=2)) == 0
    assert seen == ["feed1", "feed1"]


def test_a_failed_provenance_read_warns_once_across_every_repetition(tmp_path, monkeypatch):
    """`cmd_run` inherits `git_commit`'s new default of `warn=True`, and once is once.

    This call site was silent before 2026-09-21 and is now the reason `cmd_run` says anything at
    all about a missing commit. The once-per-process guard lives in `provenance`, so the thing
    worth asserting here is that the guard actually covers this caller: reading provenance once per
    invocation and warning once per invocation have to be the same "once", or a `--repeat 10` run
    prints ten identical lines about a single read.
    """
    from ds_agents import provenance

    monkeypatch.setattr(provenance, "_warned", False)
    monkeypatch.setattr(
        "ds_agents.provenance.describe_commit", lambda *a, **k: (None, "git not found")
    )

    said: list[str] = []
    monkeypatch.setattr("ds_agents.provenance.warn_to_stderr", lambda message: said.append(message))

    assert cmd_run(_args(tmp_path, repeat=3)) == 0
    assert len(said) == 1
    assert "no commit recorded" in said[0]
    assert "git not found" in said[0]
