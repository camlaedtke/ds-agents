"""The Phase 4 eval harness: planning, execution, and the gates around both.

Every test here uses a fake `runner`, never the real pipeline -- that is the whole point of the
`Runner` seam in `harness.py`. A harness that could only be tested by spending money would not get
tested. `PipelineState` objects built for these tests still have to satisfy `publishable()`, or a
"successful" fake run would be refused by `_append_results_row` and every counting test would be
lying about what it counts: see `NodeEvent(node=..., started=utc_now(), model="claude-haiku-4-5")`
in `tests/test_state.py`'s `TestWhyTheLoopDidNotConverge` for the pattern this borrows.
"""

import json
from collections import Counter
from dataclasses import fields
from datetime import date

import pytest

from ds_agents.fixtures import available
from ds_agents.harness import SUBSETS, Cell, HarnessReport, PlannedRun, plan, run_eval
from ds_agents.state import NodeEvent, PipelineState, utc_now

pytestmark = pytest.mark.fast


def _state(*, cost: float = 0.01, model: str = "claude-haiku-4-5") -> PipelineState:
    """A minimal `PipelineState` that passes `publishable()`.

    One `NodeEvent` with a real-looking model name and a cost is all `publishable()` checks for
    (a non-empty trace with no placeholder model). Everything else on `PipelineState` is free to
    stay default, because these tests are about the harness's bookkeeping, not the pipeline.
    """
    return PipelineState(
        dataset_id="toy",
        task_description="Predict something.",
        node_trace=[NodeEvent(node="intake", started=utc_now(), model=model, cost_usd=cost)],
    )


class TestThePlanIsReplicateMajor:
    """A cost cap binds on the plan's execution order, so the order decides who gets starved.

    Cell-major ordering is what put the Sonnet reviewer cells at n=4 and n=3 against a budgeted
    n=8 in the 2026-08-28 reviewer-ablation run: the cap bound partway through the first cell's
    replicates, the last cell never started, and the resulting n=4-vs-n=3 pairing was unquotable as
    a 2x2. Replicate-major ordering -- every cell's replicate 1 before any cell's replicate 2 --
    means a cap that binds partway through drops the SAME fraction of runs from every cell.
    """

    def test_the_seq_order_is_replicate_major_not_cell_major(self):
        a = Cell(name="a", dataset="toy")
        b = Cell(name="b", dataset="toy")

        planned = plan([a, b], replicates=2, n=2)

        assert [(r.cell.name, r.replicate, r.index) for r in planned] == [
            ("a", 1, 0),
            ("a", 1, 1),
            ("b", 1, 0),
            ("b", 1, 1),
            ("a", 2, 0),
            ("a", 2, 1),
            ("b", 2, 0),
            ("b", 2, 1),
        ]
        assert [r.seq for r in planned] == list(range(8))

    def test_a_cap_partway_through_truncates_every_cell_roughly_equally(self):
        """The regression case: two cells, four replicates each, a cap that binds after 3 runs."""
        a = Cell(name="a", dataset="toy")
        b = Cell(name="b", dataset="toy")

        planned = plan([a, b], replicates=4, n=1)
        surviving = planned[:3]

        counts = Counter(r.cell.name for r in surviving)
        # Neither cell was starved to zero and neither ran to its full n=4 -- a cell-major plan
        # would have given "a" all 3 survivors and left "b" at zero.
        assert 0 < counts["a"] < 4
        assert 0 < counts["b"] < 4

    def test_every_planned_run_carries_its_cell_and_replicate(self):
        cell = Cell(name="a", dataset="toy")

        planned = plan([cell], replicates=2, n=3)

        for run in planned:
            assert run.cell is cell
            assert run.replicate in (1, 2)


class TestSubsetsPointAtRealFixtures:
    """A typo'd dataset name in a `Cell` must fail here, in a fast test, not $0.20 into a live
    run."""

    def test_every_cell_in_every_subset_names_a_registered_fixture(self):
        known = set(available())

        for subset_name, cells in SUBSETS.items():
            for cell in cells:
                assert cell.dataset in known, (
                    f"SUBSETS[{subset_name!r}] cell {cell.name!r} names unregistered fixture "
                    f"{cell.dataset!r}"
                )

    def test_the_ci_subset_covers_all_three_fixtures(self):
        assert {cell.dataset for cell in SUBSETS["ci"]} == set(available())

    def test_the_toy_subset_is_just_toy_default(self):
        assert [cell.name for cell in SUBSETS["toy"]] == ["toy-default"]

    def test_ci_cells_share_loop_cap_three_and_closure_off(self):
        """The spec's cell definitions, pinned so a future edit notices it changed one.

        The three `est_cost_usd` values are measured means from the 2026-08-31 `ci` baseline, not
        guesses -- see the comment above `SUBSETS`. They are pinned here so that a future revision
        has to be deliberate, but note what they are NOT: `est_cost_usd` never reaches a results
        row, so no published number moves when these do. Only the point at which a cost cap
        truncates a plan moves.
        """
        for cell in SUBSETS["ci"]:
            assert cell.loop_cap == 3
            assert cell.objection_closure == "off"

        by_name = {cell.name: cell for cell in SUBSETS["ci"]}
        assert by_name["toy-default"].est_cost_usd == pytest.approx(0.015)
        claims = by_name["claims-opaque-which"]
        assert claims.dataset == "claims_timing"
        assert claims.naming == "opaque"
        assert claims.reviewer_prompt == "which_column"
        assert claims.objection_routing == "by_category"
        assert claims.est_cost_usd == pytest.approx(0.030)
        reissued = by_name["reissued-opaque-which"]
        assert reissued.dataset == "reissued_ids"
        assert reissued.naming == "opaque"
        assert reissued.reviewer_prompt == "which_column"
        assert reissued.objection_routing == "by_category"
        assert reissued.est_cost_usd == pytest.approx(0.029)


class TestForcedDropReleaseIsAClosedAxis:
    """`forced_drop_release` had exactly one legitimate use and it has been spent (see
    docs/NEXT.md's parking lot). No `Cell` may express it, so no subset can either."""

    def test_cell_has_no_forced_drop_release_field(self):
        names = {f.name for f in fields(Cell)}
        assert "forced_drop_release" not in names


class TestTheFullSubsetIsNotYetRunnable:
    """`full` no longer waits on the manifest -- it waits on the re-scorer.

    The manifest landed 2026-08-31 and the session-0 question behind it is answered, so the old
    message (which named the missing file and the open OpenML question) is now false on both
    counts. What blocks `full` is that nothing withholds a holdout or computes
    `verified_holdout_score` / `baseline_score`, so every row would carry a null `score_ratio`.
    The assertion below deliberately checks the message names the SCORE fields: if someone wires
    up `Cell` for manifest datasets without the re-scorer, this is what should still fail.
    """

    def test_full_raises_pointing_at_the_missing_re_scorer(self):
        with pytest.raises(ValueError) as excinfo:
            run_eval(subset="full", name="probe", dry_run=True)

        message = str(excinfo.value)
        assert "manifest.yaml" in message
        assert "verified_holdout_score" in message
        assert "baseline_score" in message
        assert "toy" in message
        assert "ci" in message

    def test_an_unknown_subset_also_lists_what_is_available(self):
        with pytest.raises(ValueError) as excinfo:
            run_eval(subset="not-a-real-subset", name="probe", dry_run=True)

        message = str(excinfo.value)
        assert "toy" in message
        assert "ci" in message


class TestFailureHandling:
    """A run that raises is a data point about that run, not a reason to stop the cell -- the
    parking-lot fix for `cmd_run --repeat`'s missing per-run try/except."""

    def test_a_failed_run_does_not_end_the_cell(self, tmp_path):
        outcomes = [_state(), RuntimeError("boom"), _state(), _state()]
        calls: list[PlannedRun] = []

        def runner(planned_run: PlannedRun, root):
            calls.append(planned_run)
            outcome = outcomes[planned_run.index]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        report = run_eval(
            subset="toy",
            name="probe",
            n=4,
            out_dir=tmp_path,
            runner=runner,
            today=date(2026, 8, 29),
        )

        assert len(calls) == 4, "the failure at run 2 must not have stopped runs 3 and 4"
        assert report.rows_written == 3
        assert report.runs_failed == 1
        assert report.per_cell["toy-default"].runs_failed == 1

    def test_three_consecutive_failures_abort_the_invocation(self, tmp_path):
        calls: list[PlannedRun] = []

        def runner(planned_run: PlannedRun, root):
            calls.append(planned_run)
            raise RuntimeError("boom")

        report = run_eval(
            subset="toy",
            name="probe",
            n=10,
            out_dir=tmp_path,
            runner=runner,
            today=date(2026, 8, 29),
        )

        assert len(calls) == 3, "a broken config must not burn through all 10 runs one at a time"
        assert report.runs_failed == 3
        assert report.stopped_early == "consecutive failures"

    def test_a_systemexit_propagates_instead_of_counting_as_a_failure(self, tmp_path):
        """`cli._run_once` raises `SystemExit` to mean the run never started (unknown fixture, the
        tool server would not start). That will not start for run 2 either, so it is not a
        per-run failure to catch and continue past -- it ends the invocation."""

        def runner(planned_run: PlannedRun, root):
            raise SystemExit("could not start the tool server")

        with pytest.raises(SystemExit):
            run_eval(
                subset="toy",
                name="probe",
                n=3,
                out_dir=tmp_path,
                runner=runner,
                today=date(2026, 8, 29),
            )


class TestTheCostCap:
    """Invocation-level, checked before each run against the plan's estimate; spend is tracked in
    actual dollars so the report is never quoting a number nobody measured."""

    def test_the_cap_stops_before_the_run_that_would_exceed_it(self, tmp_path):
        def runner(planned_run: PlannedRun, root):
            return _state(cost=0.005)

        report = run_eval(
            subset="ci",
            name="probe",
            n=1,
            max_cost_usd=0.02,
            out_dir=tmp_path,
            runner=runner,
            today=date(2026, 8, 29),
        )

        # toy-default (est 0.015) fits; toy-default's actual spend (0.005) + claims-opaque-which's
        # estimate (0.030) does not fit under a 0.02 cap, so the cap stops there.
        assert report.rows_written == 1
        assert report.stopped_early == "cost cap"
        assert report.per_cell["toy-default"].rows_written == 1
        assert report.per_cell["claims-opaque-which"].rows_written == 0
        assert report.per_cell["reissued-opaque-which"].rows_written == 0

    def test_spend_is_reported_off_actual_cost_not_the_estimate(self, tmp_path):
        report = run_eval(
            subset="toy",
            name="probe",
            n=1,
            out_dir=tmp_path,
            runner=lambda planned_run, root: _state(cost=0.005),
            today=date(2026, 8, 29),
        )

        # toy-default's est_cost_usd is 0.015; the fake run actually cost 0.005.
        assert report.spend_usd == pytest.approx(0.005)
        assert report.per_cell["toy-default"].spend_usd == pytest.approx(0.005)
        assert report.charged_estimate_usd == 0.0

    def test_a_failed_run_is_charged_its_estimate_and_the_guess_is_reported_separately(
        self, tmp_path
    ):
        """A run that raises took its partial cost with it -- the tokens were spent, but the state
        that counted them never came back. Charging nothing would let a cell that fails late, after
        paying for most of a pipeline, walk straight through the cap; charging the estimate errs in
        the safe direction. It is reported separately because a spend figure that silently mixes
        measured and guessed dollars is exactly the kind of number this project refuses to publish.
        """

        def runner(planned_run: PlannedRun, root):
            raise RuntimeError("the API fell over")

        report = run_eval(
            subset="toy",
            name="probe",
            n=1,
            out_dir=tmp_path,
            runner=runner,
            today=date(2026, 8, 29),
        )

        assert report.runs_failed == 1
        assert report.rows_written == 0
        # toy-default's est_cost_usd, charged in full because nothing measured what it really cost.
        assert report.spend_usd == pytest.approx(0.015)
        assert report.charged_estimate_usd == pytest.approx(0.015)


class TestPublishabilityGate:
    """The same gate `cli._append_results_row` applies to `--results`, so a stub run cannot reach
    a Phase 4 results file any more than it can reach a `--results` file."""

    def test_a_stub_row_is_refused_and_counted(self, tmp_path):
        report = run_eval(
            subset="toy",
            name="probe",
            n=2,
            out_dir=tmp_path,
            runner=lambda planned_run, root: _state(model="stub"),
            today=date(2026, 8, 29),
        )

        assert report.rows_written == 0
        assert report.rows_refused == 2
        assert report.per_cell["toy-default"].rows_refused == 2
        assert report.path is not None
        assert not report.path.exists(), "a refused row must never create the results file"

    def test_a_written_row_carries_cell_replicate_run_index_and_commit(self, tmp_path):
        report = run_eval(
            subset="toy",
            name="probe",
            n=1,
            out_dir=tmp_path,
            runner=lambda planned_run, root: _state(),
            today=date(2026, 8, 29),
        )

        assert report.path is not None
        lines = report.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["cell"] == "toy-default"
        assert row["replicate"] == 1
        assert row["run_index"] == 0
        assert row["eval_subset"] == "toy"
        assert row["eval_name"] == "probe"
        assert "commit" in row


class TestOutputFileNaming:
    def test_rows_land_in_a_dated_file_named_for_the_run(self, tmp_path):
        report = run_eval(
            subset="toy",
            name="smoke-test",
            n=1,
            out_dir=tmp_path,
            runner=lambda planned_run, root: _state(),
            today=date(2026, 3, 4),
        )

        assert report.path == tmp_path / "2026-03-04_smoke-test.jsonl"
        assert report.path is not None
        assert report.path.exists()

    def test_a_second_invocation_appends_rather_than_truncates(self, tmp_path, capsys):
        def _one_run() -> None:
            run_eval(
                subset="toy",
                name="probe",
                n=1,
                out_dir=tmp_path,
                runner=lambda planned_run, root: _state(),
                today=date(2026, 8, 29),
            )

        _one_run()
        capsys.readouterr()  # drop the first invocation's stderr
        _one_run()
        captured = capsys.readouterr()

        lines = (tmp_path / "2026-08-29_probe.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert "2026-08-29_probe.jsonl" in captured.err, (
            "the second invocation must note on stderr that the file already existed"
        )

    def test_a_name_with_a_path_separator_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            run_eval(subset="toy", name="probe/evil", out_dir=tmp_path, dry_run=True)

    def test_a_name_with_a_capital_letter_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            run_eval(subset="toy", name="Probe", out_dir=tmp_path, dry_run=True)


class TestDryRun:
    def test_dry_run_writes_no_file_and_calls_no_runner(self, tmp_path):
        calls: list[PlannedRun] = []

        def runner(planned_run: PlannedRun, root):
            calls.append(planned_run)
            return _state()

        report = run_eval(
            subset="ci",
            name="probe",
            n=2,
            replicates=2,
            out_dir=tmp_path,
            runner=runner,
            dry_run=True,
            today=date(2026, 8, 29),
        )

        assert calls == []
        assert report.path is None
        assert report.rows_written == 0
        assert list(tmp_path.iterdir()) == []

    def test_dry_run_still_prints_the_plan_and_the_cost_estimate(self, tmp_path, capsys):
        run_eval(subset="ci", name="probe", n=1, out_dir=tmp_path, dry_run=True)

        captured = capsys.readouterr()
        assert "toy-default" in captured.err
        assert "claims-opaque-which" in captured.err
        assert "reissued-opaque-which" in captured.err
        assert "$" in captured.err


def test_a_harness_report_has_a_tally_for_every_cell_in_the_subset(tmp_path):
    """Sanity check on `per_cell`'s shape: every cell gets a `CellTally`, even one that never ran
    because the cap or a run of failures stopped the invocation first."""
    report = run_eval(subset="ci", name="probe", n=1, out_dir=tmp_path, dry_run=True)

    assert set(report.per_cell) == {cell.name for cell in SUBSETS["ci"]}


def test_planned_run_and_harness_report_are_the_documented_shape():
    """Cheap contract pins so a field rename here is caught by a one-line test, not a caller."""
    run = PlannedRun(cell=Cell(name="a", dataset="toy"), replicate=1, index=0, seq=0)
    assert (run.cell.name, run.replicate, run.index, run.seq) == ("a", 1, 0, 0)

    report = HarnessReport(
        path=None,
        rows_written=0,
        rows_refused=0,
        runs_failed=0,
        spend_usd=0.0,
        charged_estimate_usd=0.0,
        stopped_early=None,
        per_cell={},
    )
    assert report.stopped_early is None


class TestTheLiveRunnerMaterializesOncePerDatasetAndNaming:
    """`_live_run` is the one part of the harness a fake `runner` never exercises, so a bug in it
    would only ever surface on a paid run.

    What it must guarantee is the rule `cmd_run --repeat` already follows: every run in an arm sees
    the same bytes. Rewriting the CSV once per run is N chances for them not to, and under `--naming
    opaque` the file is a rewritten copy rather than the committed fixture, so "the same bytes" is
    not automatic.
    """

    def test_the_same_dataset_and_naming_is_materialized_once_across_runs(
        self, tmp_path, monkeypatch
    ):
        from ds_agents import cli, harness
        from ds_agents.naming import materialize as real_materialize

        calls: list[tuple[str, str]] = []

        def counting_materialize(fixture, naming, dest):
            calls.append((fixture.name, naming))
            return real_materialize(fixture, naming, dest)

        monkeypatch.setattr("ds_agents.naming.materialize", counting_materialize)
        monkeypatch.setattr(cli, "_run_once", lambda fixture, **kwargs: _state(cost=0.001))

        runner = harness._live_run(tmp_path, transport="local", no_live=True)
        cells = (
            Cell(name="a", dataset="toy"),
            Cell(name="b", dataset="toy"),
            Cell(name="c", dataset="toy", naming="opaque"),
        )
        for seq, cell in enumerate(cells + cells):
            runner(PlannedRun(cell=cell, replicate=1, index=0, seq=seq), tmp_path / f"r{seq}")

        # Six runs over two distinct (dataset, naming) pairs: descriptive and opaque, once each.
        assert calls == [("toy", "descriptive"), ("toy", "opaque")]


class TestProvenanceIsReadOncePerInvocation:
    """The harness must not record its own output as a change to the tree that produced its runs.

    Found by the 2026-08-31 `ci` baseline, not by a test: `_run_once` used to call `git_commit()`
    per run, and the results JSONL is untracked until someone commits it, so writing row 0 made
    `git status --porcelain` non-empty. Run 0 recorded `8a629bf` and runs 1..29 recorded
    `8a629bf-dirty`. Because `commit` is one of `evaldiff.CONDITION_FIELDS`, that split
    `toy-default` into a cell of n=1 and a cell of n=9 -- the field meant to guarantee that pooled
    rows came from one tree instead guaranteed that they could not be pooled at all.

    This is the same once-per-invocation rule `materialize` already follows above, for the same
    reason: everything that is supposed to be identical across the runs of one invocation has to be
    read once, before the runs start changing the thing being read.
    """

    def test_every_run_in_one_invocation_gets_the_same_commit(self, tmp_path, monkeypatch):
        from ds_agents import cli, harness

        # A git_commit that answers differently every call, standing in for a tree the invocation
        # dirties as it goes. If provenance were read per run, these would reach the rows verbatim.
        answers = iter(["cafe1", "cafe1-dirty", "cafe1-dirty", "cafe1-dirty"])
        monkeypatch.setattr("ds_agents.provenance.git_commit", lambda *a, **k: next(answers))

        seen: list[str | None] = []

        def capturing_run_once(fixture, **kwargs):
            seen.append(kwargs["commit"])
            return _state(cost=0.001)

        monkeypatch.setattr(cli, "_run_once", capturing_run_once)

        runner = harness._live_run(tmp_path, transport="local", no_live=True)
        cell = Cell(name="a", dataset="toy")
        for seq in range(3):
            runner(PlannedRun(cell=cell, replicate=1, index=seq, seq=seq), tmp_path / f"r{seq}")

        assert seen == ["cafe1", "cafe1", "cafe1"]

    def test_run_once_has_no_default_commit_to_fall_back_to(self):
        """The bug was a default, not a call site, so the guard is on the default.

        `commit` is keyword-only with no default and `None` is a legitimate value (git missing, not
        a checkout), so a sentinel default would be indistinguishable from the answer it is
        standing in for. Giving this parameter any default at all would let a third caller
        reintroduce per-run provenance silently, which is how the first one did it.
        """
        import inspect

        from ds_agents import cli

        param = inspect.signature(cli._run_once).parameters["commit"]
        assert param.default is inspect.Parameter.empty
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
