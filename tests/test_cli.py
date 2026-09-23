"""CLI parsing, model selection, and results-writing: argv to a runnable RunConfig and back."""

import argparse
import json

import pytest

from ds_agents.cli import (
    _append_results_row,
    _build_parser,
    _run_state,
    _select_model,
    _select_reviewer_model,
    cmd_eval,
    cmd_eval_diff,
    cmd_run,
)
from ds_agents.fixtures import load_fixture
from ds_agents.naming import NAMINGS, header_of, materialize
from ds_agents.state import NodeEvent, PipelineState, RunConfig, utc_now
from ds_agents.tools.llm import AnthropicModel, StubModel
from tests.conftest import runnable

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch):
    """Every test starts with no key in the environment, regardless of the calling shell."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def test_no_live_returns_the_stub_even_with_a_key_present(monkeypatch):
    # --no-live must win over a present key, or a run could not be forced offline.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    model = _select_model(RunConfig(default_model="haiku"), no_live=True)

    assert isinstance(model, StubModel)


def test_no_key_present_falls_back_to_the_stub():
    model = _select_model(RunConfig(default_model="haiku"))

    assert isinstance(model, StubModel)


def test_a_key_present_builds_a_real_client_with_the_resolved_name(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    model = _select_model(RunConfig(default_model="haiku"))

    assert isinstance(model, AnthropicModel)
    assert model.name == "claude-haiku-4-5"


def test_an_unpriced_model_raises_systemexit_rather_than_running_unbilled(monkeypatch):
    # UnknownModelError is fatal: a run that reports $0.00 for an unpriced model is worse than none.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    with pytest.raises(SystemExit):
        _select_model(RunConfig(default_model="not-a-real-model"))


class TestGroundTruthFollowsTheRename:
    """Under `--naming opaque`, planted_leakage_columns must name what the agents actually see."""

    def test_descriptive_keeps_the_manifest_names(self):
        fixture = load_fixture("claims_timing")
        state = _run_state(runnable("claims_timing"))
        assert state.planted_leakage_columns == fixture.manifest.planted_columns
        assert state.config.naming == "descriptive"

    def test_opaque_carries_the_renamed_columns(self, tmp_path):
        fixture = load_fixture("claims_timing")
        _, rename = materialize(runnable("claims_timing"), "opaque", tmp_path)
        state = _run_state(runnable("claims_timing"), naming="opaque")

        assert state.config.naming == "opaque"
        assert state.planted_leakage_columns == [
            rename[c] for c in fixture.manifest.planted_columns
        ]
        assert all(c.startswith("var_") for c in state.planted_leakage_columns)

    def test_the_planted_columns_exist_in_the_data_the_agents_see(self, tmp_path):
        """The whole point of the previous test, stated against the file rather than the map."""
        path, _ = materialize(runnable("claims_timing"), "opaque", tmp_path)
        state = _run_state(runnable("claims_timing"), naming="opaque")
        assert set(state.planted_leakage_columns) <= set(header_of(path))

    def test_the_task_description_is_unaffected(self):
        """The rename touches columns, not the target, which is never renamed."""
        assert (
            _run_state(runnable("claims_timing"), naming="opaque").task_description
            == _run_state(runnable("claims_timing")).task_description
        )

    def test_the_arm_cannot_be_claimed_without_the_rename_being_applied(self):
        """The rename map is derived from `naming`, not passed alongside it, so the two can't
        drift."""
        fixture = load_fixture("claims_timing")
        for naming in NAMINGS:
            state = _run_state(runnable("claims_timing"), naming=naming)
            renamed = state.planted_leakage_columns != fixture.manifest.planted_columns
            assert renamed == (naming == "opaque")


class TestTheRunParser:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = _build_parser()
        return parser.parse_args(argv)

    def test_the_defaults_are_the_old_behaviour(self):
        args = self._parse(["run"])
        assert args.naming == "descriptive"
        assert args.repeat == 1
        assert args.results is None

    def test_the_ablation_flags_parse_together(self):
        args = self._parse(
            [
                "run",
                "--dataset",
                "claims_timing",
                "--naming",
                "opaque",
                "--repeat",
                "10",
                "--results",
                "evals/results/x.jsonl",
            ]
        )
        assert (args.naming, args.repeat, args.results) == (
            "opaque",
            10,
            "evals/results/x.jsonl",
        )

    def test_a_zero_cap_is_valid_and_not_confused_with_a_negative_one(self):
        """0 is the reviewer-off condition expressed as a cap; only a negative cap is refused."""
        assert self._parse(["run", "--loop-cap", "0"]).loop_cap == 0

    def test_a_negative_cap_exits_two_rather_than_raising(self):
        """RunConfig's `ge=0` would raise a Pydantic traceback; cmd_run converts it to exit 2."""
        args = self._parse(["run", "--loop-cap", "-1"])
        assert cmd_run(args) == 2


# (flag, field, default, variant): every run condition round-trips the same way, from argv through
# the parser onto RunConfig and into results_row(), or rows run under different conditions would be
# averaged together as if they were one arm. forced_drop_release's default is the one exception
# to "default reproduces old behaviour": it is the fixed rule, not the pre-existing one.
RUN_CONDITIONS = [
    ("--loop-cap", "loop_cap", 3, 5),
    ("--reviewer-prompt", "reviewer_prompt", "base", "which_column"),
    ("--objection-routing", "objection_routing", "as_addressed", "by_category"),
    ("--objection-closure", "objection_closure", "off", "on"),
    ("--forced-drop-release", "forced_drop_release", "withdrawn_only", "resolved_or_withdrawn"),
]

# (flag, bogus value): refused by the parser before a run can start.
BOGUS_CONDITIONS = [
    ("--naming", "scrambled"),
    ("--reviewer-prompt", "helpful_hints"),
    ("--objection-routing", "by_vibes"),
    ("--objection-closure", "sometimes"),
    ("--forced-drop-release", "eventually"),
]


class TestRunConditionsRoundTrip:
    """Each RunConfig condition: parser default, variant recorded on state and row, bogus
    refused."""

    @pytest.mark.parametrize(("flag", "field", "default", "variant"), RUN_CONDITIONS)
    def test_the_default_matches_the_parser_and_state(self, flag, field, default, variant):
        args = _build_parser().parse_args(["run"])
        assert getattr(args, field) == default
        assert getattr(_run_state(runnable("claims_timing")).config, field) == default

    @pytest.mark.parametrize(("flag", "field", "default", "variant"), RUN_CONDITIONS)
    def test_the_variant_parses_and_is_recorded(self, flag, field, default, variant):
        args = _build_parser().parse_args(["run", flag, str(variant)])
        assert getattr(args, field) == variant
        state = _run_state(runnable("claims_timing"), **{field: variant})
        assert getattr(state.config, field) == variant
        assert state.results_row()[field] == variant

    @pytest.mark.parametrize(("flag", "bogus"), BOGUS_CONDITIONS)
    def test_a_bogus_value_is_refused_by_the_parser(self, flag, bogus):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["run", flag, bogus])


class TestConditionsAreOrthogonal:
    """Each axis must be settable independently, or its effect can never be separated from
    another's."""

    def test_closure_is_orthogonal_to_the_prompt_condition(self):
        state = _run_state(
            runnable("claims_timing"), reviewer_prompt="base", objection_closure="on"
        )
        assert state.config.reviewer_prompt == "base"
        assert state.config.objection_closure == "on"

    def test_forced_drop_release_is_orthogonal_to_routing_and_closure(self):
        """The benchmark cell crosses this with by_category; a coupling here would confound it."""
        state = _run_state(
            runnable("claims_timing"),
            objection_routing="by_category",
            objection_closure="off",
            forced_drop_release="resolved_or_withdrawn",
        )
        assert state.config.objection_routing == "by_category"
        assert state.config.objection_closure == "off"
        assert state.config.forced_drop_release == "resolved_or_withdrawn"


class TestTheReviewerClient:
    """`_select_reviewer_model` picks the reviewer's client, reusing the base one when unchanged."""

    def test_no_live_returns_the_base_unchanged(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        base = StubModel()
        config = RunConfig(default_model="haiku", reviewer_model="sonnet")
        assert _select_reviewer_model(config, base, no_live=True) is base

    def test_a_stub_base_returns_itself_even_with_a_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        base = StubModel()
        config = RunConfig(default_model="haiku", reviewer_model="sonnet")
        assert _select_reviewer_model(config, base) is base

    def test_the_same_model_id_reuses_the_base_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        base = AnthropicModel(model="haiku")
        config = RunConfig(default_model="haiku", reviewer_model="haiku")
        assert _select_reviewer_model(config, base) is base

    def test_a_differing_reviewer_model_builds_a_second_client(self, monkeypatch):
        """A different reviewer_model builds a distinct client while the base keeps its own."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        base = AnthropicModel(model="haiku")
        config = RunConfig(default_model="haiku", reviewer_model="sonnet")
        reviewer = _select_reviewer_model(config, base)
        assert reviewer is not base
        assert isinstance(reviewer, AnthropicModel)
        assert reviewer.name == "claude-sonnet-5"
        assert base.name == "claude-haiku-4-5"

    def test_an_unpriced_reviewer_model_is_fatal_not_free(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        base = AnthropicModel(model="haiku")
        config = RunConfig(default_model="haiku", reviewer_model="not-a-real-model")
        with pytest.raises(SystemExit):
            _select_reviewer_model(config, base)


class TestTheResultsWriter:
    def test_a_stub_run_is_refused_not_written(self, tmp_path, capsys):
        """A stub run must be refused, not written, or it would look like a real results row."""
        state = PipelineState(dataset_id="toy", task_description="x")
        path = tmp_path / "results.jsonl"

        _append_results_row(state, path)

        assert not path.exists()
        assert "REFUSED" in capsys.readouterr().err

    def test_a_publishable_run_appends_one_line_per_call(self, tmp_path):
        state = PipelineState(
            dataset_id="toy",
            task_description="x",
            node_trace=[
                NodeEvent(node="intake", model="claude-haiku-4-5-20251001", started=utc_now())
            ],
        )
        assert state.publishable()[0] is True
        path = tmp_path / "nested" / "results.jsonl"

        _append_results_row(state, path)
        _append_results_row(state, path)

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["dataset_id"] == "toy"
        assert rows[0]["naming"] == "descriptive"


class TestTheEvalCommands:
    """`cmd_eval` and `cmd_eval_diff`: the argv wiring, exit codes, and defaults for a benchmark
    run."""

    def parse(self, *argv):
        return _build_parser().parse_args(argv)

    def test_the_eval_parser_carries_the_sampling_design_and_the_cost_cap(self):
        args = self.parse(
            "eval",
            "--subset",
            "ci",
            "--name",
            "a-cell",
            "--replicates",
            "2",
            "--n",
            "5",
            "--max-cost-usd",
            "0.25",
        )

        assert (args.subset, args.name, args.replicates, args.n) == ("ci", "a-cell", 2, 5)
        assert args.max_cost_usd == 0.25
        assert args.dry_run is False

    def test_eval_requires_a_name(self):
        """The name is what the results file is called; a run cannot be nameless."""
        with pytest.raises(SystemExit):
            self.parse("eval", "--subset", "toy")

    @pytest.mark.parametrize(("flag", "value"), [("--replicates", "0"), ("--n", "0")])
    def test_a_sampling_flag_below_one_is_refused_with_exit_2(self, flag, value, capsys):
        """Zero would silently produce an empty plan instead of a clear error."""
        args = self.parse("eval", "--subset", "toy", "--name", "x", flag, value)

        assert cmd_eval(args) == 2
        assert "must be at least 1" in capsys.readouterr().err

    def test_an_unknown_subset_exits_2_and_lists_what_would_have_worked(self, capsys):
        """A typo is a clean exit 2 naming the real subsets, not a traceback."""
        args = self.parse("eval", "--subset", "fulll", "--name", "x")

        assert cmd_eval(args) == 2
        err = capsys.readouterr().err
        assert "unknown eval subset" in err
        assert "full" in err and "bench-tall" in err

    def test_the_subset_help_names_every_runnable_subset(self):
        """A flag whose help omits a value is how that subset goes unused."""
        from ds_agents.harness import SUBSETS

        eval_parser = _build_parser()._subparsers._group_actions[0].choices["eval"]
        action = next(
            a for a in eval_parser._actions if "--subset" in getattr(a, "option_strings", [])
        )
        for subset in SUBSETS:
            assert subset in action.help, f"--subset help does not mention {subset!r}"
        assert "not implemented" not in action.help

    def test_a_dry_run_exits_0_without_writing(self, tmp_path):
        """A dry run writes no file and that is not an error."""
        args = self.parse(
            "eval", "--subset", "toy", "--name", "x", "--dry-run", "--out-dir", str(tmp_path)
        )

        assert cmd_eval(args) == 0
        assert list(tmp_path.iterdir()) == []

    def test_eval_diff_on_a_missing_file_exits_2(self, tmp_path, capsys):
        args = self.parse("eval-diff", str(tmp_path / "nope.jsonl"), str(tmp_path / "also.jsonl"))

        assert cmd_eval_diff(args) == 2
        assert "no such results file" in capsys.readouterr().err

    def test_eval_diff_compares_two_real_files_and_refuses_to_call_it_an_effect(
        self, tmp_path, capsys
    ):
        """Rows with no `replicate` must report "underpowered", not a delta."""
        row = {"dataset_id": "toy", "naming": "descriptive", "leakage_remediated": True}
        before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
        for path in (before, after):
            path.write_text(json.dumps(row) + "\n")
        args = self.parse("eval-diff", str(before), str(after))

        assert cmd_eval_diff(args) == 0
        assert "underpowered" in capsys.readouterr().out

    def test_the_default_metrics_are_the_ones_evaldiff_declares(self):
        """Not a second copy of the metrics list, sourced from evaldiff instead."""
        from ds_agents.evaldiff import DEFAULT_METRICS

        args = self.parse("eval-diff", "b.jsonl", "a.jsonl")
        assert tuple(args.metrics.split(",")) == DEFAULT_METRICS

    def test_a_null_predicate_metric_runs_end_to_end(self, tmp_path, capsys):
        """A `:notnull` predicate must count null rows in the denominator, unlike a bare column
        name."""
        rows = [
            {"dataset_id": "toy", "naming": "descriptive", "halted_at": None},
            {"dataset_id": "toy", "naming": "descriptive", "halted_at": "profiler"},
        ]
        before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
        for path in (before, after):
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        args = self.parse("eval-diff", str(before), str(after), "--metrics", "halted_at:notnull")

        assert cmd_eval_diff(args) == 0
        out = capsys.readouterr().out
        assert "halted_at:notnull" in out, out
        # 2, not 1. A bare `halted_at` would have excluded the healthy row and read 1/1.
        assert "before  1/2" in out and "after   1/2" in out, out

    def test_an_unknown_predicate_exits_2_before_either_file_is_read(self, tmp_path, capsys):
        """A bad predicate must not fall through as a column name and silently tally n=0."""
        before, after = tmp_path / "b.jsonl", tmp_path / "a.jsonl"
        for path in (before, after):
            path.write_text(json.dumps({"dataset_id": "toy"}) + "\n")
        args = self.parse("eval-diff", str(before), str(after), "--metrics", "halted_at:notnul")

        assert cmd_eval_diff(args) == 2
        err = capsys.readouterr().err
        assert "unknown metric predicate" in err
        assert "notnul" in err
