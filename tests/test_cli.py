"""cli.py's model selection: real client when a key resolves, placeholder otherwise.

`_select_model` is the one place a benchmark run could quietly become a placeholder run because an
env var went missing, so the tests here are about that boundary: `--no-live` always wins, no key
falls back to the stub, a key present builds a real client, and an unpriced `--model` value fails
loudly (`UnknownModelError` -> `SystemExit`) rather than running unbilled.

No test here makes a network call: `AnthropicModel.__post_init__` only constructs an SDK client
(`anthropic.Anthropic(timeout=...)`), which needs no network -- the client talks to the network
only inside `generate()`, which nothing here calls.
"""

import argparse
import json

import pytest

from ds_agents.cli import (
    _append_results_row,
    _build_parser,
    _fixture_state,
    _select_model,
    _select_reviewer_model,
    cmd_run,
)
from ds_agents.fixtures import load_fixture
from ds_agents.naming import NAMINGS, header_of, materialize
from ds_agents.state import NodeEvent, PipelineState, RunConfig, utc_now
from ds_agents.tools.llm import AnthropicModel, StubModel

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch):
    """Every test starts from "no key in the environment," regardless of the shell this suite
    happens to run in, so a developer's exported ANTHROPIC_API_KEY cannot flip a test's outcome."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def test_no_live_returns_the_stub_even_with_a_key_present(monkeypatch):
    # The only way to ask for the stub when a key IS available -- must win over the key, not the
    # other way around, or a benchmark run could not be forced offline for a cheap smoke test.
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
    # UnknownModelError is deliberately fatal (see pricing.py): a run that reports $0.00 for an
    # unpriced model is worse than a run that never starts.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    with pytest.raises(SystemExit):
        _select_model(RunConfig(default_model="not-a-real-model"))


class TestGroundTruthFollowsTheRename:
    """`_fixture_state` has to write the names the agents will actually see.

    Under `--naming opaque` the planted column is `var_07`, not `adjuster_touches`. If the state
    kept the manifest's names, `results_row()` would compare the reviewer's objections against
    columns that do not exist in that arm's data, every opaque run would score a silent zero, and
    the ablation would report a name effect that was entirely an artefact of the bookkeeping.
    """

    def test_descriptive_keeps_the_manifest_names(self):
        fixture = load_fixture("claims_timing")
        state = _fixture_state(fixture)
        assert state.planted_leakage_columns == fixture.manifest.planted_columns
        assert state.config.naming == "descriptive"

    def test_opaque_carries_the_renamed_columns(self, tmp_path):
        fixture = load_fixture("claims_timing")
        _, rename = materialize(fixture, "opaque", tmp_path)
        state = _fixture_state(fixture, naming="opaque")

        assert state.config.naming == "opaque"
        assert state.planted_leakage_columns == [
            rename[c] for c in fixture.manifest.planted_columns
        ]
        assert all(c.startswith("var_") for c in state.planted_leakage_columns)

    def test_the_planted_columns_exist_in_the_data_the_agents_see(self, tmp_path):
        """The whole point of the previous test, stated against the file rather than the map."""
        fixture = load_fixture("claims_timing")
        path, _ = materialize(fixture, "opaque", tmp_path)
        state = _fixture_state(fixture, naming="opaque")
        assert set(state.planted_leakage_columns) <= set(header_of(path))

    def test_the_task_description_is_unaffected(self):
        """It names only the target, and the target is never renamed -- intake still has to infer
        it from prose, which is the node's actual job."""
        fixture = load_fixture("claims_timing")
        assert (
            _fixture_state(fixture, naming="opaque").task_description
            == _fixture_state(fixture).task_description
        )

    def test_the_arm_cannot_be_claimed_without_the_rename_being_applied(self):
        """The footgun the signature used to allow.

        `naming` and the map were once two independent arguments, which made a state constructible
        that said `opaque` while carrying the fixture's real column names -- config and ground truth
        disagreeing, every objection compared against columns absent from that arm's data, and the
        arm scoring a silent zero that looks exactly like a profiler success. There is now no way to
        express it: the map is derived from `naming` inside the function.
        """
        fixture = load_fixture("claims_timing")
        for naming in NAMINGS:
            state = _fixture_state(fixture, naming=naming)
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

    def test_the_ablation_flags_parse(self):
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

    def test_an_unknown_naming_is_refused(self):
        with pytest.raises(SystemExit):
            self._parse(["run", "--naming", "scrambled"])

    def test_the_reviewer_prompt_defaults_to_base(self):
        assert self._parse(["run"]).reviewer_prompt == "base"

    def test_the_reviewer_prompt_variant_parses(self):
        args = self._parse(["run", "--reviewer-prompt", "which_column"])
        assert args.reviewer_prompt == "which_column"

    def test_an_unknown_reviewer_prompt_is_refused(self):
        with pytest.raises(SystemExit):
            self._parse(["run", "--reviewer-prompt", "helpful_hints"])

    def test_the_loop_cap_defaults_to_the_config_default(self):
        """3 in two places -- the flag and `RunConfig` -- and they must agree, or `ds-agents run`
        and the not-yet-written harness would run the same nominal condition differently."""
        assert self._parse(["run"]).loop_cap == RunConfig().loop_cap == 3

    def test_the_loop_cap_parses(self):
        assert self._parse(["run", "--loop-cap", "5"]).loop_cap == 5

    def test_a_zero_cap_is_valid_and_not_confused_with_a_negative_one(self):
        """`RunConfig` allows `ge=0` and the router already special-cases it -- a cap of 0 is the
        reviewer-off condition expressed as a cap, not a typo. Only `< 0` is refused."""
        assert self._parse(["run", "--loop-cap", "0"]).loop_cap == 0


class TestTheLoopCapIsRecorded:
    """Mirror of TestThePromptConditionIsRecorded. The cap decides how many chances feature_eng
    gets to act on an objection, so a row that did not carry it would be averaged together with
    rows run under a different cap."""

    def test_fixture_state_records_the_cap(self):
        fixture = load_fixture("claims_timing")
        assert _fixture_state(fixture).config.loop_cap == 3
        assert _fixture_state(fixture, loop_cap=5).config.loop_cap == 5

    def test_a_negative_cap_exits_two_rather_than_raising(self):
        """`RunConfig` would refuse this with `ge=0`, but as a Pydantic traceback. This is the
        exit-code-2 pattern `--repeat` already uses."""
        args = _build_parser().parse_args(["run", "--loop-cap", "-1"])
        assert cmd_run(args) == 2


class TestThePromptConditionIsRecorded:
    """Mirror of TestGroundTruthFollowsTheRename: the condition must land on the frozen config,
    or the arms are indistinguishable in the results file."""

    def test_fixture_state_records_the_variant(self):
        fixture = load_fixture("claims_timing")
        assert _fixture_state(fixture).config.reviewer_prompt == "base"
        state = _fixture_state(fixture, reviewer_prompt="which_column")
        assert state.config.reviewer_prompt == "which_column"


class TestTheRoutingConditionIsRecorded:
    """Mirror of TestThePromptConditionIsRecorded, and the one with the sharpest edge: the two
    arms differ in whether a column-scoped objection can be acted on at all, so their remediation
    rates are not comparable and a row that did not carry the condition would be averaged with
    rows that had a capability it did not."""

    def test_the_default_is_the_old_behaviour(self):
        """Every row committed before 2026-08-28 ran under this, so the default has to be the
        arm that reproduces them byte for byte."""
        args = _build_parser().parse_args(["run"])
        assert args.objection_routing == "as_addressed"
        assert _fixture_state(load_fixture("claims_timing")).config.objection_routing == (
            "as_addressed"
        )

    def test_fixture_state_records_the_variant(self):
        state = _fixture_state(load_fixture("claims_timing"), objection_routing="by_category")
        assert state.config.objection_routing == "by_category"
        assert state.results_row()["objection_routing"] == "by_category"

    def test_an_unknown_routing_is_refused_by_the_parser(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["run", "--objection-routing", "by_vibes"])


class TestTheReviewerClient:
    """`_select_reviewer_model`, untested until the session that spends money on it.

    Same no-network property as `_select_model`'s tests above: `AnthropicModel.__post_init__`
    only constructs an SDK client.
    """

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
        """The Haiku/Sonnet ablation arm. The reviewer's client must be a different object with
        the Sonnet id while the base keeps Haiku."""
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
        """The same gate the Phase 4 harness applies. A results file that quietly accepted stub
        rows would look exactly like a results file."""
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
