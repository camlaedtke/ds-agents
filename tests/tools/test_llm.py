"""The model client and its pricing.

These tests exist because `cost_usd` is a published number. Everything here is about one class of
failure: a run that reports a cost or a result which is quietly wrong in the flattering direction.
No test in this file makes a network call.
"""

import pytest
from pydantic import BaseModel

from ds_agents.tools import pricing
from ds_agents.tools.llm import AnthropicModel, Completion, ModelRefusal, StubModel

pytestmark = pytest.mark.fast


class Answer(BaseModel):
    verdict: str


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50, cache_write=0, cache_read=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_write
        self.cache_read_input_tokens = cache_read


class FakeResponse:
    def __init__(self, parsed_output, stop_reason="end_turn", usage=None):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.usage = usage or FakeUsage()


class FakeMessages:
    """Stands in for `client.messages`. Records the kwargs so the tests can assert on the call."""

    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def model_with(response, **kwargs) -> AnthropicModel:
    return AnthropicModel(client=FakeClient(response), **kwargs)


# --- pricing -------------------------------------------------------------------------------


def test_aliases_resolve_to_full_model_ids():
    # RunConfig carries "haiku", not a dated id, so this mapping is what connects a config field
    # to a billed model.
    assert pricing.resolve("haiku") == "claude-haiku-4-5"
    assert pricing.resolve("sonnet") == "claude-sonnet-5"


def test_a_full_model_id_passes_through_resolve_unchanged():
    assert pricing.resolve("claude-haiku-4-5") == "claude-haiku-4-5"


def test_the_published_rate_is_what_gets_charged():
    # Haiku 4.5 is $1.00 per Mtok in, $5.00 out. One million of each is $6.00 and nothing else.
    assert pricing.cost_usd("haiku", input_tokens=1_000_000, output_tokens=1_000_000) == 6.00


def test_cache_tiers_are_priced_separately_from_input():
    # Costing a cached read at the full input rate would overstate every Phase 4 benchmark run by
    # roughly the cache hit rate.
    cost = pricing.cost_usd(
        "haiku",
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    assert cost == pytest.approx(1.25 + 0.10)


def test_an_unknown_model_raises_rather_than_costing_nothing():
    # The whole point. A silent 0.0 is indistinguishable from a genuinely free run.
    with pytest.raises(pricing.UnknownModelError) as exc:
        pricing.cost_usd("gpt-4", input_tokens=10, output_tokens=10)
    assert "gpt-4" in str(exc.value)


def test_every_alias_points_at_a_priced_model():
    # An alias with no price is a config value that crashes only once someone runs an ablation.
    for alias, target in pricing.ALIASES.items():
        assert target in pricing.PRICES, f"alias {alias!r} resolves to unpriced {target!r}"


# --- the client ----------------------------------------------------------------------------


def test_generate_returns_the_parsed_value_with_its_usage_and_cost():
    model = model_with(FakeResponse(Answer(verdict="ok"), usage=FakeUsage(1000, 200)))
    completion = model.generate(system="s", user="u", schema=Answer)

    assert isinstance(completion, Completion)
    assert completion.value.verdict == "ok"
    assert completion.model == "claude-haiku-4-5"
    assert completion.input_tokens == 1000
    assert completion.output_tokens == 200
    assert completion.cost_usd == pytest.approx((1000 * 1.00 + 200 * 5.00) / 1_000_000)


def test_the_recorded_model_is_the_resolved_id_not_the_alias():
    # NodeEvent.model ends up in a results row. It must name what was billed, not what someone
    # typed in a config file.
    model = model_with(FakeResponse(Answer(verdict="ok")), model="haiku")
    assert model.name == "claude-haiku-4-5"
    assert model.generate(system="s", user="u", schema=Answer).model == "claude-haiku-4-5"


def test_the_schema_is_sent_as_output_format_not_asked_for_in_the_prompt():
    # Constrained decoding is the guarantee. If this ever regresses to prompt-and-parse, a
    # reviewer objection missing its `columns` becomes a silent zero in leakage_recall.
    model = model_with(FakeResponse(Answer(verdict="ok")))
    model.generate(system="sys", user="usr", schema=Answer)

    call = model.client.messages.calls[0]
    assert call["output_format"] is Answer
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "usr"}]
    assert call["model"] == "claude-haiku-4-5"


def test_constructing_a_client_for_an_unpriced_model_fails_immediately():
    # Fail before a benchmark run spends money, not after it produces an unpriceable row.
    with pytest.raises(pricing.UnknownModelError):
        AnthropicModel(model="claude-imaginary-9", client=FakeClient(None))


def test_a_refusal_raises_rather_than_returning_an_empty_result():
    # "The model declined" and "the model found nothing" are different findings. Collapsing them
    # would make a refusing reviewer look like a clean pass.
    model = model_with(FakeResponse(None, stop_reason="refusal"))
    with pytest.raises(ModelRefusal):
        model.generate(system="s", user="u", schema=Answer)


def test_missing_parsed_output_raises_and_says_why():
    model = model_with(FakeResponse(None, stop_reason="max_tokens"))
    with pytest.raises(ModelRefusal) as exc:
        model.generate(system="s", user="u", schema=Answer)
    assert "max_tokens" in str(exc.value)


def test_a_client_error_propagates_for_the_node_to_record():
    # Nodes wrap the model call in a broad except and turn this into a PipelineError plus a
    # NodeEvent. The client's job is to not swallow it.
    model = model_with(TimeoutError("upstream timeout"))
    with pytest.raises(TimeoutError):
        model.generate(system="s", user="u", schema=Answer)


# --- the stub ------------------------------------------------------------------------------


def test_every_stub_completion_is_stamped_as_a_placeholder():
    # This stamp is the only thing standing between a placeholder run and a results table.
    from ds_agents.nodes.intake import IntakeDecision

    stub = StubModel()
    completion = stub.generate(
        system="s",
        user='{"columns": ["a", "churned"], "n_unique": {"churned": 2}}',
        schema=IntakeDecision,
    )
    assert completion.model == "stub"
    assert completion.cost_usd == 0.0


def test_the_stub_refuses_a_schema_it_was_never_taught():
    with pytest.raises(ModelRefusal):
        StubModel().generate(system="s", user="u", schema=Answer)
