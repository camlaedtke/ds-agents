"""cli.py's model selection: real client when a key resolves, placeholder otherwise.

`_select_model` is the one place a benchmark run could quietly become a placeholder run because an
env var went missing, so the tests here are about that boundary: `--no-live` always wins, no key
falls back to the stub, a key present builds a real client, and an unpriced `--model` value fails
loudly (`UnknownModelError` -> `SystemExit`) rather than running unbilled.

No test here makes a network call: `AnthropicModel.__post_init__` only constructs an SDK client
(`anthropic.Anthropic(timeout=...)`), which needs no network -- the client talks to the network
only inside `generate()`, which nothing here calls.
"""

import pytest

from ds_agents.cli import _select_model
from ds_agents.state import RunConfig
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
