"""The model client, as a Protocol, plus an offline placeholder.

Nodes call `model.generate(system=..., user=..., schema=SomePydanticModel)` and get back a typed
object plus its usage. They never import an SDK, which is what makes the Haiku/Sonnet reviewer
ablation a config change rather than an edit across six nodes.

The LangChain-vs-direct fork is settled: `AnthropicModel` below calls the Anthropic SDK directly.
`langgraph` already pulls in `langchain-core` and `langsmith`, so tracing was never the thing
LangChain would have bought us. Going direct buys constrained decoding via `messages.parse`
(a schema the server enforces, not a tool call we hope validates) and `response.usage` verbatim,
which matters because token counts are a published output of this project. See DECISIONS.md
(2026-08-26).

`StubModel` is NOT a model. It is a placeholder so the graph is runnable with no API key, and it
is deliberately bad at the thing this project measures: it nominates no leakage candidates at all.
Every `NodeEvent` it produces records `model="stub"`, so a results row built from one is
identifiable and must never be published.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from langsmith import traceable
from pydantic import BaseModel, ValidationError

from ds_agents.tools import pricing


@dataclass(frozen=True)
class Completion[T: BaseModel]:
    """A parsed structured response and what it cost.

    Usage travels with the value because `NodeEvent` needs it and a node that had to ask the
    client separately would drop it on every error path.
    """

    value: T
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class StructuredModel(Protocol):
    """The only way a node is allowed to talk to a model."""

    name: str

    def generate[M: BaseModel](
        self, *, system: str, user: str, schema: type[M]
    ) -> Completion[M]: ...


class ModelRefusal(RuntimeError):
    """The model returned something that will not validate against the schema.

    Raised rather than repaired: a node that quietly patched malformed output would turn "the
    model could not fill in Objection.columns" into "the reviewer found nothing", which are very
    different findings.
    """


@dataclass
class StubModel:
    """Offline placeholder. Mechanical, deterministic, and no substitute for a model.

    It answers only the schemas the nodes need, from the prompt payload the node passes as JSON.
    Anything else raises, so a new node cannot silently inherit fake answers.
    """

    name: str = "stub"

    def generate[M: BaseModel](self, *, system: str, user: str, schema: type[M]) -> Completion[M]:
        handler = getattr(self, f"_for_{schema.__name__}", None)
        if handler is None:
            raise ModelRefusal(
                f"StubModel has no answer for {schema.__name__}. Run with a real model, or add a "
                f"fake in the node's test rather than teaching the stub the answer."
            )
        return Completion(value=handler(_payload(user)), model=self.name)

    def _for_IntakeDecision(self, payload: dict[str, Any]) -> Any:  # noqa: N802 - schema name
        from ds_agents.nodes.intake import IntakeDecision

        columns: list[str] = payload.get("columns", [])
        n_unique: dict[str, int] = payload.get("n_unique", {})
        # Last column is the conventional target position in a tabular CSV. This is a convention,
        # not an inference, which is exactly why the stub is not a model.
        target = columns[-1] if columns else "target"
        uniques = n_unique.get(target, 2)
        task_type = "binary" if uniques == 2 else ("multiclass" if uniques <= 20 else "regression")
        metric = {"binary": "roc_auc", "multiclass": "accuracy", "regression": "rmse"}[task_type]
        return IntakeDecision(
            target=target,
            task_type=task_type,
            metric=metric,
            rationale="stub: last column by convention, task type by target cardinality",
        )

    def _for_LeakageNomination(self, payload: dict[str, Any]) -> Any:  # noqa: N802 - schema name
        from ds_agents.nodes.profiler import LeakageNomination

        # Empty on purpose. A stub that flagged the planted column would make the toy run look
        # like a working reviewer.
        return LeakageNomination(candidates=[], notes="stub: no model was called")

    def _for_FeaturePlan(self, payload: dict[str, Any]) -> Any:  # noqa: N802 - schema name
        from ds_agents.nodes.feature_eng import FeaturePlan

        # Drops nothing. The node's own forced drops (target, id columns, objected columns) still
        # apply, so the toy run produces a matrix -- it just produces the one that keeps the leak,
        # which is the honest depiction of a pipeline with no model in it.
        return FeaturePlan(drops=[], kept_despite_flag=[])

    def _for_ModelChoice(self, payload: dict[str, Any]) -> Any:  # noqa: N802 - schema name
        from ds_agents.nodes.modeler import ModelChoice

        # Takes the arithmetic answer the node already computed. Choosing between two fitted
        # candidates on cv_mean is not judgement, and pretending otherwise would let the stub's
        # runs look like the model contributed something.
        best = payload.get("best_by_cv")
        if not best:
            candidates = payload.get("candidates") or []
            best = candidates[0]["name"] if candidates else ""
        return ModelChoice(chosen=best, rationale="stub: best cv_mean, no model was called")

    def _for_ReviewFinding(self, payload: dict[str, Any]) -> Any:  # noqa: N802 - schema name
        from ds_agents.nodes.reviewer import ReviewFinding

        # Always "pass", no objections proposed, no dispositions offered. A stub that blocked, or
        # that echoed the open_objections it was shown back as findings, would exercise the review
        # loop with a reviewer that never actually looked at anything -- the same "looks like a
        # working pipeline that found nothing" problem `_for_LeakageNomination` exists to avoid,
        # one node downstream.
        return ReviewFinding(
            claim="pass", objections=[], dispositions=[], summary="stub: no model was called"
        )


@dataclass
class AnthropicModel:
    """The real client. One API call per `generate`, structured output enforced by the server.

    Uses `messages.parse` rather than asking for JSON in the prompt and parsing what comes back.
    The difference is not stylistic: the schema is enforced during decoding, so "the reviewer
    returned an objection with no `columns`" becomes impossible rather than becoming a silent
    zero in the leakage-recall column.

    Failures raise. There is no retry-with-a-nudge and no repair pass, because both would convert
    "this model could not do the task" -- a finding -- into "this model found nothing", which is a
    different finding that happens to look better.
    """

    model: str = "haiku"
    max_tokens: int = 4096
    timeout_s: float = 120.0
    client: Any = None
    name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        # Resolved once, here, so `NodeEvent.model` records the id that was actually billed
        # rather than the alias someone typed in a config file.
        self.name = pricing.resolve(self.model)
        # Fail now rather than after a benchmark run has already spent money on an unpriceable
        # model. This raises UnknownModelError for an id we have no rate for.
        pricing.price_for(self.name)
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic(timeout=self.timeout_s)

    # LangSmith gets the per-call span; `langgraph` already traces the graph itself, since it
    # depends on `langchain-core`. Inert unless LANGSMITH_TRACING and LANGSMITH_API_KEY are set,
    # so no key means no network call and no behaviour change -- and `node_trace` carries the
    # token counts regardless, so results files never depend on the tracing backend being up.
    @traceable(run_type="llm", name="ds_agents.generate")
    def generate[M: BaseModel](self, *, system: str, user: str, schema: type[M]) -> Completion[M]:
        response = self.client.messages.parse(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise ModelRefusal(
                f"{self.name} declined to answer for {schema.__name__}. Recorded as a node error "
                f"rather than an empty result."
            )
        value = getattr(response, "parsed_output", None)
        if value is None:
            raise ModelRefusal(
                f"{self.name} returned no parsed output for {schema.__name__} "
                f"(stop_reason={response.stop_reason!r}); it may have hit max_tokens mid-object."
            )
        if not isinstance(value, schema):
            # Belt and braces. The SDK validates, but a shape mismatch that reached a node would
            # be a Pydantic error thrown from deep inside unrelated code.
            try:
                value = schema.model_validate(value)
            except ValidationError as exc:
                raise ModelRefusal(f"{self.name} output failed {schema.__name__}: {exc}") from exc

        usage = response.usage
        return Completion(
            value=value,
            model=self.name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=pricing.cost_usd(
                self.name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                # Read directly, not via getattr with a default. A field the SDK renamed would
                # then price silently at zero, which is the same "quietly wrong in the flattering
                # direction" failure pricing.py refuses for unknown model ids. `or 0` only covers
                # the API's documented `null` for "no cache activity on this call".
                cache_write_tokens=usage.cache_creation_input_tokens or 0,
                cache_read_tokens=usage.cache_read_input_tokens or 0,
            ),
        )


def api_key_present() -> bool:
    """Whether a real client can be constructed.

    Lives here rather than in a node because nodes never read the environment -- that rule is what
    lets `tools/local.py` hand the sandbox a stripped env with no key in it. Only `cli.py` and the
    harness call this.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _payload(user: str) -> dict[str, Any]:
    """Nodes put their structured facts in the user message as JSON, so the stub can read them
    without parsing prose. A real model reads the same block."""
    start, end = user.find("{"), user.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        loaded = json.loads(user[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
