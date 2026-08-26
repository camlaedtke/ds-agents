"""The model client, as a Protocol, plus an offline placeholder.

Nodes call `model.generate(system=..., user=..., schema=SomePydanticModel)` and get back a typed
object plus its usage. They never import an SDK. That keeps the open fork in docs/NEXT.md
(LangChain `with_structured_output` vs the Anthropic API directly) a one-file decision instead of
one spread across six nodes, and it is what makes the Haiku/Sonnet reviewer ablation a config
change rather than an edit.

`StubModel` is NOT a model. It is a placeholder so the graph is runnable with no API key, and it
is deliberately bad at the thing this project measures: it nominates no leakage candidates at all.
Every `NodeEvent` it produces records `model="stub"`, so a results row built from one is
identifiable and must never be published.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel


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

    It answers only the schemas Phase 1 needs, from the prompt payload the node passes as JSON.
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
