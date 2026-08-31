"""One dataset a run can be built from, whichever registry it came from.

Two registries feed this pipeline and they are not the same kind of thing. `tests/fixtures/` holds
datasets this project wrote, each with a complete answer key: `manifest.json` names every planted
trap, so a run on one can be graded. `evals/datasets/manifest.yaml` holds 13 external datasets from
the AutoML Benchmark, which carry no answer key at all and say so with `leakage_labelled: false`.
Both need to reach `run_pipeline`; only one of them can be scored for leakage.

NEXT.md called for a `Runnable` *protocol*. This is a concrete adapter with two named constructors
instead, and the deviation is the point. There is no type checker in this repo -- the dev
dependencies are pytest and ruff -- so a structural `typing.Protocol` would be a comment, and the
thing it would be commenting on is precisely the guardrail `DatasetEntry`'s own docstring says must
be a property rather than a promise: if a `DatasetEntry` were duck-substitutable for a `Fixture`,
`_run_state` would accept one and silently build a run with `planted_leakage_columns=[]` -- a row
asserting the dataset contains no leak and that the reviewer failed to find it, with no evidence
for either. Under this shape you cannot reach `_run_state` from a `DatasetEntry` without passing
through `from_dataset`, which is one reviewed line that writes the empty list on purpose.

Nothing in `nodes/` imports this module, for the reason `fixtures.py` gives and one more: this one
reaches *both* registries, so a node that imported it could read the answer key by name.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ds_agents import benchmark, fixtures
from ds_agents.benchmark import DatasetEntry
from ds_agents.fixtures import Fixture
from ds_agents.state import DatasetSource

# How much of an external dataset is withheld from the agents entirely. Fixtures get 0.0; see
# `from_fixture` for why that is a decision rather than an omission.
WITHHELD_FRACTION = 0.2


class Runnable(BaseModel):
    """A dataset resolved to the handful of facts a run actually needs.

    Deliberately absent: `metric`, `task_type` and `positive_class`, all of which `DatasetEntry`
    carries. Naming the metric and the task is intake's job, and that node is under test -- carrying
    the manifest's answer alongside the prose would create a second source of truth that nothing
    reconciles, and the first time they disagreed the run would use one and the row would report the
    other.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    csv_path: Path
    target: str = Field(description="The label column. Never renamed by the naming ablation.")
    task_description: str
    planted_columns: list[str] = Field(
        description=(
            "The COMPLETE list of planted leakage columns, or [] where there is no answer key. "
            "Read as complete downstream: `results_row()` scores every other flagged column as a "
            "false alarm against it, which is why a partial list is worse than an empty one."
        )
    )
    withheld_fraction: float = Field(ge=0.0, lt=0.5)
    source: DatasetSource

    @classmethod
    def from_fixture(cls, fixture: Fixture) -> "Runnable":
        """A fixture: complete answer key, and no withheld holdout.

        `withheld_fraction=0.0` is a decision. Carving 20% out of a 200-row toy changes what the
        agents are shown, which changes what they do, which makes all 145 committed rows
        incomparable to everything written after -- and it buys nothing, because planted leakage is
        a *column*. A randomly withheld holdout still contains it, so a leaky model's claimed score
        does not collapse on those rows and there is nothing there to catch.
        """
        return cls(
            dataset_id=fixture.dataset_id,
            csv_path=fixture.csv_path,
            target=fixture.manifest.target,
            task_description=fixture.task_description,
            planted_columns=list(fixture.manifest.planted_columns),
            withheld_fraction=0.0,
            source="fixture",
        )

    @classmethod
    def from_dataset(cls, entry: DatasetEntry) -> "Runnable":
        """An external benchmark dataset: no answer key, and a withheld holdout.

        `planted_columns=[]` is the line this whole module exists to make visible. It is empty
        because nobody knows what is leaky in someone else's dataset, and `graded_for_leakage` in
        `results_row()` reads the emptiness and returns `None` from all nine leakage-rate columns
        rather than scoring a rate against a list that is not an answer key.

        `entry.known_leakage` is NOT promoted into it. `bank_marketing`'s documented `V12` is a
        real, externally sourced leak, but one known leak is not a complete list, and claiming
        completeness would score every other genuinely suspicious column the reviewer names as a
        false alarm. `KnownLeak.in_planted_leakage_columns` is a `Literal[False]` so this cannot
        change without a schema change.
        """
        return cls(
            dataset_id=entry.dataset_id,
            csv_path=benchmark.require_cached_csv(entry),
            target=entry.target,
            task_description=entry.task_description,
            planted_columns=[],
            withheld_fraction=WITHHELD_FRACTION,
            source="benchmark",
        )

    def sha256(self) -> str:
        """The bytes on disk. Not `DatasetEntry.csv_sha256`, which is the upstream file -- what
        belongs on a row is what this run actually read."""
        return hashlib.sha256(self.csv_path.read_bytes()).hexdigest()


def available() -> dict[str, DatasetSource]:
    """Every runnable name, and which registry it came from. Sorted, for stable help text."""
    names: dict[str, DatasetSource] = {name: "fixture" for name in fixtures.available()}
    names.update({name: "benchmark" for name in benchmark.available()})
    return dict(sorted(names.items()))


def resolve(name: str) -> Runnable:
    """Resolve a dataset name against both registries, or `SystemExit` naming both.

    Fixtures are checked first. The order is only defensive -- a test asserts the two namespaces
    are disjoint -- but if that ever breaks, resolving to the fixture is the safe side of the
    collision: it is the one with an answer key, so a run under the wrong resolution is
    over-graded and loud rather than silently ungraded.
    """
    known = available()
    source = known.get(name)
    if source == "fixture":
        return Runnable.from_fixture(fixtures.load_fixture(name))
    if source == "benchmark":
        return Runnable.from_dataset(benchmark.load_dataset(name))
    fixture_names = ", ".join(fixtures.available()) or "(none found)"
    dataset_names = ", ".join(benchmark.available()) or "(none; run `ds-agents datasets refresh`)"
    raise SystemExit(
        f"no dataset named {name!r}.\n  fixtures:  {fixture_names}\n  benchmark: {dataset_names}"
    )
