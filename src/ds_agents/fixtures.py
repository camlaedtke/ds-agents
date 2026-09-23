"""The fixture registry: how a dataset name on the command line becomes a run.

Each fixture is a directory under `tests/fixtures/` holding three committed files -- `generate.py`
with a fixed seed, `<name>.csv`, and `manifest.json`. The manifest is ground truth: it records what
was planted so a run can be graded against what is actually in the data rather than against a
recollection of what was intended. `generate.py` writes the manifest at the same moment it writes
the CSV, so the two cannot drift apart without the fixture tests noticing.

This module is deliberately dumb. It resolves paths, validates the manifest shape, and hands back a
typed object. It does not read the CSV, and nothing in `nodes/` imports it -- a node that could load
a manifest could read the answer key, and `planted_leakage_columns` would stop measuring anything.
Only `cli.py` and the harness call in here.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


class PlantedLeak(BaseModel):
    """One planted trap, keyed to the column that carries it.

    `kind` is free text on purpose. The traps differ in mechanism -- a noisy copy of the target, a
    field recorded after the outcome, an identifier re-issued by outcome -- and forcing them into an
    enum now would mean guessing the taxonomy before the third and fourth traps exist.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    kind: str
    why: str
    agreement_with_target: float | None = Field(
        default=None,
        description="Only meaningful for a trap that implies a label per row. A trap whose signal "
        "is distributional rather than per-row leaves this null rather than inventing a number.",
    )
    mutual_info_with_target: float | None = Field(
        default=None,
        description="The realised NMI, recorded because it is the entire evidence the profiler's "
        "model is given. It is what says whether this trap is a scream or a whisper.",
    )


class FixtureManifest(BaseModel):
    """What `manifest.json` is allowed to contain.

    `extra="forbid"` for the same reason `Contract` uses it: a typo in a hand-edited manifest would
    otherwise become a silently missing piece of ground truth, and the fixture would grade runs
    against a field nothing reads.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int
    n_rows: int
    target: str
    task_type: str
    metric: str
    planted_leakage: list[PlantedLeak]
    legit_strong_features: list[str] = Field(default_factory=list)
    noise_features: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    positive_rate: float | None = None

    @property
    def planted_columns(self) -> list[str]:
        return [leak.column for leak in self.planted_leakage]


class Fixture(BaseModel):
    """A resolved fixture: the manifest plus where its data lives."""

    model_config = ConfigDict(extra="forbid")

    name: str
    csv_path: Path
    manifest: FixtureManifest

    @property
    def dataset_id(self) -> str:
        return self.name

    @property
    def task_description(self) -> str:
        """What a person would actually say, not a spec.

        Naming the target is intake's job; this is the prose it has to work from.
        """
        return f"Predict {self.manifest.target} and report {self.manifest.metric}."


def available() -> list[str]:
    """Every fixture directory that has both files. Sorted, so error messages are stable."""
    if not FIXTURES_ROOT.is_dir():
        return []
    return sorted(
        entry.name
        for entry in FIXTURES_ROOT.iterdir()
        if entry.is_dir()
        and (entry / "manifest.json").exists()
        and (entry / f"{entry.name}.csv").exists()
    )


def load_fixture(name: str) -> Fixture:
    """Resolve a fixture by name, or raise `SystemExit` with something actionable.

    `SystemExit` rather than an exception because both callers are entry points, and a traceback
    for "you typed a name that does not exist" reads like a pipeline bug.
    """
    directory = FIXTURES_ROOT / name
    manifest_path = directory / "manifest.json"
    csv_path = directory / f"{name}.csv"
    if not manifest_path.exists() or not csv_path.exists():
        known = ", ".join(available()) or "(none found)"
        raise SystemExit(
            f"no fixture named {name!r} under {FIXTURES_ROOT}. Known fixtures: {known}. "
            f"These paths assume an editable install; run from a checkout with `uv run ds-agents`."
        )
    manifest = FixtureManifest.model_validate(json.loads(manifest_path.read_text()))
    return Fixture(name=name, csv_path=csv_path, manifest=manifest)
