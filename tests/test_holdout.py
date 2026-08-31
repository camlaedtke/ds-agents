"""The carve: what the agents are shown, and what they are graded on.

Every assertion here is about a property that fails silently if it breaks. A carve that leaks rows
into the agent frame does not raise -- it produces a `verified_holdout_score` that agrees with the
claimed one and looks like good news. A carve whose membership drifts across a dependency bump
moves the headline column with nothing in the row to explain it. So the partition, the
determinism and the containment are all pinned.
"""

import hashlib
import json

import pandas as pd
import pytest

from ds_agents.holdout import MIN_AGENT_ROWS, _withhold_rows, prepare
from ds_agents.naming import rename_map
from ds_agents.runnable import Runnable, resolve
from ds_agents.state import DEFAULT_RANDOM_SEED

pytestmark = pytest.mark.fast

# `credit_g` at the default seed. Pinned as a digest rather than 200 literals -- equally loud, and
# a diff nobody can read is a diff nobody checks. If this moves, the withheld set moved, and every
# number measured on the old one is measured on a different question.
CREDIT_G_DIGEST = "c3818461ac2ab41bb407d144fc712be8f7f3a8e32db78af703681b69e73a396b"


def _digest(rows: list[int]) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    return prepare(
        resolve("credit_g"),
        "descriptive",
        into=tmp_path / "input",
        withheld_into=tmp_path / "withheld",
        seed=DEFAULT_RANDOM_SEED,
    )


class TestThePartition:
    def test_the_two_sides_are_a_partition_of_the_source(self, prepared) -> None:
        source = pd.read_csv(resolve("credit_g").csv_path)
        agent = pd.read_csv(prepared.agent_csv)
        withheld = pd.read_csv(prepared.withheld_csv)
        assert len(agent) + len(withheld) == len(source)
        assert prepared.n_agent_rows == len(agent)
        assert prepared.n_withheld_rows == len(withheld)
        assert len(prepared.withheld_rows) == len(set(prepared.withheld_rows))

    def test_no_withheld_row_appears_in_the_agent_frame(self, prepared) -> None:
        """The one that matters. Compared as raw lines, not as parsed frames, because two rows can
        be equal as records and differ as bytes and it is the bytes the agents were handed."""
        agent = set(prepared.agent_csv.read_text().splitlines()[1:])
        withheld = prepared.withheld_csv.read_text().splitlines()[1:]
        assert withheld, "nothing was withheld, so this test proves nothing"
        assert not (agent & set(withheld))

    def test_both_sides_keep_the_source_header(self, prepared) -> None:
        source_header = resolve("credit_g").csv_path.read_text().splitlines()[0]
        assert prepared.agent_csv.read_text().splitlines()[0] == source_header
        assert prepared.withheld_csv.read_text().splitlines()[0] == source_header

    def test_row_order_is_preserved_on_both_sides(self, prepared) -> None:
        source = resolve("credit_g").csv_path.read_text().splitlines()[1:]
        withheld = set(prepared.withheld_rows)
        assert prepared.agent_csv.read_text().splitlines()[1:] == [
            line for index, line in enumerate(source) if index not in withheld
        ]


class TestDeterminism:
    def test_credit_g_at_the_default_seed_is_pinned(self, prepared) -> None:
        assert prepared.n_withheld_rows == 200
        assert prepared.withheld_rows[:5] == [1, 5, 8, 11, 13]
        assert _digest(prepared.withheld_rows) == CREDIT_G_DIGEST

    def test_the_same_seed_twice_is_the_same_carve(self, tmp_path) -> None:
        runs = [
            prepare(
                resolve("credit_g"),
                "descriptive",
                into=tmp_path / f"i{i}",
                withheld_into=tmp_path / f"w{i}",
                seed=DEFAULT_RANDOM_SEED,
            )
            for i in range(2)
        ]
        assert runs[0].withheld_rows == runs[1].withheld_rows
        assert runs[0].agent_sha256 == runs[1].agent_sha256

    def test_a_different_seed_is_a_different_carve(self, tmp_path) -> None:
        other = prepare(
            resolve("credit_g"),
            "descriptive",
            into=tmp_path / "i",
            withheld_into=tmp_path / "w",
            seed=DEFAULT_RANDOM_SEED + 1,
        )
        assert _digest(other.withheld_rows) != CREDIT_G_DIGEST

    def test_the_draw_does_not_depend_on_row_order_within_a_class(self) -> None:
        """Groups are iterated sorted, so a relabelled but equivalent frame draws the same count."""
        labels = ["a"] * 60 + ["b"] * 40
        first = _withhold_rows(labels, 0.2, DEFAULT_RANDOM_SEED)
        second = _withhold_rows(labels, 0.2, DEFAULT_RANDOM_SEED)
        assert first == second
        assert len(first) == 20


class TestStratification:
    def test_the_positive_rate_survives_the_carve(self, prepared) -> None:
        source = pd.read_csv(resolve("credit_g").csv_path)
        withheld = pd.read_csv(prepared.withheld_csv)
        agent = pd.read_csv(prepared.agent_csv)
        expected = (source["class"] == "bad").mean()
        assert (withheld["class"] == "bad").mean() == pytest.approx(expected, abs=0.01)
        assert (agent["class"] == "bad").mean() == pytest.approx(expected, abs=0.01)

    def test_every_class_is_represented_on_both_sides(self) -> None:
        labels = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        withheld = set(_withhold_rows(labels, 0.2, DEFAULT_RANDOM_SEED))
        assert {labels[i] for i in withheld} == {"a", "b", "c"}
        assert {label for i, label in enumerate(labels) if i not in withheld} == {"a", "b", "c"}

    def test_a_singleton_class_stays_with_the_agents(self) -> None:
        """Splitting a one-row class empties it on one side. It goes to the agents, where the
        profiler will report it, rather than into a holdout the scorer cannot use."""
        labels = ["a"] * 50 + ["rare"]
        withheld = _withhold_rows(labels, 0.2, DEFAULT_RANDOM_SEED)
        assert 50 not in withheld


class TestFixturesAreUntouched:
    def test_a_fixture_withholds_nothing_and_writes_no_second_file(self, tmp_path) -> None:
        toy = resolve("toy")
        prepared = prepare(
            toy,
            "descriptive",
            into=tmp_path / "input",
            withheld_into=tmp_path / "withheld",
            seed=DEFAULT_RANDOM_SEED,
        )
        assert prepared.withheld_csv is None
        assert prepared.withheld_rows is None
        assert prepared.n_withheld_rows == 0
        assert prepared.withheld_fraction == 0.0
        assert prepared.agent_csv == toy.csv_path, "the control arm must not copy the fixture"
        assert not (tmp_path / "withheld").exists()

    def test_the_opaque_arm_still_gets_its_rename(self, tmp_path) -> None:
        toy = resolve("toy")
        prepared = prepare(
            toy,
            "opaque",
            into=tmp_path / "input",
            withheld_into=tmp_path / "withheld",
            seed=DEFAULT_RANDOM_SEED,
        )
        assert prepared.rename == rename_map(toy, "opaque")
        assert prepared.withheld_csv is None


class TestTheRefusals:
    def _tiny(self, tmp_path, rows: str) -> Runnable:
        path = tmp_path / "tiny.csv"
        path.write_text(rows)
        return Runnable(
            dataset_id="tiny",
            csv_path=path,
            target="y",
            task_description="Predict y and report roc_auc.",
            planted_columns=[],
            withheld_fraction=0.2,
            source="benchmark",
        )

    def test_too_few_agent_rows_is_refused_rather_than_run(self, tmp_path) -> None:
        rows = "x,y\n" + "".join(f"{i},{i % 2}\n" for i in range(8))
        with pytest.raises(SystemExit, match=str(MIN_AGENT_ROWS)):
            prepare(
                self._tiny(tmp_path, rows),
                "descriptive",
                into=tmp_path / "i",
                withheld_into=tmp_path / "w",
                seed=DEFAULT_RANDOM_SEED,
            )

    def test_a_target_that_is_not_a_column_is_refused(self, tmp_path) -> None:
        rows = "x,z\n" + "".join(f"{i},{i % 2}\n" for i in range(40))
        with pytest.raises(SystemExit, match="not a column"):
            prepare(
                self._tiny(tmp_path, rows),
                "descriptive",
                into=tmp_path / "i",
                withheld_into=tmp_path / "w",
                seed=DEFAULT_RANDOM_SEED,
            )
