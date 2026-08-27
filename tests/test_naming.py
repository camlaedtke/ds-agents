"""Does the opaque arm change the names and nothing else?

This module is the validity check for the name-transparency ablation. The finding it supports --
that the profiler reads column names rather than the association numbers -- is only a finding if
the two arms are identical in every other respect. If `materialize` re-formatted a float, dropped a
quote, reordered a column or renamed the target, the difference between the arms would have a
second candidate explanation and the whole comparison would be worth nothing.

So the assertions here are deliberately about bytes rather than about frames. `test_only_the_header
_line_differs` is the one the result rests on: it compares the file body character for character.

Marked `fast`: no model, no sandbox, one small CSV per fixture.
"""

import csv

import pandas as pd
import pytest

from ds_agents import naming
from ds_agents.fixtures import available, load_fixture

pytestmark = pytest.mark.fast

FIXTURES = available()


@pytest.fixture(params=FIXTURES)
def fixture(request):
    return load_fixture(request.param)


class TestDescriptiveIsTheIdentity:
    def test_the_map_is_empty(self, fixture):
        assert naming.rename_map(fixture, "descriptive") == {}

    def test_it_hands_back_the_committed_file_and_writes_nothing(self, fixture, tmp_path):
        into = tmp_path / "unused"
        path, mapping = naming.materialize(fixture, "descriptive", into)
        assert path == fixture.csv_path
        assert mapping == {}
        assert not into.exists(), (
            "the control arm copied the dataset, which is a way for it to drift"
        )


class TestTheOpaqueMap:
    def test_it_covers_every_column_except_the_target(self, fixture):
        mapping = naming.rename_map(fixture, "opaque")
        columns = naming.header_of(fixture.csv_path)
        target = fixture.manifest.target
        assert set(mapping) == set(columns) - {target}
        assert target not in mapping

    def test_it_is_injective(self, fixture):
        """Two columns collapsing onto one name would silently drop a column from the matrix."""
        mapping = naming.rename_map(fixture, "opaque")
        assert len(set(mapping.values())) == len(mapping)

    def test_no_new_name_collides_with_the_target(self, fixture):
        mapping = naming.rename_map(fixture, "opaque")
        assert fixture.manifest.target not in set(mapping.values())

    def test_the_names_carry_no_signal(self, fixture):
        """Dense numbering in column order. A gap would say which position was skipped."""
        mapping = naming.rename_map(fixture, "opaque")
        expected = [f"var_{i:02d}" for i in range(1, len(mapping) + 1)]
        columns = naming.header_of(fixture.csv_path)
        assert [mapping[c] for c in columns if c != fixture.manifest.target] == expected

    def test_it_is_deterministic(self, fixture):
        assert naming.rename_map(fixture, "opaque") == naming.rename_map(fixture, "opaque")

    def test_every_declared_column_survives_the_rename(self, fixture):
        """The manifest's ground truth has to still name something that exists in the data."""
        mapping = naming.rename_map(fixture, "opaque")
        header = set(naming.header_of(fixture.csv_path))
        manifest = fixture.manifest
        declared = (
            manifest.planted_columns
            + manifest.legit_strong_features
            + manifest.noise_features
            + manifest.id_columns
        )
        for column in declared:
            assert column in header, f"{column} is declared but not in the CSV"
            assert naming.apply([column], mapping)[0] in set(mapping.values())


class TestTheMaterialisedFile:
    def test_only_the_header_line_differs(self, fixture, tmp_path):
        """The assertion the finding rests on. Everything below line 1 is copied through."""
        path, _ = naming.materialize(fixture, "opaque", tmp_path)
        with fixture.csv_path.open("r", encoding="utf-8", newline="") as handle:
            original = handle.read()
        with path.open("r", encoding="utf-8", newline="") as handle:
            renamed = handle.read()
        assert renamed.split("\n", 1)[1] == original.split("\n", 1)[1]
        assert renamed.split("\n", 1)[0] != original.split("\n", 1)[0]

    def test_the_header_is_exactly_the_map_applied(self, fixture, tmp_path):
        path, mapping = naming.materialize(fixture, "opaque", tmp_path)
        original = naming.header_of(fixture.csv_path)
        assert naming.header_of(path) == naming.apply(original, mapping)

    def test_column_order_and_the_target_are_untouched(self, fixture, tmp_path):
        """intake still has to find the target from prose, so the target keeps its name."""
        path, _ = naming.materialize(fixture, "opaque", tmp_path)
        original = naming.header_of(fixture.csv_path)
        opaque = naming.header_of(path)
        target = fixture.manifest.target
        assert len(opaque) == len(original)
        assert opaque.index(target) == original.index(target)

    def test_the_values_are_identical_column_by_column(self, fixture, tmp_path):
        """The bytes assertion again, this time as pandas sees it -- because pandas is what the
        agents' snippets actually read the file with."""
        path, mapping = naming.materialize(fixture, "opaque", tmp_path)
        opaque = pd.read_csv(path)
        original = pd.read_csv(fixture.csv_path)
        pd.testing.assert_frame_equal(
            opaque.rename(columns={v: k for k, v in mapping.items()}), original
        )

    def test_it_writes_beside_nothing_else(self, fixture, tmp_path):
        path, _ = naming.materialize(fixture, "opaque", tmp_path)
        assert path.parent == tmp_path
        assert path.name == fixture.csv_path.name
        assert [p.name for p in tmp_path.iterdir()] == [fixture.csv_path.name]

    def test_the_header_is_a_single_line(self, fixture, tmp_path):
        """A stray \\r from the csv writer would make line 1 differ in a second way."""
        path, _ = naming.materialize(fixture, "opaque", tmp_path)
        first = path.read_text(encoding="utf-8").split("\n", 1)[0]
        assert "\r" not in first
        assert next(csv.reader([first])) == naming.header_of(path)


class TestTheRefusals:
    """A malformed fixture must fail loudly here, not produce a quietly wrong arm.

    Each of these is reachable from a hand-edited manifest or a half-written CSV, and each would
    otherwise surface as a rename that silently did the wrong thing -- which in this module means
    an ablation arm whose numbers look ordinary and mean nothing.
    """

    def test_an_empty_csv_has_no_header_to_rename(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="no header"):
            naming.header_of(empty)

    def test_a_target_that_is_not_a_column_is_refused(self, tmp_path):
        """The map is built around the target, so getting it wrong would rename the target and
        leave a feature un-renamed -- silently inverting the one column the arm holds fixed."""
        fixture = load_fixture("toy").model_copy(deep=True)
        fixture.manifest.target = "not_a_column"
        with pytest.raises(ValueError, match="not a column"):
            naming.rename_map(fixture, "opaque")

    def test_a_header_with_no_rows_under_it_is_refused(self, tmp_path):
        header_only = tmp_path / "toy.csv"
        header_only.write_text("a,b,churned")
        fixture = load_fixture("toy").model_copy(deep=True)
        fixture.manifest.target = "churned"
        fixture.csv_path = header_only
        with pytest.raises(ValueError, match="no rows"):
            naming.materialize(fixture, "opaque", tmp_path / "out")


class TestLineEndings:
    def test_crlf_survives_the_header_rewrite(self, tmp_path):
        """No committed fixture uses CRLF, but a `generate.py` run on another platform could
        introduce one. Dropping the \\r would terminate line 1 differently from every other line --
        a second way for the arms to differ -- and would fold it into the last column's name, so
        that column would never be renamed at all."""
        source = tmp_path / "toy.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            handle.write("id,feature,churned\r\n1,2,0\r\n2,3,1\r\n")
        fixture = load_fixture("toy").model_copy(deep=True)
        fixture.manifest.target = "churned"
        fixture.csv_path = source

        path, mapping = naming.materialize(fixture, "opaque", tmp_path / "out")

        assert mapping == {"id": "var_01", "feature": "var_02"}
        with path.open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        with source.open("r", encoding="utf-8", newline="") as handle:
            original = handle.read()
        assert text.startswith("var_01,var_02,churned\r\n")
        assert text.split("\r\n", 1)[1] == original.split("\r\n", 1)[1]
