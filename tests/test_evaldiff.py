"""What breaks if this file's assertions stop holding.

`evaldiff.py` exists because a single 10-run count moved this project's headline number from a
retracted 5/10 -> 9/10 to a same-commit control reading 7/10 vs 6/10 -- the wrong sign. These
tests pin the two failure modes that produced that: calling a Wilson-interval overlap an effect
(the retired headline, reproduced synthetically below), and calling a single replicate evidence at
all. Every test here builds rows as plain dicts. Nothing runs the pipeline.
"""

import json
from typing import Any

import pytest

from ds_agents.evaldiff import (
    CONDITION_FIELDS,
    Count,
    cell_key,
    compare,
    load_rows,
    parse_metric,
    separating,
    tally,
    verdict,
    wilson,
)
from tests.conftest import RESULTS_DIR

pytestmark = pytest.mark.fast


def _row(**overrides: Any) -> dict[str, Any]:
    """A synthetic results row carrying a default value for every `CONDITION_FIELDS` entry.

    Real rows carry dozens more columns; only the condition fields and whichever metric a test
    names ever matter to this module, so the default is deliberately narrow.
    """
    row: dict[str, Any] = {
        "dataset_id": "claims_timing",
        "arm": "team",
        "reviewer_enabled": True,
        "default_model": "haiku",
        "reviewer_model": "haiku",
        "reviewer_prompt": "base",
        "reviewer_sees_code": True,
        "naming": "opaque",
        "loop_cap": 3,
        "objection_routing": "as_addressed",
        "objection_closure": "off",
        "forced_drop_release": "withdrawn_only",
        "random_seed": 20260822,
        "commit": "deadbeef",
    }
    row.update(overrides)
    return row


class TestWilson:
    """The interval this whole tool refuses to skip. A pooled count with no interval next to it is
    exactly the shape the retired headline had."""

    def test_nine_of_ten_matches_a_hand_computed_interval(self):
        lo, hi = wilson(9, 10)
        assert lo == pytest.approx(0.596, abs=1e-3)
        assert hi == pytest.approx(0.982, abs=1e-3)

    def test_zero_of_ten_stays_within_bounds(self):
        lo, hi = wilson(0, 10)
        assert 0.0 <= lo <= hi <= 1.0

    def test_ten_of_ten_stays_within_bounds(self):
        lo, hi = wilson(10, 10)
        assert 0.0 <= lo <= hi <= 1.0
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_zero_n_returns_the_full_interval(self):
        """No rows carry the metric at all -- the interval must say "unknown", not "zero"."""
        assert wilson(0, 0) == (0.0, 1.0)


class TestTheRetiredHeadline:
    """A 5/10 -> 9/10 result read as this project's largest measured effect, until a same-commit
    control came back the wrong sign. `separating()` on the two raw counts would have refused the
    claim for free: the intervals are [0.237, 0.763] and [0.596, 0.982], which overlap."""

    def test_five_of_ten_against_nine_of_ten_does_not_separate(self):
        before = [_row(leakage_remediated=v) for v in [True] * 5 + [False] * 5]
        after = [_row(leakage_remediated=v) for v in [True] * 9 + [False] * 1]
        before_count = tally(before, "leakage_remediated")
        after_count = tally(after, "leakage_remediated")

        assert before_count.interval == pytest.approx((0.237, 0.763), abs=1e-3)
        assert after_count.interval == pytest.approx((0.596, 0.982), abs=1e-3)
        assert not separating(before_count, after_count)


class TestSeparatingIsNotUselesslyConservative:
    def test_zero_of_ten_against_ten_of_ten_does_separate(self):
        before = [_row(leakage_remediated=False) for _ in range(10)]
        after = [_row(leakage_remediated=True) for _ in range(10)]
        assert separating(tally(before, "leakage_remediated"), tally(after, "leakage_remediated"))


class TestReplicatesGateThePooledInterval:
    """A Wilson interval assumes ten independent Bernoulli draws with one fixed success
    probability. That assumption is exactly what a second replicate at the same nominal
    configuration tests, and one replicate cannot test it -- so a single cell per arm is refused
    outright, however wide apart its counts land."""

    def test_a_single_replicate_is_underpowered_even_when_the_intervals_separate(self):
        before = [_row(leakage_remediated=False, replicate=0) for _ in range(10)]
        after = [_row(leakage_remediated=True, replicate=0) for _ in range(10)]
        before_count = tally(before, "leakage_remediated")
        after_count = tally(after, "leakage_remediated")

        # The intervals themselves are as disjoint as it gets -- this is not a power problem with
        # the statistic, it is a problem with having run the cell exactly once.
        assert separating(before_count, after_count)
        assert before_count.replicates == 1
        assert after_count.replicates == 1
        assert verdict(before_count, after_count) == "underpowered"

    def test_two_replicates_a_side_lets_a_real_gap_separate(self):
        before = [
            _row(leakage_remediated=v, replicate=r)
            for r, values in enumerate([[False] * 10, [False] * 10])
            for v in values
        ]
        after = [
            _row(leakage_remediated=v, replicate=r)
            for r, values in enumerate([[True] * 10, [True] * 10])
            for v in values
        ]
        before_count = tally(before, "leakage_remediated")
        after_count = tally(after, "leakage_remediated")

        assert before_count.replicates == 2
        assert after_count.replicates == 2
        assert verdict(before_count, after_count) == "separates"


class TestCellKeyGroupsByConfigNotByFile:
    def test_rows_from_different_files_with_the_same_config_group_together(self):
        """Nothing in `cell_key` reads `run_id` or any other per-row identity, so two rows that
        happen to differ only in which committed file they came from must still collide."""
        row_a = _row(run_id="from-file-a")
        row_b = _row(run_id="from-file-b")
        assert cell_key(row_a) == cell_key(row_b)

    def test_two_cells_differing_only_in_loop_cap_are_not_merged(self):
        """The opposite failure: collapsing a real condition into one bucket would average a
        loop_cap=1 run into a loop_cap=3 run and delete the loop-cap sweep's own finding."""
        row_a = _row(loop_cap=1)
        row_b = _row(loop_cap=3)
        assert cell_key(row_a) != cell_key(row_b)

    def test_a_row_missing_condition_fields_entirely_groups_with_none(self):
        """Every committed results file predates `commit`, `objection_routing`, and several other
        `CONDITION_FIELDS` entries. `cell_key` must fill the gap with `None` rather than raise, or
        the tool cannot even read its own archive."""
        old_style_row = {"dataset_id": "claims_timing", "arm": "team"}
        key = cell_key(old_style_row)
        assert len(key) == len(CONDITION_FIELDS)
        assert key[CONDITION_FIELDS.index("commit")] is None


class TestACellOnOneSideOnlyIsReportedNotDropped:
    """Silently dropping an unmatched cell is how an arm loses its control -- the same failure mode
    the retired headline's own control was built to avoid."""

    def test_a_cell_present_only_in_before_and_one_only_in_after_both_survive(self):
        before = [_row(loop_cap=1, leakage_remediated=True) for _ in range(3)]
        after = [_row(loop_cap=3, leakage_remediated=False) for _ in range(3)]

        comparisons = compare(before, after, metrics=("leakage_remediated",))

        assert len(comparisons) == 2
        sides = {c.only_in for c in comparisons}
        assert sides == {"before", "after"}
        for comparison in comparisons:
            metric = comparison.metrics[0]
            present = metric.before if comparison.only_in == "before" else metric.after
            missing = metric.after if comparison.only_in == "before" else metric.before
            assert present is not None
            assert present.n == 3
            assert missing is None


class TestNoneMetricValuesAreExcludedNotFailures:
    """The most important correctness rule in the file. `leakage_remediated` is `None` when
    `final_features` is empty -- a `feature_eng` crash, not a reviewer miss -- and counting that as
    False would make an upstream crash look like a downstream catch that got missed."""

    def test_none_values_drop_out_of_the_denominator_and_are_counted_as_excluded(self):
        rows = [_row(leakage_remediated=None) for _ in range(3)] + [
            _row(leakage_remediated=True) for _ in range(2)
        ]
        count = tally(rows, "leakage_remediated")

        assert count.n == 2
        assert count.successes == 2
        assert count.excluded == 3


class TestPerReplicateCountsSitBesideThePooledCount:
    def test_per_replicate_counts_match_the_pooled_total(self):
        rows = [_row(leakage_remediated=v, replicate=0) for v in [True, True, False]] + [
            _row(leakage_remediated=v, replicate=1) for v in [True, False, False]
        ]
        count = tally(rows, "leakage_remediated")

        assert count.successes == 3
        assert count.n == 6
        assert count.replicates == 2
        assert count.per_replicate == ((2, 3), (1, 3))

    def test_rows_carrying_no_replicate_field_report_zero_replicates(self):
        """Every row in the six committed results files. `()` here, not a single fake replicate
        standing in for the whole pooled count -- that would silently grant power the archive never
        earned."""
        rows = [_row(leakage_remediated=True) for _ in range(10)]
        count = tally(rows, "leakage_remediated")
        assert count.per_replicate == ()
        assert count.replicates == 0


class TestALeakageFlipIsFlaggedRegardlessOfPower:
    """A flip in `leakage_remediated` or `leakage_caught` is worth a second look even at n=1 --
    noise or not, a metric this project's headline outcome depends on reversing direction is not
    something a `verdict` of "underpowered" should let disappear silently."""

    def test_a_full_reversal_at_a_single_replicate_is_flagged(self):
        before = [_row(leakage_remediated=True)]
        after = [_row(leakage_remediated=False)]

        comparisons = compare(before, after, metrics=("leakage_remediated",))

        assert len(comparisons) == 1
        metric = comparisons[0].metrics[0]
        assert metric.verdict == "underpowered"
        assert metric.flipped is True

    def test_no_reversal_is_not_flagged(self):
        """5/10 vs 9/10 both read as majority-True: a difference in rate, not a flip in which
        outcome is more common, so it must not raise the same flag."""
        before = [_row(leakage_remediated=v) for v in [True] * 5 + [False] * 5]
        after = [_row(leakage_remediated=v) for v in [True] * 9 + [False] * 1]

        comparisons = compare(before, after, metrics=("leakage_remediated",))

        assert comparisons[0].metrics[0].flipped is False

    def test_a_non_leakage_metric_is_never_flagged_as_a_flip(self):
        """The rule names `leakage_remediated` and `leakage_caught` specifically. `errored`
        reversing direction is a reliability story, not the mechanism this flag exists for."""
        before = [_row(errored=True)]
        after = [_row(errored=False)]

        comparisons = compare(before, after, metrics=("errored",))

        assert comparisons[0].metrics[0].flipped is False


class TestTheCommittedResultsFilesStillParseAndGroup:
    """Ground truth per CLAUDE.md: every committed results file must still load and group, however
    much its schema has drifted from the current row shape."""

    def test_every_committed_file_loads_and_every_row_groups_without_raising(self):
        files = sorted(RESULTS_DIR.glob("*.jsonl"))
        assert files, "no committed results files found"

        for path in files:
            rows = load_rows(path)
            assert rows, f"{path} loaded no rows"
            for row in rows:
                key = cell_key(row)
                assert len(key) == len(CONDITION_FIELDS)

    def test_load_rows_never_writes_to_the_file_it_reads(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text(json.dumps(_row()) + "\n")
        before = path.read_bytes()
        load_rows(path)
        assert path.read_bytes() == before


class TestCount:
    def test_replicates_is_the_length_of_per_replicate(self):
        assert Count(successes=1, n=1, excluded=0, per_replicate=((1, 1),)).replicates == 1
        assert Count(successes=0, n=0, excluded=0).replicates == 0

    def test_interval_is_computed_from_successes_and_n(self):
        count = Count(successes=9, n=10, excluded=0)
        assert count.interval == pytest.approx(wilson(9, 10))


class TestRenderIsPlainText:
    """Not one of the eleven required cases, but cheap insurance: `render` is the only function a
    human reads directly, so it earns a smoke test that it never raises and never uses the words
    this tool is built to refuse."""

    def test_render_never_says_effect_or_p_value(self):
        from ds_agents.evaldiff import render

        before = [_row(leakage_remediated=v) for v in [True] * 5 + [False] * 5]
        after = [_row(leakage_remediated=v) for v in [True] * 9 + [False] * 1]
        text = render(compare(before, after, metrics=("leakage_remediated",)))

        assert "effect" not in text.lower()
        assert "p-value" not in text.lower() and "p value" not in text.lower()
        assert "underpowered" in text


class TestAMetricMayCarryANullPredicate:
    """`halted_at` names the node that ended a run and is null on every healthy one.

    Under rule 1 a bare `halted_at` therefore EXCLUDES every healthy run from the denominator and
    reports the halt rate over the halts alone -- 1/1 for a cell with one halt in ten runs. The
    predicate makes the null test itself the success test, and switches rule 1 off for that call
    because there is nothing left to exclude: every row has an answer.
    """

    def test_a_bare_column_still_obeys_rule_one(self):
        """Unchanged, and this is the point: `leakage_remediated` is what rule 1 exists for."""
        rows = [_row(leakage_remediated=v) for v in (True, False, None)]
        count = tally(rows, "leakage_remediated")
        assert (count.successes, count.n, count.excluded) == (1, 2, 1)

    def test_a_bare_halted_at_is_the_dead_end_the_predicate_exists_for(self):
        rows = [_row(halted_at=v) for v in ("profiler", None, None, None)]
        count = tally(rows, "halted_at")
        assert (count.successes, count.n, count.excluded) == (1, 1, 3)
        assert count.successes / count.n == 1.0, "a 25% halt rate reading as 100%"

    def test_notnull_counts_every_run_and_gets_the_rate_right(self):
        rows = [_row(halted_at=v) for v in ("profiler", None, None, None)]
        count = tally(rows, "halted_at:notnull")
        assert (count.successes, count.n, count.excluded) == (1, 4, 0)

    def test_isnull_is_the_complement(self):
        rows = [_row(halted_at=v) for v in ("profiler", None, None, None)]
        assert tally(rows, "halted_at:isnull").successes == 3

    def test_a_false_value_is_not_a_null_value(self):
        """The distinction the predicate turns on. `errored=False` is an observation of a healthy
        run; under `notnull` it is a success, because the question asked is whether the column has
        a value, not whether that value is truthy."""
        rows = [_row(errored=False) for _ in range(3)]
        assert tally(rows, "errored").successes == 0
        assert tally(rows, "errored:notnull").successes == 3

    def test_compare_accepts_a_predicate_metric_end_to_end(self):
        before = [_row(halted_at=None) for _ in range(4)]
        after = [_row(halted_at="profiler")] + [_row(halted_at=None) for _ in range(3)]
        [comparison] = compare(before, after, ("halted_at:notnull",))
        [metric] = comparison.metrics
        assert metric.metric == "halted_at:notnull"
        assert metric.before.n == 4 and metric.after.n == 4
        assert metric.before.successes == 0 and metric.after.successes == 1

    def test_a_flip_metric_is_still_recognised_through_its_predicate(self):
        assert parse_metric("leakage_caught:notnull")[0] == "leakage_caught"

    def test_an_unknown_predicate_raises_rather_than_reading_as_a_column_name(self):
        """A column that does not exist tallies as n=0 and renders as an empty row, so a typo would
        otherwise report 'no data' instead of an error."""
        with pytest.raises(ValueError, match="unknown metric predicate"):
            parse_metric("halted_at:notnul")

    def test_a_bare_spec_parses_to_no_predicate(self):
        assert parse_metric("errored") == ("errored", None)
