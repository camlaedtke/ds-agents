"""Compare two eval runs without letting sampling noise pass as a finding.

This exists because of one number. `leakage_remediated` moved 5/10 -> 9/10 across the sticky-drop
fix and was quoted as this project's largest effect for one session (DECISIONS.md 2026-08-28,
fourth entry). A same-commit control -- the same code, the same condition, run again -- came back
7/10 vs 6/10: the wrong sign. Once that control existed, four cells stood at the same nominal
configuration, and two of them running byte-identical behaviour returned 9/10 and 6/10. Model
nondeterminism alone moves a 10-run count on this benchmark by about 3. Every 10-run count this
project has published therefore carries error bars roughly +/-0.25 wide on the underlying rate, and
a single cell per arm cannot resolve a 4/10 difference -- it never could have. This module's job is
to refuse to call such a difference an effect.

A pooled Wilson interval alone would already have caught the retired headline for free, before a
single one of the runs that eventually falsified it: 5/10 is [0.237, 0.763] and 9/10 is
[0.596, 0.982], and those overlap on [0.596, 0.763]. So why also require two replicates before
computing one? Because a Wilson interval assumes the 10 runs behind a count are independent
Bernoulli draws with one fixed success probability, and that assumption is exactly what running the
same cell twice tests. A model whose behaviour drifts, a fixture whose difficulty depends on which
split it drew, or a bug that fires in bursts would all break the assumption invisibly, and a broken
assumption makes the interval a lie rather than merely wide. The replicate requirement is not about
shrinking the interval -- it is about earning the right to compute one at all. That is why `Count`
carries `per_replicate` printed beside the pooled count rather than folded into it: a reader can see
whether the replicates agree with each other before trusting what their sum implies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONDITION_FIELDS = (
    "dataset_id",
    "arm",
    "reviewer_enabled",
    "default_model",
    "reviewer_model",
    "reviewer_prompt",
    "reviewer_sees_code",
    "naming",
    "loop_cap",
    "objection_routing",
    "objection_closure",
    "forced_drop_release",
    "random_seed",
    "commit",
)

DEFAULT_METRICS = ("leakage_remediated", "leakage_caught", "reviewer_caught", "errored")

CONTINUOUS_COLUMNS = ("cost_usd", "review_loops", "n_final_features")

# The two metrics a direction flip is worth flagging on regardless of power (rule 5). Not every
# boolean column: `errored` reversing direction is a reliability story, and folding it in here
# would dilute the one signal this rule exists to surface, which is the caught-vs-remediated
# mechanism this project's whole thesis rests on.
_FLIP_METRICS = frozenset({"leakage_remediated", "leakage_caught"})

PREDICATES = ("notnull", "isnull")
"""The suffixes a metric spec may carry, as `column:notnull`.

For columns where `None` IS the observation rather than a missing one. `halted_at` is the case that
forced this: it names the node that ended a run and is `None` on every healthy one, so under rule 1
a bare `halted_at` excludes every healthy run from the denominator and reports the halt rate as
100% of however many halts there were. `errored` reaches the same dead end from the other side --
it is never null, so it can only ever be tallied over its truthiness, and a reader who wants "how
many runs halted" cannot ask for it.

A predicate turns the null test itself into the success test and switches rule 1 OFF, because there
is nothing left to exclude: every row has an answer. The bare `column` form is untouched, so rule 1
and its docstring stay exactly true for `leakage_remediated`, which is what rule 1 exists for.
"""


def parse_metric(spec: str) -> tuple[str, str | None]:
    """Split `column:predicate` into its parts. A bare column returns `(column, None)`.

    Raises on an unknown predicate rather than treating the whole spec as a column name: a column
    that does not exist tallies as n=0 and renders as an empty row, so a typo like
    `halted_at:notnul` would silently produce "no data" instead of an error.
    """
    if ":" not in spec:
        return spec, None
    column, _, predicate = spec.partition(":")
    if predicate not in PREDICATES:
        raise ValueError(
            f"unknown metric predicate {predicate!r} in {spec!r}; "
            f"expected one of {', '.join(PREDICATES)}"
        )
    return column, predicate


def load_rows(path: Path) -> list[dict[str, Any]]:
    """One JSON object per line. Read-only, always -- `evals/results/` is ground truth per
    CLAUDE.md, and this function has no write path because nothing here should ever need one."""
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """The run condition a row belongs to, as a hashable tuple over `CONDITION_FIELDS`.

    `row.get(field)` rather than `row[field]`: every committed results file predates several of
    these columns (`commit`, `objection_routing`, `objection_closure`, `forced_drop_release`,
    `default_model` did not exist when the first files were written), so a row written before a
    field existed reads as `None` on that field instead of raising `KeyError`. That is also why a
    row missing a field and a row explicitly recording its default are indistinguishable here --
    which is the honest answer, since the earlier row genuinely ran under whatever the code did at
    the time, and this module has no way to know if that matched today's default.
    """
    return tuple(row.get(field) for field in CONDITION_FIELDS)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A Wilson score interval for a binomial proportion. `z=1.96` is the 95% interval.

    `n == 0` returns `(0.0, 1.0)` -- the full range, meaning "unknown" -- rather than raising or
    returning `(0.0, 0.0)`. A metric nobody measured is not evidence of a 0% rate.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = phat + z2 / (2 * n)
    adjustment = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lo = (center - adjustment) / denom
    hi = (center + adjustment) / denom
    # Floating point can push a boundary case a hair outside [0, 1] (10/10, 0/10); clip rather
    # than let a bound like 1.0000000000000002 read as "wider than certain".
    return (max(0.0, lo), min(1.0, hi))


@dataclass(frozen=True)
class Count:
    """One metric, tallied over one cell.

    `n` is rows where the metric is not `None`; `excluded` is rows where it was. See `tally` for
    why that split exists -- it is rule 1, the most important correctness rule in this file.
    `per_replicate` is `()` when no row in the cell carries a `replicate` field, which is every row
    in every results file committed before this module existed. A `Count` with zero replicates is
    not zero evidence -- `n` may be large -- but it is evidence that has never been asked whether
    it replicates, which is exactly what `verdict` refuses to treat as power.
    """

    successes: int
    n: int
    excluded: int
    per_replicate: tuple[tuple[int, int], ...] = ()

    @property
    def replicates(self) -> int:
        return len(self.per_replicate)

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.successes, self.n)


def tally(rows: Sequence[dict[str, Any]], metric: str) -> Count:
    """Count successes for `metric` over `rows`, honouring the None-exclusion rule.

    Rule 1, applied here and nowhere else: a `None` metric value is EXCLUDED from the denominator,
    never counted as a failure. `leakage_remediated` reads `None` when `final_features` is empty --
    a `feature_eng` crash, not a reviewer miss -- and scoring that as False would make an upstream
    crash look like a downstream catch that got missed, silently inflating every "the reviewer
    failed" count with runs the reviewer never got to see. Excluded rows are reported on `Count`,
    never silently dropped, so a cell whose denominator shrank is visible rather than assumed clean.

    `metric` may carry a predicate -- `halted_at:notnull` -- which replaces the truthiness test and
    suspends rule 1 for that call only. See `PREDICATES` for why that is not a violation of it.
    """
    column, predicate = parse_metric(metric)
    successes = 0
    n = 0
    excluded = 0
    saw_replicate_field = False
    by_replicate: dict[Any, list[int]] = {}
    for row in rows:
        replicate = row.get("replicate")
        if replicate is not None:
            saw_replicate_field = True
        value = row.get(column)
        if predicate is None:
            if value is None:
                excluded += 1
                continue
            success = 1 if value else 0
        else:
            success = 1 if ((value is not None) == (predicate == "notnull")) else 0
        n += 1
        successes += success
        bucket = by_replicate.setdefault(replicate, [0, 0])
        bucket[0] += success
        bucket[1] += 1

    per_replicate: tuple[tuple[int, int], ...] = ()
    if saw_replicate_field:
        # Sort key guards against mixing a real replicate id with rows that had none (replicate is
        # None): comparing None to an int raises in Python 3, and a row missing the field must not
        # crash the sort of a cell where every other row has it.
        ordered = sorted(by_replicate.items(), key=lambda item: (item[0] is None, item[0]))
        per_replicate = tuple((s, n_) for _, (s, n_) in ordered)

    return Count(successes=successes, n=n, excluded=excluded, per_replicate=per_replicate)


def separating(a: Count, b: Count, z: float = 1.96) -> bool:
    """Whether `a` and `b`'s Wilson intervals are disjoint.

    Rule 3 and rule 4 in one function: overlapping intervals are "not separating" and disjoint ones
    "separate at this n" -- but this function only ever answers the interval question. It does not
    know about replicates, on purpose: `verdict` is where power gates the answer, so that
    `separating` stays usable on its own for exactly the case the module docstring opens with --
    checking whether a headline could have been refused before a single replicate was run.
    """
    a_lo, a_hi = wilson(a.successes, a.n, z)
    b_lo, b_hi = wilson(b.successes, b.n, z)
    return a_hi < b_lo or b_hi < a_lo


def verdict(a: Count, b: Count) -> str:
    """ "underpowered" | "not separating" | "separates", in that priority order.

    Rule 2: underpowered beats everything. If EITHER side has fewer than 2 replicates, the answer
    is "underpowered" no matter how far apart the counts land -- which is every row in the six
    committed results files, since none of them carry a `replicate` field at all. This is what
    stops `separating()` alone from being the whole tool: the retired headline's own counts (5/10,
    9/10) do NOT separate, but a pair that DID separate at one replicate a side would still be
    refused here, because one replicate cannot test the iid-Bernoulli assumption the interval
    depends on. Only past that gate does rule 3 ("not separating") or rule 4 ("separates") apply.
    """
    if a.replicates < 2 or b.replicates < 2:
        return "underpowered"
    if separating(a, b):
        return "separates"
    return "not separating"


@dataclass(frozen=True)
class MetricComparison:
    """One metric's before/after picture within one cell.

    `before` or `after` is `None` exactly when that side of the cell has no rows at all (rule 6):
    `verdict` then reads `"only in before"` / `"only in after"` instead of one of the three power
    states, because there is nothing on the missing side to compare against. `flipped` is computed
    for `leakage_remediated` and `leakage_caught` only (rule 5) and only when both sides are
    present -- a side with nothing to compare against cannot have reversed direction.
    """

    metric: str
    before: Count | None
    after: Count | None
    verdict: str
    flipped: bool


@dataclass(frozen=True)
class Comparison:
    """Everything one run condition (`cell_key`) has to say, before and after."""

    key: tuple[Any, ...]
    only_in: str | None  # "before" | "after" | None (present on both sides)
    metrics: tuple[MetricComparison, ...]
    continuous: tuple[tuple[str, float | None, float | None], ...]


def _group_by_cell(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(cell_key(row), []).append(row)
    return groups


def _mean(rows: list[dict[str, Any]] | None, column: str) -> float | None:
    if not rows:
        return None
    values = [row[column] for row in rows if row.get(column) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _majority(count: Count) -> bool | None:
    """Whether True or False is the more common outcome in `count`. `None` when `n == 0`, so an
    empty side cannot be read as "majority False"."""
    if count.n == 0:
        return None
    return count.successes * 2 >= count.n


def _direction_flipped(before: Count, after: Count) -> bool:
    before_majority = _majority(before)
    after_majority = _majority(after)
    if before_majority is None or after_majority is None:
        return False
    return before_majority != after_majority


def _sort_token(value: Any) -> tuple[int, str]:
    # `None` sorts first, everything else by its string form -- CONDITION_FIELDS mixes bools,
    # ints, and strings, and Python 3 cannot compare `None` against any of them directly.
    return (0, "") if value is None else (1, str(value))


def _sortable_key(key: tuple[Any, ...]) -> tuple[tuple[int, str], ...]:
    return tuple(_sort_token(value) for value in key)


def compare(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[Comparison]:
    """Compare two sets of results rows, one `Comparison` per distinct `cell_key` in either set.

    Every cell that appears in `before`, `after`, or both is reported (rule 6) -- a cell present on
    only one side is exactly how an arm loses its control, which is the failure the same-commit
    control in DECISIONS.md 2026-08-28 (fifth entry) exists to avoid repeating. Cells are ordered
    deterministically so `render()` output is stable across runs with the same input.
    """
    before_by_cell = _group_by_cell(before)
    after_by_cell = _group_by_cell(after)
    keys = sorted(set(before_by_cell) | set(after_by_cell), key=_sortable_key)

    comparisons: list[Comparison] = []
    for key in keys:
        before_rows = before_by_cell.get(key)
        after_rows = after_by_cell.get(key)
        if before_rows is None:
            only_in = "after"
        elif after_rows is None:
            only_in = "before"
        else:
            only_in = None

        metric_comparisons = []
        for metric in metrics:
            before_count = tally(before_rows, metric) if before_rows is not None else None
            after_count = tally(after_rows, metric) if after_rows is not None else None
            if before_count is not None and after_count is not None:
                metric_verdict = verdict(before_count, after_count)
                flipped = parse_metric(metric)[0] in _FLIP_METRICS and _direction_flipped(
                    before_count, after_count
                )
            else:
                metric_verdict = "only in after" if before_count is None else "only in before"
                flipped = False
            metric_comparisons.append(
                MetricComparison(metric, before_count, after_count, metric_verdict, flipped)
            )

        continuous = tuple(
            (column, _mean(before_rows, column), _mean(after_rows, column))
            for column in CONTINUOUS_COLUMNS
        )
        comparisons.append(Comparison(key, only_in, tuple(metric_comparisons), continuous))
    return comparisons


def _format_key(key: tuple[Any, ...]) -> str:
    parts = [
        f"{field}={value}"
        for field, value in zip(CONDITION_FIELDS, key, strict=True)
        if value is not None
    ]
    return "CELL " + " ".join(parts) if parts else "CELL <no condition fields recorded>"


def _format_count(count: Count) -> str:
    lo, hi = count.interval
    text = f"{count.successes}/{count.n} [{lo:.3f}, {hi:.3f}]"
    if count.excluded:
        text += f" ({count.excluded} excluded)"
    if count.replicates:
        per_replicate = ", ".join(f"{s}/{n}" for s, n in count.per_replicate)
        text += f" -- {count.replicates} replicates: {per_replicate}"
    return text


def _format_verdict(metric_verdict: str) -> str:
    # Rule 4: the word is "separates at this n", never "effect" and never a p-value -- both of
    # which is exactly the vocabulary the retired headline was described in.
    return "separates at this n" if metric_verdict == "separates" else metric_verdict


def _format_metric(mc: MetricComparison) -> str:
    if mc.before is None or mc.after is None:
        present = mc.before if mc.before is not None else mc.after
        side = "before" if mc.before is not None else "after"
        assert present is not None
        return f"  {mc.metric}: only in {side} -- {_format_count(present)}"
    flag = "  [FLIP]" if mc.flipped else ""
    return (
        f"  {mc.metric}: {_format_verdict(mc.verdict)}{flag}\n"
        f"    before  {_format_count(mc.before)}\n"
        f"    after   {_format_count(mc.after)}"
    )


def _format_continuous(column: str, before_mean: float | None, after_mean: float | None) -> str:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4g}"

    return f"  {column}: before={fmt(before_mean)} after={fmt(after_mean)}"


def render(comparisons: list[Comparison]) -> str:
    """Plain text for a terminal. One block per cell, minimal and scannable.

    No summary statistic rolls cells together, and nothing here computes or prints a p-value: the
    whole point of this module is that a count out of 10 is not evidence on its own, and a single
    combined score would recreate exactly the illusion of precision the retired headline had.
    """
    lines: list[str] = []
    for comparison in comparisons:
        lines.append(_format_key(comparison.key))
        if comparison.only_in is not None:
            lines.append(f"  only in {comparison.only_in} -- not compared")
        for metric_comparison in comparison.metrics:
            lines.append(_format_metric(metric_comparison))
        for column, before_mean, after_mean in comparison.continuous:
            if before_mean is None and after_mean is None:
                continue
            lines.append(_format_continuous(column, before_mean, after_mean))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
