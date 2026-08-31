"""Name transparency as a run condition: the same rows under descriptive or opaque headers.

Trap difficulty in this project turned out to be set mostly by what the fixture author called a
column. Two fixtures were tuned until the planted column's association with the target was
indistinguishable from a legitimate strong feature, and the profiler nominated the planted one
anyway; renaming the traps defeated it far more effectively than any amount of association tuning
did. That makes the column name an uncontrolled variable sitting underneath every leakage number,
and the fix is to make it a condition that gets recorded rather than a property of the fixture.

The rename is applied here, above the tools boundary, rather than declared on the manifest or
shipped as a second fixture. Both alternatives were considered and both change the rows: a paired
fixture is a different CSV with a different seed's worth of noise, and a manifest field would mean
maintaining two committed CSVs that are supposed to be identical and will eventually not be. What
this module does instead is rewrite one line of one file. `materialize` copies the CSV body through
byte for byte and replaces only the header, so the two arms of the ablation differ in the header row
and provably nowhere else -- which is the only shape in which the difference between the arms can be
attributed to the names.

Deliberately not in `fixtures.py`: that module commits to never reading the CSV, and building the
map needs the header. `claims_timing` has a column no manifest field declares, so the manifest is
not a complete list of columns and could not be the source. The rule that matters is unchanged
downstream -- nothing in `nodes/` imports this module either, because a node that could ask which
arm it was running in could answer the question the arm exists to pose.
"""

import csv
import io
from pathlib import Path
from typing import Literal

from ds_agents.runnable import Runnable

Naming = Literal["descriptive", "opaque"]
NAMINGS: tuple[Naming, ...] = ("descriptive", "opaque")

OPAQUE_PREFIX = "var"


def header_of(csv_path: Path) -> list[str]:
    """The column names, and nothing else. Reads one line, not the file."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            return row
    raise ValueError(f"{csv_path} is empty, so it has no header to rename")


def rename_map(runnable: Runnable, naming: Naming) -> dict[str, str]:
    """Old name -> new name. Empty under `descriptive`, which is the identity condition.

    Every column is renamed except the target. The target has to keep its name because intake is
    given prose -- "Predict claim_denied and report roc_auc" -- and inferring the target from that
    sentence is the node's actual job; renaming it would change the task rather than the condition.

    Renaming *everything* else, rather than only the planted columns, is the point. An earlier
    ad hoc version renamed the traps alone, which makes an opaque name the thing only traps have
    and hands the profiler a cue in place of the one it lost.

    Numbering is dense and follows the original column order, so it leaks nothing about which
    columns were skipped and nothing about which are planted.
    """
    if naming == "descriptive":
        return {}
    target = runnable.target
    columns = header_of(runnable.csv_path)
    if target not in columns:
        raise ValueError(
            f"{runnable.dataset_id}: target {target!r} is not a column in {runnable.csv_path.name}"
        )
    renamed = [column for column in columns if column != target]
    return {column: f"{OPAQUE_PREFIX}_{index:02d}" for index, column in enumerate(renamed, start=1)}


def apply(names: list[str], mapping: dict[str, str]) -> list[str]:
    """Rename a list of column names, leaving anything unmapped alone."""
    return [mapping.get(name, name) for name in names]


def materialize(runnable: Runnable, naming: Naming, into: Path) -> tuple[Path, dict[str, str]]:
    """The CSV this run's agents should see, plus the map that produced it.

    Under `descriptive` this returns the source CSV's own path and writes nothing: a copy
    that could differ from the checked-in file is a way for the control arm to drift, and there is
    no reason to take it.

    Under `opaque` the header line is replaced and every remaining byte is copied through
    unchanged. Not a pandas round trip: reading and rewriting the frame would re-format floats and
    re-quote strings, and then "the arms differ only in the header" would be a claim about pandas
    rather than a fact about the file.
    """
    if naming == "descriptive":
        return runnable.csv_path, {}

    mapping = rename_map(runnable, naming)
    # `newline=""` on both sides disables universal-newline translation. Without it Python reads
    # \r\n as \n and writes \n back, which silently rewrites the line ending of every row in the
    # file -- the body would differ from the original in a way this module exists to prevent, and
    # the byte-comparison test would be comparing two already-normalised strings and pass anyway.
    with runnable.csv_path.open("r", encoding="utf-8", newline="") as handle:
        raw = handle.read()
    header, newline, body = raw.partition("\n")
    if not newline:
        raise ValueError(f"{runnable.csv_path} has no rows under its header")

    # Whatever line ending the file already uses, the rewritten header keeps it. Dropping a \r
    # here would leave line 1 terminated differently from every other line, which is a second way
    # for the arms to differ and would also mis-key the last column's rename -- `csv.reader` would
    # read the trailing \r as part of that column's name and the map would miss it.
    carriage, header = ("\r", header[:-1]) if header.endswith("\r") else ("", header)

    buffer = io.StringIO()
    # `lineterminator` because csv writes \r\n by default, which would make the one line we rewrite
    # differ from the rest of the file in a second way.
    csv.writer(buffer, lineterminator="").writerow(apply(next(csv.reader([header])), mapping))

    into.mkdir(parents=True, exist_ok=True)
    destination = into / runnable.csv_path.name
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue() + carriage + newline + body)
    return destination, mapping
