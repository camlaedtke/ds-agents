"""Which tree produced a run.

One function, called by `cli.py` and the harness and by nothing under `nodes/` -- a node that could
read the repository could read the answer key, which is the same rule that keeps `fixtures.py` out
of the graph. The value lands on the frozen `RunConfig` and from there on the results row.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit(root: Path = REPO_ROOT) -> str | None:
    """`c17a885`, or `c17a885-dirty` when the working tree had uncommitted changes.

    Returns `None` rather than raising when git is missing, slow, or this is not a checkout. A
    provenance helper that can take a benchmark run down is worse than a null field: the row would
    be lost entirely to record something that is only ever read afterwards.

    The `-dirty` suffix is part of the same string rather than a second boolean so the cell key
    stays one value wide, and so a dirty run groups as its own cell in `eval-diff`. That is the
    honest outcome -- a dirty run is not reproducible, and pooling it with the commit it almost
    matches would be a claim nobody can check.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        if not head:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - see the docstring: every failure here is the same failure.
        return None
    return f"{head}-dirty" if status else head
