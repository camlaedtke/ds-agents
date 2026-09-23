"""Which tree produced a run.

Called by `cli.py` and the harness and by nothing under `nodes/` -- a node that could
read the repository could read the answer key, which is the same rule that keeps `fixtures.py` out
of the graph. The value lands on the frozen `RunConfig` and from there on the results row.
"""

import contextlib
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes on disk, hex-encoded. The one implementation every caller that
    hashes a CSV or a manifest file shares, so a change to how the digest is taken -- chunked
    reading, say -- only has to happen once."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


GIT_ENV_VAR = "DS_AGENTS_GIT"
"""Which git to shell out to, when the one on PATH is not the one that works.

Set to an absolute path. This exists because a real run lost its provenance to it: on a macOS
machine with an unaccepted Xcode license, bare `git` resolves to `/usr/bin/git`, which answers
every invocation with a license notice on stderr and a non-zero exit. `git_commit` did what it
promises and returned `None`, and ten benchmark rows produced at a clean `46bd4ed` recorded no
commit at all. An override is the fix a session can apply to itself; `sudo xcodebuild -license` is
the fix to the machine.
"""

_GIT_TIMEOUT_S = 5
"""Seconds either git call may take. Named because it appears in both calls and in the message
that reports the timeout, and three copies of a 5 are three chances to desync."""

_warned = False


def _git_executable() -> str:
    return os.environ.get(GIT_ENV_VAR) or "git"


def describe_commit(root: Path = REPO_ROOT) -> tuple[str | None, str]:
    """`(commit, reason)`. `reason` is empty exactly when `commit` is not `None`.

    The same work `git_commit` does, with the failure kept instead of discarded. Split out because
    "no commit" and "why there is no commit" are different facts and only one of them was being
    recorded -- see `GIT_ENV_VAR` for the run that paid for this.
    """
    git = _git_executable()
    try:
        head = subprocess.run(
            [git, "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        ).stdout.strip()
        if not head:
            return None, f"{git} rev-parse printed nothing"
        status = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        return None, f"{git} exited {exc.returncode}: {detail[0] if detail else 'no stderr'}"
    except FileNotFoundError:
        return None, f"{git} not found"
    except subprocess.TimeoutExpired:
        return None, f"{git} timed out after {_GIT_TIMEOUT_S}s"
    except Exception as exc:  # noqa: BLE001 - deliberate, see below.
        # The four cases above exist to give a useful reason, not to enumerate what can go wrong.
        # This catch-all is what actually holds the contract, and narrowing it to the anticipated
        # exceptions would have quietly given that up: `text=True` decodes git's output, so a
        # non-UTF-8 byte in a branch name or a stderr notice raises `UnicodeDecodeError`, which is
        # a `ValueError` and would have escaped a subprocess-flavoured except list.
        return None, f"{git} could not be run: {type(exc).__name__}: {exc}"
    return (f"{head}-dirty" if status else head), ""


def git_commit(root: Path = REPO_ROOT, *, warn: bool = True) -> str | None:
    """`c17a885`, or `c17a885-dirty` when the working tree had uncommitted changes.

    Returns `None` rather than raising when git is missing, slow, or this is not a checkout. A
    provenance helper that can take a benchmark run down is worse than a null field: the row would
    be lost entirely to record something that is only ever read afterwards.

    It now says so on stderr, once per process, rather than returning `None` silently. Silence is
    what let a run set up specifically to close a provenance gap write ten rows with no commit on
    them and report success. `warn=False` is for tests and for callers that print their own.

    The `-dirty` suffix is part of the same string rather than a second boolean so the cell key
    stays one value wide, and so a dirty run groups as its own cell in `eval-diff`. That is the
    honest outcome -- a dirty run is not reproducible, and pooling it with the commit it almost
    matches would be a claim nobody can check.
    """
    global _warned
    commit, reason = describe_commit(root)
    if commit is None and warn and not _warned:
        _warned = True
        warn_to_stderr(
            f"provenance: no commit recorded -- {reason}. Every row this process writes will "
            f"carry commit=null, and `commit` is an eval-diff condition field. Set "
            f"{GIT_ENV_VAR} to a git that works."
        )
    return commit


def warn_to_stderr(message: str) -> None:
    """Print to stderr, or give up quietly if stderr will not take it.

    The whole point of this module is that recording provenance must not be able to end a run. A
    bare `print` reintroduces exactly that: stderr can be closed or its reader gone, and then the
    logging added to fix a silent failure becomes a loud one. `BrokenPipeError` is an `OSError`;
    a closed file object raises `ValueError`.
    """
    with contextlib.suppress(OSError, ValueError):
        print(message, file=sys.stderr)
