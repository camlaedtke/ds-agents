"""`provenance.git_commit` and what it does when git does not answer.

The behaviour under test was bought by a real run: `evals/results/2026-09-21_claims-repro.jsonl`
is ten rows produced at a clean `46bd4ed` that carry `commit: null`, because bare `git` on that
machine is `/usr/bin/git` under an unaccepted Xcode license and `git_commit` swallowed the
refusal. Returning `None` is correct -- a provenance helper must not take a benchmark run down --
and doing it quietly is the defect.
"""

import subprocess
import sys

import pytest

from ds_agents import provenance
from ds_agents.provenance import GIT_ENV_VAR, describe_commit, git_commit

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def _reset_warning_state(monkeypatch):
    """The warning is once per process, so a test that asserts on it must start from unwarned."""
    monkeypatch.setattr(provenance, "_warned", False)


class TestWhichGitIsCalled:
    def test_the_env_var_overrides_the_git_on_path(self, monkeypatch):
        """The override exists so a session on a machine with a broken system git can record
        provenance without `sudo`. Without it the only fix is to the machine."""
        seen: list[list[str]] = []

        outputs = iter(["abc1234\n", "\n"])

        def fake_run(argv, **kwargs):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout=next(outputs), stderr="")

        monkeypatch.setenv(GIT_ENV_VAR, "/opt/homebrew/bin/git")
        monkeypatch.setattr(subprocess, "run", fake_run)

        assert git_commit() == "abc1234"
        assert all(argv[0] == "/opt/homebrew/bin/git" for argv in seen)

    def test_bare_git_is_the_default(self, monkeypatch):
        monkeypatch.delenv(GIT_ENV_VAR, raising=False)
        outputs = iter(["abc1234\n", "\n"])
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(
                argv, 0, stdout=next(outputs), stderr=""
            ),
        )

        assert provenance._git_executable() == "git"
        assert git_commit() == "abc1234"

    def test_an_empty_env_var_falls_back_rather_than_calling_the_empty_string(self, monkeypatch):
        monkeypatch.setenv(GIT_ENV_VAR, "")

        assert provenance._git_executable() == "git"


class TestTheFailureIsReported:
    """`describe_commit` keeps the reason; `git_commit` prints it once and returns `None`."""

    def _refuse(self, monkeypatch, stderr: str, returncode: int = 69):
        def fake_run(argv, **kwargs):
            raise subprocess.CalledProcessError(returncode, argv, output="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_the_xcode_license_refusal_is_carried_out_as_the_reason(self, monkeypatch):
        """The exact failure that cost a run its provenance. git exits non-zero with the notice on
        stderr, which is not a git error and reads like one in a log."""
        self._refuse(monkeypatch, "You have not agreed to the Xcode license agreements.")

        commit, reason = describe_commit()

        assert commit is None
        assert "exited 69" in reason
        assert "Xcode license" in reason

    def test_a_missing_git_says_so(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
        )

        commit, reason = describe_commit()

        assert commit is None
        assert "not found" in reason

    def test_an_unanticipated_failure_still_degrades_to_a_reason(self, monkeypatch):
        """The catch-all is what holds the contract, not the named branches above it.

        `subprocess.run(text=True)` decodes git's output, so a non-UTF-8 byte in a branch name or
        in a stderr notice raises `UnicodeDecodeError` -- a `ValueError`, which a
        subprocess-flavoured except list does not catch. An earlier revision of this module listed
        only the four expected exceptions and would have raised here.
        """

        def fake_run(argv, **kwargs):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        monkeypatch.setattr(subprocess, "run", fake_run)

        commit, reason = describe_commit()

        assert commit is None
        assert "UnicodeDecodeError" in reason

    def test_a_permission_error_is_a_reason_not_a_crash(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(subprocess, "run", fake_run)

        commit, reason = describe_commit()

        assert commit is None
        assert "PermissionError" in reason

    def test_a_slow_git_says_so_rather_than_hanging_the_run(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 5)

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert describe_commit() == (None, "git timed out after 5s")

    def test_git_commit_still_returns_none_and_never_raises(self, monkeypatch, capsys):
        """The contract that matters most: a run must not die for want of a provenance string."""
        self._refuse(monkeypatch, "You have not agreed to the Xcode license agreements.")

        assert git_commit() is None

    def test_the_failure_reaches_stderr_and_names_the_consequence(self, monkeypatch, capsys):
        """A null `commit` fragments an eval-diff cell, which is a thing the operator can only act
        on before spending the money. Saying "no commit" without saying so is half a warning."""
        self._refuse(monkeypatch, "You have not agreed to the Xcode license agreements.")

        git_commit()

        err = capsys.readouterr().err
        assert "no commit recorded" in err
        assert "Xcode license" in err
        assert "commit=null" in err
        assert GIT_ENV_VAR in err

    def test_it_warns_once_per_process_not_once_per_call(self, monkeypatch, capsys):
        """`cli._run_once` takes `commit` as a keyword for exactly this reason, but a 52-run
        invocation that found a second caller would otherwise print 52 identical lines."""
        self._refuse(monkeypatch, "nope")

        for _ in range(5):
            git_commit()

        assert capsys.readouterr().err.count("no commit recorded") == 1

    def test_warn_false_stays_quiet(self, monkeypatch, capsys):
        self._refuse(monkeypatch, "nope")

        assert git_commit(warn=False) is None
        assert capsys.readouterr().err == ""

    def test_nothing_is_printed_when_git_answers(self, monkeypatch, capsys):
        outputs = iter(["abc1234\n", "\n"])
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(
                argv, 0, stdout=next(outputs), stderr=""
            ),
        )

        assert git_commit() == "abc1234"
        assert capsys.readouterr().err == ""


class TestTheWarningCannotItselfEndARun:
    """The logging added to fix a silent failure must not become a loud one."""

    def test_a_broken_stderr_is_swallowed(self, monkeypatch):
        import io

        class Broken(io.StringIO):
            def write(self, _):
                raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
        )
        monkeypatch.setattr(sys, "stderr", Broken())

        assert git_commit() is None

    def test_a_closed_stderr_is_swallowed(self, monkeypatch):
        import io

        closed = io.StringIO()
        closed.close()
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
        )
        monkeypatch.setattr(sys, "stderr", closed)

        assert git_commit() is None


class TestTheDirtySuffix:
    def test_a_dirty_tree_is_one_string_not_a_second_field(self, monkeypatch):
        """`commit` is an eval-diff condition field and a dirty run is not reproducible, so it has
        to group as its own cell. One value wide is what makes that automatic."""
        outputs = iter(["abc1234\n", " M src/ds_agents/harness.py\n"])
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(
                argv, 0, stdout=next(outputs), stderr=""
            ),
        )

        assert git_commit() == "abc1234-dirty"

    def test_a_clean_tree_has_no_suffix(self, monkeypatch):
        outputs = iter(["abc1234\n", "\n"])
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(
                argv, 0, stdout=next(outputs), stderr=""
            ),
        )

        assert git_commit() == "abc1234"

    def test_an_empty_head_is_a_failure_with_a_reason(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="\n", stderr=""),
        )

        commit, reason = describe_commit()

        assert commit is None
        assert "printed nothing" in reason
