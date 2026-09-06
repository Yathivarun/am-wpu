"""What this unit is running.

A fleet sweep of `--check --json` is only useful if it says which version each
device answered from — otherwise 50 identical reports cannot distinguish "all
up to date" from "half of them never got the update".

Three sources, most specific first:

1. `wpu_client/VERSION`, written by scripts/make_release.sh into the zip. This
   is what a deployed unit has, and it names the release exactly.
2. `git describe`, for a checkout. Carries the commit and a -dirty marker,
   which is what a development box actually wants to report.
3. The version in pyproject.toml metadata, as a floor.

None of these can fail the caller: an un-versioned unit reports "unknown"
rather than refusing to run a health check.
"""

import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent / "VERSION"
REPO_ROOT = Path(__file__).resolve().parent.parent

UNKNOWN = "unknown"


def _from_file() -> str | None:
    try:
        value = VERSION_FILE.read_text().strip()
    except OSError:
        return None
    return value or None


def _from_git() -> str | None:
    """`git describe` in a checkout. None in a release zip, which has no .git."""
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        done = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _from_metadata() -> str | None:
    try:
        return _pkg_version("wpu-client")
    except PackageNotFoundError:
        return None


def version() -> str:
    for source in (_from_file, _from_git, _from_metadata):
        value = source()
        if value:
            return value
    return UNKNOWN
