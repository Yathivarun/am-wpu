"""Build identification.

A fleet sweep of `--check --json` is only worth running if each report says
which build answered it. The sources are ordered so a deployed unit reports
its release and a checkout reports its commit — and so neither can fail a
health check that exists to run in broken environments.
"""

from wpu_client import version as version_mod
from wpu_client.version import UNKNOWN, version


def _isolate(monkeypatch, tmp_path, file_text=None, git=None, meta=None):
    version_file = tmp_path / "VERSION"
    if file_text is not None:
        version_file.write_text(file_text)
    monkeypatch.setattr(version_mod, "VERSION_FILE", version_file)
    monkeypatch.setattr(version_mod, "_from_git", lambda: git)
    monkeypatch.setattr(version_mod, "_from_metadata", lambda: meta)


def test_release_file_wins(monkeypatch, tmp_path):
    """What a deployed unit has: make_release.sh writes it into the zip."""
    _isolate(monkeypatch, tmp_path, file_text="v1.2.0\n", git="v1.1.0-3-gabc", meta="0.1.0")

    assert version() == "v1.2.0"


def test_git_describe_is_used_in_a_checkout(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, git="v1.1.0-35-g7ac64bd-dirty", meta="0.1.0")

    assert version() == "v1.1.0-35-g7ac64bd-dirty"


def test_metadata_is_the_floor(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, meta="0.1.0")

    assert version() == "0.1.0"


def test_unknown_rather_than_an_error(monkeypatch, tmp_path):
    """An un-versioned unit still runs its health check."""
    _isolate(monkeypatch, tmp_path)

    assert version() == UNKNOWN


def test_blank_version_file_falls_through(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, file_text="\n", git="v1.1.0")

    assert version() == "v1.1.0"


def test_git_is_not_consulted_outside_a_checkout(monkeypatch, tmp_path):
    """A release zip has no .git; _from_git must not shell out looking for one."""
    monkeypatch.setattr(version_mod, "REPO_ROOT", tmp_path)
    called = []
    monkeypatch.setattr(version_mod.subprocess, "run", lambda *a, **k: called.append(a))

    assert version_mod._from_git() is None
    assert not called
