from pathlib import Path

from wpu_client import paths


def test_project_root_is_repo_root():
    assert paths.PROJECT_ROOT == Path(__file__).resolve().parent.parent


def test_derived_dirs_are_under_project_root():
    for d in (paths.MODELS_DIR, paths.DATA_DIR, paths.CONFIG_DIR):
        assert d.parent == paths.PROJECT_ROOT


def test_derived_dirs_actually_exist():
    # These are tracked directories, not just a naming scheme.
    assert paths.MODELS_DIR.is_dir()
    assert paths.DATA_DIR.is_dir()
    assert paths.CONFIG_DIR.is_dir()
