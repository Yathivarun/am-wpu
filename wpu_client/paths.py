"""Project-root-relative path resolution, shared by services, tools, and scripts.

Resolved from this file's own location rather than the process's current working
directory, so it's correct regardless of where `python` is invoked from (as long
as the package is installed, editable or otherwise, from this repo checkout).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
