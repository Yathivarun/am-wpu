"""Shared test fixtures.

`slideshow_service.py` does `import gi; gi.require_version(...); from
gi.repository import Gtk, ...` at module scope — real GTK4 typelibs aren't
installed in CI (no GUI stack), so we stub `gi` before any test imports that
module. This only needs to satisfy import-time attribute access; nothing here
runs actual GTK.
"""

import sys
from unittest.mock import MagicMock

if "gi" not in sys.modules:
    gi_stub = MagicMock()
    gi_stub.require_version = lambda *args, **kwargs: None

    repository_stub = MagicMock()

    sys.modules["gi"] = gi_stub
    sys.modules["gi.repository"] = repository_stub
