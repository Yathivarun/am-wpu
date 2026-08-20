"""Where a visitor's slides are sourced from: local composition vs the server.

Getting this branch wrong is silent — the display still works, it just
quietly reaches for the network in a mode that is supposed to stay offline —
so the policy is pinned here rather than left to inspection.
"""

import pytest

from wpu_client.services.slideshow.slideshow_service import visitor_image_source


@pytest.fixture
def slides(tmp_path):
    """A populated display/ dir, as local composition would leave it."""
    d = tmp_path / "display"
    d.mkdir()
    (d / "sau_scene_1.jpg").write_bytes(b"x")
    return str(d)


# ── Base (server-recognition) mode ──────────────────────────────────────


def test_base_mode_prefers_locally_composed_slides(slides):
    assert visitor_image_source(slides, False, False) == "local"


def test_base_mode_falls_back_to_server_without_local_slides():
    assert visitor_image_source(None, False, False) == "server"


def test_base_mode_falls_back_when_sketch_dir_was_evicted(tmp_path):
    """The disk cap can delete a dir between composing and displaying it."""
    assert visitor_image_source(str(tmp_path / "gone"), False, False) == "server"


def test_base_mode_falls_back_on_empty_sketch_dir_string():
    assert visitor_image_source("", False, False) == "server"


def test_legacy_valve_forces_server_even_with_local_slides(slides):
    """The rollback valve must win, or flipping it would do nothing."""
    assert visitor_image_source(slides, False, True) == "server"


def test_legacy_valve_still_uses_server_without_local_slides():
    assert visitor_image_source(None, False, True) == "server"


# ── Diagnostic (offline) mode ───────────────────────────────────────────


def test_diagnostic_mode_uses_local_slides(slides):
    assert visitor_image_source(slides, True, False) == "local"


@pytest.mark.parametrize("sketch_dir", [None, "", "/nonexistent/path"])
def test_diagnostic_mode_never_calls_the_server(sketch_dir):
    """Regression guard — diagnostic mode is offline by definition.

    An unrecognized face carries sketch_dir=None, so an asset-driven branch
    that falls through to the server would fire on every unknown visitor,
    stalling the recognition thread for timeout x max_retries and breaking
    the offline guarantee. This returned "server" before it was fixed.
    """
    assert visitor_image_source(sketch_dir, True, False) == "none"


def test_diagnostic_mode_stays_offline_under_the_legacy_valve(slides):
    """The valve selects a *server* source, which diagnostic mode can't use."""
    assert visitor_image_source(slides, True, True) == "none"


def test_diagnostic_mode_never_returns_server_in_any_combination(tmp_path, slides):
    """Exhaustive: no input combination may route diagnostic mode online."""
    for sketch_dir in (None, "", slides, str(tmp_path / "absent")):
        for legacy in (False, True):
            assert visitor_image_source(sketch_dir, True, legacy) != "server"


# ── Unrecognized faces ──────────────────────────────────────────────────


def test_unrecognized_face_never_reaches_the_server():
    """An unrecognized face is tracked under a generated
    "__unrecognized__<hex>" pseudo-id that no endpoint can resolve. Asking
    anyway is a guaranteed-failing round trip on the recognition thread, and
    the server answers a non-UUID id with a 500 — observed in a live Pi run
    firing on every unknown visitor."""
    assert visitor_image_source(None, False, False, unrecognized=True) == "none"


def test_unrecognized_face_ignores_the_legacy_valve():
    """The rollback valve switches where RECOGNISED people's slides come
    from; it can't conjure a registration for someone who has none."""
    assert visitor_image_source(None, False, True, unrecognized=True) == "none"


def test_unrecognized_face_still_shows_local_slides_if_any(slides):
    """Nothing assigns these today (sketch_dir is always None for unknown
    faces), but honouring them keeps the rule 'local wins when it exists'."""
    assert visitor_image_source(slides, False, False, unrecognized=True) == "local"


def test_recognised_person_is_unaffected():
    assert visitor_image_source(None, False, False, unrecognized=False) == "server"
