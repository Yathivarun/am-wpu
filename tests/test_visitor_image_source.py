"""Where a visitor's slides are sourced from at display time.

Slides are prepared upstream in every mode (composed on the Pi, or
pre-composed scenes placed in the same display/ dir), so the slideshow's only
choice is local-or-nothing. Getting this wrong is silent — the display still
works, it just quietly reaches for the network — so the policy is pinned here.
"""

import pytest

from wpu_client.services.slideshow.slideshow_service import visitor_image_source


@pytest.fixture
def slides(tmp_path):
    """A populated display/ dir, as the face service would leave it."""
    d = tmp_path / "display"
    d.mkdir()
    (d / "sau_scene_1.jpg").write_bytes(b"x")
    return str(d)


def test_prepared_slides_are_shown(slides):
    assert visitor_image_source(slides) == "local"


@pytest.mark.parametrize("sketch_dir", [None, "", "/nonexistent/path"])
def test_nothing_prepared_means_stock_images(sketch_dir):
    """Covers unrecognized faces (sketch_dir is always None) and an evicted
    dir alike: neither may trigger a request from the display path."""
    assert visitor_image_source(sketch_dir) == "none"


def test_evicted_dir_means_stock_images(tmp_path):
    """The disk cap can delete a dir between preparing and displaying it."""
    assert visitor_image_source(str(tmp_path / "gone")) == "none"


def test_the_display_path_never_returns_server(tmp_path, slides):
    """Regression guard: the old display-time server fetch returned the raw
    cutouts as slides once the server switched to serving cutouts, and had to
    special-case diagnostic mode and unrecognized faces to stay offline."""
    for sketch_dir in (None, "", slides, str(tmp_path / "absent")):
        assert visitor_image_source(sketch_dir) in ("local", "none")
