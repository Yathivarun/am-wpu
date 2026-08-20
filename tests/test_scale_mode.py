"""Display geometry: production 6:7 panel vs an ordinary test monitor.

`scale_mode` is the only thing deciding on-screen shape, so the guarantee
worth pinning down is that the default still reproduces exactly what the
production panel ships with — anything else silently distorts the live
display — while `fit` gives undistorted output on test hardware.

Gtk here is the MagicMock stub from conftest; ContentFit members are stable
sentinel objects, which is all these identity comparisons need.
"""

from gi.repository import Gtk

from wpu_client.config.settings import SlideshowConfig
from wpu_client.services.slideshow.slideshow_service import _SCALE_MODE_FITS


def _fit(scale_mode):
    """The mapping SlideshowWindow._content_fit() performs."""
    return _SCALE_MODE_FITS.get(scale_mode, Gtk.ContentFit.FILL)


def test_default_scale_mode_is_the_production_panel_setting():
    """The 6:7 panel advertises 1920x1080 and squeezes it horizontally; FILL's
    stretch is what cancels that out. A different default would distort the
    live display the moment this shipped."""
    assert SlideshowConfig().scale_mode == "fill"
    assert _fit(SlideshowConfig().scale_mode) is Gtk.ContentFit.FILL


def test_fit_preserves_aspect_for_an_ordinary_monitor():
    assert _fit("fit") is Gtk.ContentFit.CONTAIN


def test_crop_fills_without_distorting():
    assert _fit("crop") is Gtk.ContentFit.COVER


def test_unknown_mode_falls_back_to_production_behaviour():
    """A typo in config.yaml must not letterbox the production panel."""
    assert _fit("cover") is Gtk.ContentFit.FILL
    assert _fit("") is Gtk.ContentFit.FILL
    assert _fit(None) is Gtk.ContentFit.FILL


def test_every_documented_mode_is_mapped():
    """settings.py documents exactly these three; a mode named in config but
    missing here would silently behave as 'fill'."""
    assert sorted(_SCALE_MODE_FITS) == ["crop", "fill", "fit"]


def test_modes_are_distinct():
    """Regression guard: the whole point is that they behave differently."""
    assert len(set(map(id, _SCALE_MODE_FITS.values()))) == 3
