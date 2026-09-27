"""Video slide playback: frame hand-off and teardown races.

GStreamer and GTK are stubbed (see conftest), so these pin the logic that
decides what reaches the screen: frames coalesce into one pending texture,
late frames are dropped rather than slowing playback, and nothing a stopped
player queued can land on the slide that replaced it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wpu_client.services.slideshow import slideshow_service as ss


@pytest.fixture
def gi(monkeypatch):
    """Fresh GLib/Gdk/Gst stubs per test; idle callbacks are captured, not run."""
    glib, gdk, gst = MagicMock(), MagicMock(), MagicMock()
    gdk.MemoryTexture.new.side_effect = lambda *a: object()   # distinct textures
    monkeypatch.setattr(ss, "GLib", glib)
    monkeypatch.setattr(ss, "Gdk", gdk)
    monkeypatch.setattr(ss, "Gst", gst)
    monkeypatch.setattr(ss, "_GST_INITIALIZED", True)
    return SimpleNamespace(GLib=glib, Gdk=gdk, Gst=gst)


def _player(gi, **callbacks):
    picture = MagicMock()
    player = ss._GstVideoPlayer(
        "/tmp/v.mp4", picture,
        on_eos=callbacks.get("on_eos", MagicMock()),
        on_error=callbacks.get("on_error", MagicMock()),
        on_duration=callbacks.get("on_duration"),
    )
    return player, picture


def _sink():
    """A sink whose pull-sample yields a mappable 2x2 RGBA frame."""
    buf = MagicMock()
    buf.map.return_value = (True, SimpleNamespace(data=b"\0" * 16))
    structure = MagicMock()
    structure.get_value.side_effect = lambda key: 2
    sample = MagicMock()
    sample.get_buffer.return_value = buf
    sample.get_caps.return_value.get_structure.return_value = structure
    sink = MagicMock()
    sink.emit.return_value = sample
    return sink


def _idle_calls(gi, fn):
    return [c for c in gi.GLib.idle_add.call_args_list if c.args and c.args[0] == fn]


# ── pipeline ─────────────────────────────────────────────────────────────


def test_pipeline_drops_late_frames_instead_of_slowing_down(gi):
    _player(gi)
    launch = gi.Gst.parse_launch.call_args.args[0]
    assert "qos=true" in launch
    assert f"max-lateness={ss._GstVideoPlayer.MAX_LATENESS_NS}" in launch
    assert "! queue" in launch and "max-buffers=1 drop=true" in launch


@pytest.mark.parametrize("has_property,expected", [(True, True), (False, False)])
def test_conversion_uses_all_cores_when_supported(gi, has_property, expected):
    convert = MagicMock()
    convert.find_property.return_value = object() if has_property else None
    gi.Gst.parse_launch.return_value.get_by_name.side_effect = (
        lambda name: convert if name == "convert" else MagicMock()
    )
    _player(gi)
    called = any(c.args == ("n-threads", 0) for c in convert.set_property.call_args_list)
    assert called is expected


# ── frame hand-off ───────────────────────────────────────────────────────


def test_frames_coalesce_into_one_pending_draw(gi):
    """A slow main loop sees only the newest frame, not a backlog."""
    player, picture = _player(gi)
    sink = _sink()
    for _ in range(3):
        player._on_new_sample(sink)

    assert len(_idle_calls(gi, player._present)) == 1       # one draw scheduled
    newest = player._pending_texture
    player._present()
    picture.set_paintable.assert_called_once_with(newest)

    player._on_new_sample(sink)                              # next frame schedules again
    assert len(_idle_calls(gi, player._present)) == 2


def test_frames_are_scheduled_ahead_of_redraw(gi):
    player, _ = _player(gi)
    player._on_new_sample(_sink())
    (call,) = _idle_calls(gi, player._present)
    assert call.kwargs["priority"] == gi.GLib.PRIORITY_HIGH_IDLE


def test_a_frame_queued_before_stop_never_reaches_the_screen(gi):
    """The 'stuck on a frame' bug: a frame queued just before teardown used to
    overwrite the next slide's image."""
    player, picture = _player(gi)
    player._on_new_sample(_sink())
    player.stop()
    player._present()
    picture.set_paintable.assert_not_called()


def test_no_frames_are_produced_after_stop(gi):
    player, _ = _player(gi)
    player.stop()
    player._on_new_sample(_sink())
    assert _idle_calls(gi, player._present) == []


# ── stale callbacks ──────────────────────────────────────────────────────


def test_stale_eos_does_not_skip_the_next_slide(gi):
    on_eos = MagicMock()
    player, _ = _player(gi, on_eos=on_eos)
    player._on_bus_message(None, SimpleNamespace(type=gi.Gst.MessageType.EOS))
    (call,) = _idle_calls(gi, player._call_if_live)
    player.stop()
    player._call_if_live(*call.args[1:])
    on_eos.assert_not_called()


def test_live_eos_still_advances(gi):
    on_eos = MagicMock()
    player, _ = _player(gi, on_eos=on_eos)
    player._on_bus_message(None, SimpleNamespace(type=gi.Gst.MessageType.EOS))
    (call,) = _idle_calls(gi, player._call_if_live)
    player._call_if_live(*call.args[1:])
    on_eos.assert_called_once()


def test_stale_duration_cannot_cancel_the_next_slides_timer(gi):
    on_duration = MagicMock()
    gi.Gst.SECOND = 1_000_000_000
    player, _ = _player(gi, on_duration=on_duration)
    player.pipeline.query_duration.return_value = (True, 12 * gi.Gst.SECOND)
    player._on_new_sample(_sink())
    (call,) = _idle_calls(gi, player._call_if_live)
    player.stop()
    player._call_if_live(*call.args[1:])
    on_duration.assert_not_called()
