"""Base-mode composition tests.

Focus is the body (SAU) placement math, which is the part genuinely new to
the client: cv2 has no equivalent of PIL's auto-clipping paste, so an anchor
near a scene edge has to be clipped by hand or cv2 raises a shape-mismatch.
These assert both that nothing raises AND that the pixels land where the
server's PIL implementation would have put them.

Face (FRU) composition isn't covered here — it needs a real YuNet detector
and a face with detectable eyes, which is on-device territory.
"""

import json

import cv2
import numpy as np
import pytest

from wpu_client.services.face_recognition.base_composer import (
    BODY_PREFIX,
    FACE_PREFIX,
    _cache_hit,
    _duo_gender_allows,
    _gender_allows,
    _paste_body,
    compose_single_body,
    derive_scene_name,
    scene_label,
)

SCENE_W, SCENE_H = 400, 300
CUTOUT_W, CUTOUT_H = 100, 200


def _scene():
    """Plain grey BGRA scene."""
    scene = np.full((SCENE_H, SCENE_W, 4), 200, np.uint8)
    scene[:, :, 3] = 255
    return scene


def _cutout():
    """Fully-opaque red BGRA body cutout."""
    cut = np.zeros((CUTOUT_H, CUTOUT_W, 4), np.uint8)
    cut[:, :, 2] = 255
    cut[:, :, 3] = 255
    return cut


def _red_bbox(scene):
    """(x0, y0, x1, y1) of the pasted red pixels, or None if none landed."""
    mask = (scene[:, :, 2] > 150) & (scene[:, :, 0] < 100) & (scene[:, :, 1] < 100)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@pytest.mark.parametrize(
    "anchor,expected",
    [
        # anchor is where the FEET land: centred on x, bottom edge at y.
        ((200, 250), (150, 50, 250, 250)),
        ((0, 250), (0, 50, 50, 250)),        # left half clipped off-scene
        ((SCENE_W, 250), (350, 50, 400, 250)),  # right half clipped
        ((200, 50), (150, 0, 250, 50)),      # head clipped at the top edge
    ],
)
def test_paste_body_geometry_and_clipping(anchor, expected):
    """Placement matches the server's paste_x/paste_y math, clipped to bounds."""
    scene = _scene()
    assert _paste_body(scene, _cutout(), 1.0, anchor) is True
    assert _red_bbox(scene) == expected


@pytest.mark.parametrize(
    "scale,anchor",
    [
        (1.0, (200, 700)),      # entirely below the scene
        (1.0, (-500, 250)),     # entirely off to the left
        (1.0, (900, 250)),      # entirely off to the right
    ],
)
def test_paste_body_skips_fully_offscreen(scale, anchor):
    """A cutout landing wholly outside is skipped, not blended or raised."""
    scene = _scene()
    assert _paste_body(scene, _cutout(), scale, anchor) is False
    assert _red_bbox(scene) is None


@pytest.mark.parametrize(
    "scale",
    [0.0, -2.0, "not-a-number", None, float("nan"), float("inf")],
)
def test_paste_body_rejects_bad_scale(scale):
    """A malformed scale is skipped rather than raising."""
    scene = _scene()
    assert _paste_body(scene, _cutout(), scale, (200, 250)) is False


@pytest.mark.parametrize("anchor", [None, (), (5,), ("a", "b")])
def test_paste_body_rejects_bad_anchor(anchor):
    """A malformed anchor is skipped rather than raising."""
    scene = _scene()
    assert _paste_body(scene, _cutout(), 1.0, anchor) is False


def test_paste_body_scale_much_larger_than_scene():
    """An oversized cutout is clipped to the scene, covering it, without error."""
    scene = _scene()
    assert _paste_body(scene, _cutout(), 9.0, (200, 250)) is True
    x0, y0, x1, y1 = _red_bbox(scene)
    assert (x0, y0) == (0, 0)
    assert (x1, y1) == (SCENE_W, 250)


def test_paste_body_blends_by_alpha():
    """Alpha is respected — a half-transparent cutout blends, not overwrites."""
    scene = _scene()
    cut = _cutout()
    cut[:, :, 3] = 128
    _paste_body(scene, cut, 1.0, (200, 250))
    centre = scene[150, 200]
    assert 90 < int(centre[0]) < 110   # grey pulled toward 0 (blue channel)
    assert 200 < int(centre[2]) < 240  # red pulled up toward 255


# ─────────────────────────────────────────────────────────────────────────
# Scene-config driven composition
# ─────────────────────────────────────────────────────────────────────────


def _write_scenes(tmp_path, config):
    scenes_dir = tmp_path / "sau_single"
    scenes_dir.mkdir()
    for scene_id in config:
        cv2.imwrite(str(scenes_dir / f"{scene_id}.png"), _scene())
    config_path = scenes_dir / "scenes_config.json"
    config_path.write_text(json.dumps(config))
    return scenes_dir, config_path


def _write_cutout(tmp_path):
    path = tmp_path / "sau.png"
    cv2.imwrite(str(path), _cutout())
    return path


def test_compose_single_body_writes_one_slide_per_scene(tmp_path):
    scenes_dir, config_path = _write_scenes(
        tmp_path, {"a": {"scale": 1.0, "anchor": [200, 250]},
                   "b": {"scale": 0.5, "anchor": [100, 250]}}
    )
    out = tmp_path / "display"
    result = compose_single_body(_write_cutout(tmp_path), scenes_dir, config_path, out)
    assert result == str(out)
    assert sorted(p.name for p in out.iterdir()) == [
        f"{BODY_PREFIX}a.jpg",
        f"{BODY_PREFIX}b.jpg",
    ]


def test_compose_single_body_missing_config_returns_none(tmp_path):
    """Scene assets don't exist yet — must degrade, not raise."""
    out = tmp_path / "display"
    result = compose_single_body(
        _write_cutout(tmp_path), tmp_path / "nope", tmp_path / "nope" / "c.json", out
    )
    assert result is None


def test_compose_single_body_missing_cutout_returns_none(tmp_path):
    scenes_dir, config_path = _write_scenes(tmp_path, {"a": {"scale": 1.0, "anchor": [200, 250]}})
    result = compose_single_body(
        tmp_path / "absent.png", scenes_dir, config_path, tmp_path / "display"
    )
    assert result is None


def test_compose_single_body_all_scenes_unplaceable_returns_none(tmp_path):
    """Every scene rejects the paste — no output, so no sketch_dir."""
    scenes_dir, config_path = _write_scenes(
        tmp_path, {"a": {"scale": 1.0, "anchor": [-9999, 250]}}
    )
    result = compose_single_body(
        _write_cutout(tmp_path), scenes_dir, config_path, tmp_path / "display"
    )
    assert result is None


# ─────────────────────────────────────────────────────────────────────────
# Cache scoping — body and face slides share one flat display/ dir
# ─────────────────────────────────────────────────────────────────────────


def test_cache_hit_is_scoped_to_its_own_prefix(tmp_path):
    """Body output must not read as a face cache hit, or vice versa.

    Both composers write into the same display/ dir. A naive "dir is
    non-empty" check would make whichever runs second skip its work entirely.
    """
    display = tmp_path / "display"
    display.mkdir()
    (display / f"{BODY_PREFIX}1.jpg").write_bytes(b"x")

    assert _cache_hit(display, BODY_PREFIX) is True
    assert _cache_hit(display, FACE_PREFIX) is False

    (display / f"{FACE_PREFIX}1.jpg").write_bytes(b"x")
    assert _cache_hit(display, FACE_PREFIX) is True


def test_cache_hit_ignores_unrelated_files(tmp_path):
    """A hardlinked video alone is not a composed-slide cache hit."""
    display = tmp_path / "display"
    display.mkdir()
    (display / "video_1.mov").write_bytes(b"x")
    assert _cache_hit(display, BODY_PREFIX) is False
    assert _cache_hit(display, FACE_PREFIX) is False


def test_cache_hit_on_missing_dir(tmp_path):
    assert _cache_hit(tmp_path / "nothing", BODY_PREFIX) is False


def test_compose_single_body_reuses_cached_output(tmp_path):
    """Second call must not recompose — and must not be fooled into skipping
    by a face slide sitting beside its own output."""
    scenes_dir, config_path = _write_scenes(tmp_path, {"a": {"scale": 1.0, "anchor": [200, 250]}})
    out = tmp_path / "display"
    cutout = _write_cutout(tmp_path)

    compose_single_body(cutout, scenes_dir, config_path, out)
    marker = out / f"{BODY_PREFIX}a.jpg"
    marker.write_bytes(b"sentinel")

    assert compose_single_body(cutout, scenes_dir, config_path, out) == str(out)
    assert marker.read_bytes() == b"sentinel"  # untouched => genuine cache hit


# ─────────────────────────────────────────────────────────────────────────
# Gender filtering
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "gender,meta,expected",
    [
        ("male", {"gender": ["male"]}, True),
        ("female", {"gender": ["male"]}, False),
        # Permissive when unknown: the server's gender field may not be rolled
        # out yet, and excluding every scene would hide all slides.
        (None, {"gender": ["male"]}, True),
        ("", {"gender": ["male"]}, True),
        ("male", {}, True),
        ("unknown", {}, True),
    ],
)
def test_gender_allows(gender, meta, expected):
    assert _gender_allows(gender, meta) is expected


# ── duo gender selection ────────────────────────────────────────────────
# A duo scene constrains the PAIR, not each person: ["male", "female"] means
# "two men or two women", and a mixed pair needs its own "mixed" opt-in.
@pytest.mark.parametrize(
    "allowed,pair,expected",
    [
        # Two men.
        (["male"], ("male", "male"), True),
        (["male"], ("female", "female"), False),
        (["male"], ("male", "female"), False),
        # Two women.
        (["female"], ("female", "female"), True),
        (["female"], ("male", "male"), False),
        (["female"], ("female", "male"), False),
        # Either same-gender pairing — NOT a mixed pair.
        (["male", "female"], ("male", "male"), True),
        (["male", "female"], ("female", "female"), True),
        (["male", "female"], ("male", "female"), False),
        # Mixed only.
        (["mixed"], ("male", "female"), True),
        (["mixed"], ("female", "male"), True),
        (["mixed"], ("male", "male"), False),
        (["mixed"], ("female", "female"), False),
        # Lists combine.
        (["male", "mixed"], ("male", "male"), True),
        (["male", "mixed"], ("male", "female"), True),
        (["male", "mixed"], ("female", "female"), False),
    ],
)
def test_duo_gender_allows(allowed, pair, expected):
    assert _duo_gender_allows(pair[0], pair[1], {"gender": allowed}) is expected


@pytest.mark.parametrize("meta", [{}, {"gender": []}])
def test_duo_scene_without_a_gender_key_takes_any_pair(meta):
    assert _duo_gender_allows("male", "female", meta) is True


@pytest.mark.parametrize("pair", [(None, "male"), ("male", None), (None, None), ("", "male")])
def test_duo_gender_is_permissive_when_either_is_unknown(pair):
    """The server may not return gender yet. Excluding every scene would
    leave those visitors with no slides at all."""
    assert _duo_gender_allows(pair[0], pair[1], {"gender": ["mixed"]}) is True


def test_mixed_never_matches_a_single_person():
    """One person is not a pair, so a mixed-only scene must not be composed
    for them in the single-person path."""
    assert _gender_allows("male", {"gender": ["mixed"]}) is False
    assert _gender_allows("female", {"gender": ["mixed"]}) is False


# ── scene names ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "filename,expected",
    [
        # What the shipped scenes look like today: nothing to humanise.
        ("4.png", "4"),
        ("data/base_scenes/fru_duo/4.png", "4"),
        # What the field is for, once backgrounds get real names.
        ("winter_market.jpg", "Winter Market"),
        ("rooftop-sunset.png", "Rooftop Sunset"),
        ("Lobby_Desk.jpeg", "Lobby Desk"),
        ("stage_2.png", "Stage 2"),
    ],
)
def test_derive_scene_name(filename, expected):
    assert derive_scene_name(filename) == expected


def test_scene_label_falls_back_to_the_id():
    assert scene_label("4", {}) == "4"
    # A name identical to the id adds nothing to a log line.
    assert scene_label("4", {"name": "4"}) == "4"
    assert scene_label("4", {"name": "Winter Market"}) == "4 (Winter Market)"
