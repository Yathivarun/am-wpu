"""Base-mode composition tests.

The placement is a port of the server pipeline's own scene composition, so
these pin its geometry: body canvases sized by height rather than by the
photo's resolution and anchored by their bottom edge; faces placed by the
server's landmarks, sized by eye distance or head height, ringed at on-scene
scale. Everything else — clipping, bad inputs, caching, gender rules — must
degrade quietly rather than take down the recognition loop.
"""

import json
import struct
import zlib

import cv2
import numpy as np
import pytest

from wpu_client.services.face_recognition import base_composer as bc
from wpu_client.services.face_recognition.base_composer import (
    BODY_PREFIX,
    COMPOSE_VERSION,
    FACE_PREFIX,
    SAU_REFERENCE_CANVAS_HEIGHT_PX,
    Cutout,
    _cache_hit,
    _duo_gender_allows,
    _gender_allows,
    _write_stamp,
    compose_duo_body,
    compose_single_body,
    compose_single_face,
    derive_scene_name,
    load_cutout,
    place_body,
    place_face,
    scene_label,
)

SCENE_W, SCENE_H = 400, 300
GREY = 200


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────


def _scene():
    scene = np.full((SCENE_H, SCENE_W, 4), GREY, np.uint8)
    scene[:, :, 3] = 255
    return scene


def _red(w, h, alpha=255):
    cut = np.zeros((h, w, 4), np.uint8)
    cut[:, :, 2] = 255
    cut[:, :, 3] = alpha
    return cut


def _red_bbox(scene):
    """(x0, y0, x1, y1) of clearly red pixels, or None."""
    mask = (scene[:, :, 2] > 150) & (scene[:, :, 0] < 100) & (scene[:, :, 1] < 100)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _scale_for_canvas_px(px):
    """The scene `scale` that sizes a canvas to `px` at the reference height."""
    return px / SAU_REFERENCE_CANVAS_HEIGHT_PX


def _png_with_text(bgra, keyword, text, chunk=b"tEXt"):
    """Encode a PNG and insert a text chunk right after IHDR."""
    ok, buf = cv2.imencode(".png", bgra)
    assert ok
    data = buf.tobytes()
    if chunk == b"tEXt":
        payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    elif chunk == b"zTXt":
        payload = keyword.encode("latin-1") + b"\x00\x00" + zlib.compress(text.encode("latin-1"))
    else:  # iTXt, uncompressed
        payload = keyword.encode("latin-1") + b"\x00\x00\x00" + b"\x00" + b"\x00" + text.encode("utf-8")
    body = chunk + payload
    new_chunk = struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
    ihdr_end = 8 + 8 + 13 + 4
    return data[:ihdr_end] + new_chunk + data[ihdr_end:]


def _write_cutout(path, bgra, geometry=None, chunk=b"tEXt"):
    if geometry is None:
        cv2.imwrite(str(path), bgra)
    else:
        path.write_bytes(_png_with_text(bgra, "am-sketch-geometry", json.dumps(geometry), chunk))
    return path


def _sau_geometry(canvas, box):
    return {"v": 1, "modality": "sau", "canvas_size": list(canvas),
            "cutout_box": list(box), "outline_baked": False}


# ─────────────────────────────────────────────────────────────────────────
# Embedded geometry
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("chunk", [b"tEXt", b"zTXt", b"iTXt"])
def test_geometry_is_read_from_every_text_chunk_kind(tmp_path, chunk):
    geo = _sau_geometry((120, 240), (10, 20, 110, 220))
    path = _write_cutout(tmp_path / "sau.png", _red(100, 200), geo, chunk)
    cutout = load_cutout(path, "sau")
    assert cutout.geometry == geo
    assert cutout.bgra.shape == (200, 100, 4)   # pixels still decode normally


def test_plain_png_has_no_geometry(tmp_path):
    assert load_cutout(_write_cutout(tmp_path / "a.png", _red(10, 10)), "sau").geometry is None


@pytest.mark.parametrize("geo", [
    {"v": 99, "modality": "sau"},                       # unknown version
    {"v": 1, "modality": "fru", "kps": [[1, 2], [3, 4]]},  # wrong modality
])
def test_unusable_geometry_is_ignored(tmp_path, geo):
    path = _write_cutout(tmp_path / "a.png", _red(10, 10), geo)
    assert load_cutout(path, "sau").geometry is None


def test_malformed_geometry_json_is_ignored(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(_png_with_text(_red(10, 10), "am-sketch-geometry", "{not json"))
    assert load_cutout(path, "sau").geometry is None


# ─────────────────────────────────────────────────────────────────────────
# Body placement
# ─────────────────────────────────────────────────────────────────────────


def test_body_canvas_bottom_sits_on_the_anchor():
    """With embedded geometry the CANVAS is anchored, so the subject sits
    above the anchor by the (scaled) padding under the feet — as on the server."""
    # 100x200 subject inside a 120x240 canvas; size the canvas to 120 px -> eff 0.5.
    cutout = Cutout(_red(100, 200), _sau_geometry((120, 240), (10, 20, 110, 220)))
    scene = _scene()
    assert place_body(scene, cutout, _scale_for_canvas_px(120), (200, 250)) is True
    x0, y0, x1, y1 = _red_bbox(scene)
    # Canvas 60x120 at (170, 130); subject 50x100 at +(5, 10) -> (175, 140)-(225, 240).
    assert abs(x0 - 175) <= 2 and abs(x1 - 225) <= 2
    assert abs(y0 - 140) <= 2 and abs(y1 - 240) <= 2


def test_body_size_does_not_depend_on_photo_resolution():
    """The bug the server fixed: the same person photographed at a different
    resolution must land at the same on-scene size."""
    small, large = _scene(), _scene()
    place_body(small, Cutout(_red(100, 200)), _scale_for_canvas_px(150), (200, 280))
    place_body(large, Cutout(_red(300, 600)), _scale_for_canvas_px(150), (200, 280))
    a, b = _red_bbox(small), _red_bbox(large)
    assert all(abs(p - q) <= 2 for p, q in zip(a, b, strict=True)), (a, b)


def test_body_without_geometry_estimates_the_padded_canvas():
    """No embedded geometry: canvas = cutout + 5% padding per side."""
    scene = _scene()
    # 100x200 cutout -> canvas 110x220; size it to 110 px -> eff 0.5.
    place_body(scene, Cutout(_red(100, 200)), _scale_for_canvas_px(110), (200, 250))
    x0, y0, x1, y1 = _red_bbox(scene)
    # Canvas 55x110 at (172, 140); subject 50x100 at +(2.5, 5).
    assert abs(y1 - 245) <= 2 and abs((x0 + x1) / 2 - 200) <= 2


@pytest.mark.parametrize("height,expected_px", [(170, 100), (85, 50), (None, 100),
                                                ("tall", 100), (-5, 100), (1000, 100)])
def test_body_scales_with_visitor_height(height, expected_px):
    """Height scales linearly against the reference; unknown or implausible
    heights fall back to it."""
    scene = _scene()
    geo = _sau_geometry((100, 200), (0, 0, 100, 200))
    place_body(scene, Cutout(_red(100, 200), geo), _scale_for_canvas_px(100), (200, 280), height)
    x0, y0, x1, y1 = _red_bbox(scene)
    assert abs((y1 - y0) - expected_px) <= 3


def test_body_scene_reference_height_is_honoured():
    """A scene calibrated for 180 cm renders a 180 cm visitor at full scale."""
    scene = _scene()
    geo = _sau_geometry((100, 200), (0, 0, 100, 200))
    place_body(scene, Cutout(_red(100, 200), geo), _scale_for_canvas_px(100), (200, 280), 180, 180)
    x0, y0, x1, y1 = _red_bbox(scene)
    assert abs((y1 - y0) - 100) <= 3


def _below_feet(shadow: bool, monkeypatch) -> int:
    monkeypatch.setitem(bc.SAU_SHADOW_PARAMS, "enabled", shadow)
    scene = _scene()
    geo = _sau_geometry((100, 200), (0, 0, 100, 200))
    # Canvas 110x220 -> ellipse 66 px wide centred 30 px above the feet line.
    place_body(scene, Cutout(_red(100, 200), geo), _scale_for_canvas_px(220), (200, 240))
    return int(scene[250, 200, 0])     # just under the feet, outside the subject


def test_body_casts_a_ground_shadow(monkeypatch):
    assert _below_feet(True, monkeypatch) < _below_feet(False, monkeypatch) - 5


def test_body_shadow_can_be_disabled(monkeypatch):
    assert _below_feet(False, monkeypatch) == GREY


def test_body_mask_edge_is_soft():
    """The server dilates + blurs the mask: the boundary blends, it doesn't step."""
    scene = _scene()
    # Padded like a real cutout; canvas sized to 240 px -> eff 1, body at x 150-250.
    geo = _sau_geometry((120, 240), (10, 20, 110, 220))
    place_body(scene, Cutout(_red(100, 200), geo), _scale_for_canvas_px(240), (200, 290))
    row = scene[180, :, 2].astype(int)   # red channel across the body
    edge = row[140:160]                   # left edge region
    assert any(GREY < v < 250 for v in edge)   # intermediate values = soft edge


@pytest.mark.parametrize("anchor", [(0, 250), (SCENE_W, 250), (200, 60)])
def test_body_partially_offscreen_is_clipped_not_raised(anchor):
    scene = _scene()
    assert place_body(scene, Cutout(_red(100, 200)), _scale_for_canvas_px(110), anchor) is True
    assert _red_bbox(scene) is not None


@pytest.mark.parametrize("anchor", [(200, 900), (-900, 250), (1500, 250)])
def test_body_fully_offscreen_is_skipped_untouched(anchor):
    scene = _scene()
    assert place_body(scene, Cutout(_red(100, 200)), _scale_for_canvas_px(110), anchor) is False
    assert (scene[:, :, :3] == GREY).all()


@pytest.mark.parametrize("scale", [0.0, -2.0, "not-a-number", None, float("nan"), float("inf")])
def test_body_rejects_bad_scale(scale):
    assert place_body(_scene(), Cutout(_red(100, 200)), scale, (200, 250)) is False


@pytest.mark.parametrize("anchor", [None, (), (5,), ("a", "b")])
def test_body_rejects_bad_anchor(anchor):
    assert place_body(_scene(), Cutout(_red(100, 200)), 0.03, anchor) is False


def test_body_geometry_that_contradicts_the_cutout_is_not_trusted():
    """A box that doesn't describe this crop falls back to the estimate."""
    scene = _scene()
    bad = _sau_geometry((120, 240), (0, 0, 50, 50))   # 50x50 box, 100x200 cutout
    assert place_body(scene, Cutout(_red(100, 200), bad), _scale_for_canvas_px(110), (200, 250))
    assert _red_bbox(scene) is not None


# ─────────────────────────────────────────────────────────────────────────
# Face placement
# ─────────────────────────────────────────────────────────────────────────


def _face_cutout(eye_l=(30, 40), eye_r=(70, 40), bbox=(20, 10, 80, 90), baked=False):
    """Opaque blue 100x100 face with a green dot on the left eye."""
    face = np.zeros((100, 100, 4), np.uint8)
    face[:, :, 0] = 255
    face[:, :, 3] = 255
    cv2.circle(face, eye_l, 3, (0, 255, 0, 255), -1)
    geo = {"v": 1, "modality": "fru", "canvas_size": [100, 100], "cutout_box": [0, 0, 100, 100],
           "outline_baked": baked, "kps": [list(eye_l), list(eye_r), [50, 60], [35, 75], [65, 75]],
           "face_bbox": list(bbox) if bbox else None}
    return Cutout(face, geo)


def _anchor(mid=(200, 150), dist=80, tilt=0, head=None):
    a = {"target_eye_midpoint": list(mid), "target_eye_distance": dist, "target_tilt_angle": tilt}
    if head is not None:
        a["target_head_height"] = head
    return a


def _green_centre(scene):
    ys, xs = np.nonzero((scene[:, :, 1] > 180) & (scene[:, :, 0] < 100) & (scene[:, :, 2] < 100))
    return float(xs.mean()), float(ys.mean())


def _blue_bbox(scene):
    ys, xs = np.nonzero((scene[:, :, 0] > 180) & (scene[:, :, 2] < 100))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_face_eyes_land_on_the_target():
    cutout = _face_cutout()
    scene = _scene()
    kps = bc.face_kps(cutout, None)
    assert place_face(scene, cutout, kps, _anchor(dist=80)) is True
    # Eye distance 40 -> 80: scale 2, so the left eye lands 40 px left of the midpoint.
    gx, gy = _green_centre(scene)
    assert abs(gx - 160) <= 2 and abs(gy - 150) <= 2


def test_face_tilt_rotates_about_the_eye_midpoint():
    cutout = _face_cutout()
    scene = _scene()
    place_face(scene, cutout, bc.face_kps(cutout, None), _anchor(dist=80, tilt=90))
    # +90 deg (counter-clockwise on screen): the left eye swings below the midpoint.
    gx, gy = _green_centre(scene)
    assert abs(gx - 200) <= 3 and abs(gy - 190) <= 3


def test_face_sized_by_head_height_when_the_scene_provides_it():
    cutout = _face_cutout(bbox=(20, 10, 80, 60))   # head height 50
    scene = _scene()
    # Eye distance alone would give scale 2; head height 100/50 also gives 2.
    # Make them disagree: target head 150 -> scale 3.
    place_face(scene, cutout, bc.face_kps(cutout, None), _anchor(dist=80, head=150))
    x0, y0, x1, y1 = _blue_bbox(scene)
    assert abs((x1 - x0) - 300) <= 12 or x1 - x0 >= SCENE_W - 5   # 100 px face x3 (may clip)
    gx, _ = _green_centre(scene)
    assert abs(gx - (200 - 20 * 3)) <= 3     # left eye 20 px from mid, x3


def test_face_without_head_bbox_falls_back_to_eye_distance():
    cutout = _face_cutout(bbox=None)
    scene = _scene()
    place_face(scene, cutout, bc.face_kps(cutout, None), _anchor(dist=80, head=150))
    gx, _ = _green_centre(scene)
    assert abs(gx - 160) <= 2


def test_face_gets_an_outline_ring():
    cutout = _face_cutout()
    scene = _scene()
    place_face(scene, cutout, bc.face_kps(cutout, None), _anchor(dist=80))
    x0, y0, x1, y1 = _blue_bbox(scene)
    ring_px = scene[(y0 + y1) // 2, x0 - 2, :3]
    assert int(ring_px.max()) < 60           # dark ring just outside the face


def test_face_baked_outline_is_not_doubled():
    cutout = _face_cutout(baked=True)
    scene = _scene()
    place_face(scene, cutout, bc.face_kps(cutout, None), _anchor(dist=80))
    x0, y0, x1, y1 = _blue_bbox(scene)
    assert (scene[(y0 + y1) // 2, x0 - 3, :3] == GREY).all()


def test_face_kps_prefer_embedded_landmarks_over_detection():
    class ExplodingDetector:
        def setInputSize(self, size):
            raise AssertionError("detector must not run when landmarks are embedded")

    kps = bc.face_kps(_face_cutout(), ExplodingDetector())
    assert kps.tolist() == [[30, 40], [70, 40]]


def test_face_without_landmarks_or_detector_has_no_kps():
    assert bc.face_kps(Cutout(np.zeros((10, 10, 4), np.uint8)), None) is None


@pytest.mark.parametrize("anchor", [{}, {"target_eye_midpoint": [1]},
                                    {"target_eye_midpoint": [1, 2], "target_eye_distance": "x",
                                     "target_tilt_angle": 0}])
def test_face_bad_anchor_raises_for_the_caller_to_skip(anchor):
    cutout = _face_cutout()
    with pytest.raises((KeyError, TypeError, ValueError)):
        place_face(_scene(), cutout, bc.face_kps(cutout, None), anchor)


def test_face_fully_offscreen_is_skipped():
    cutout = _face_cutout()
    assert place_face(_scene(), cutout, bc.face_kps(cutout, None), _anchor(mid=(5000, 5000))) is False


# ─────────────────────────────────────────────────────────────────────────
# Scene-config driven composition
# ─────────────────────────────────────────────────────────────────────────


def _write_scenes(tmp_path, config, name="sau_single"):
    scenes_dir = tmp_path / name
    scenes_dir.mkdir()
    for scene_id in config:
        cv2.imwrite(str(scenes_dir / f"{scene_id}.png"), _scene())
    config_path = scenes_dir / "scenes_config.json"
    config_path.write_text(json.dumps(config))
    return scenes_dir, config_path


def _body(tmp_path, name="sau.png"):
    return _write_cutout(tmp_path / name, _red(100, 200))


def test_compose_single_body_writes_one_slide_per_scene(tmp_path):
    scenes_dir, config_path = _write_scenes(
        tmp_path, {"a": {"scale": 0.03, "anchor": [200, 250]},
                   "b": {"scale": 0.02, "anchor": [100, 250]}}
    )
    out = tmp_path / "display"
    assert compose_single_body(_body(tmp_path), scenes_dir, config_path, out) == str(out)
    assert sorted(p.name for p in out.iterdir() if not p.name.startswith(".")) == [
        f"{BODY_PREFIX}a.jpg", f"{BODY_PREFIX}b.jpg",
    ]


def test_compose_single_body_missing_config_returns_none(tmp_path):
    out = tmp_path / "display"
    assert compose_single_body(_body(tmp_path), tmp_path / "nope", tmp_path / "nope" / "c.json", out) is None


def test_compose_single_body_missing_cutout_returns_none(tmp_path):
    scenes_dir, config_path = _write_scenes(tmp_path, {"a": {"scale": 0.03, "anchor": [200, 250]}})
    assert compose_single_body(tmp_path / "absent.png", scenes_dir, config_path, tmp_path / "d") is None


def test_compose_single_body_all_scenes_unplaceable_returns_none(tmp_path):
    scenes_dir, config_path = _write_scenes(tmp_path, {"a": {"scale": 0.03, "anchor": [-9999, 250]}})
    assert compose_single_body(_body(tmp_path), scenes_dir, config_path, tmp_path / "d") is None


def test_compose_single_face_uses_embedded_landmarks_without_a_detector(tmp_path):
    scenes_dir, config_path = _write_scenes(
        tmp_path, {"1": {"gender": ["male"], "face_anchor": _anchor()},
                   "2": {"gender": ["female"], "face_anchor": _anchor()}}, "fru_single")
    cut = _face_cutout()
    fru = _write_cutout(tmp_path / "fru.png", cut.bgra, cut.geometry)
    out = tmp_path / "display"
    assert compose_single_face(None, fru, "male", scenes_dir, config_path, out) == str(out)
    assert [p.name for p in out.iterdir() if not p.name.startswith(".")] == [f"{FACE_PREFIX}1.jpg"]


def test_compose_single_face_without_landmarks_or_detector_returns_none(tmp_path):
    scenes_dir, config_path = _write_scenes(tmp_path, {"1": {"face_anchor": _anchor()}}, "fru_single")
    fru = _write_cutout(tmp_path / "fru.png", _face_cutout().bgra)   # no geometry
    assert compose_single_face(None, fru, None, scenes_dir, config_path, tmp_path / "d") is None


def test_compose_duo_body_places_each_person_at_their_own_height(tmp_path):
    geo = _sau_geometry((100, 200), (0, 0, 100, 200))
    a = _write_cutout(tmp_path / "a.png", _red(100, 200), geo)
    b = _write_cutout(tmp_path / "b.png", _red(100, 200), geo)
    scale = _scale_for_canvas_px(100)
    scenes_dir, config_path = _write_scenes(tmp_path, {"1": {
        "scale_1": scale, "anchor_1": [120, 280], "scale_2": scale, "anchor_2": [290, 280]}}, "sau_duo")
    out = tmp_path / "display"
    assert compose_duo_body(a, b, scenes_dir, config_path, out, height_a=170, height_b=85) == str(out)
    img = cv2.imread(str(out / f"{BODY_PREFIX}1.jpg"))
    left, right = img[:, :200], img[:, 200:]
    h_left = _red_bbox(np.dstack([left, np.full(left.shape[:2], 255, np.uint8)]))
    h_right = _red_bbox(np.dstack([right, np.full(right.shape[:2], 255, np.uint8)]))
    assert abs((h_left[3] - h_left[1]) - 100) <= 4
    assert abs((h_right[3] - h_right[1]) - 50) <= 4


# ─────────────────────────────────────────────────────────────────────────
# Cache — scoped per prefix, stamped by version and height
# ─────────────────────────────────────────────────────────────────────────


def test_cache_hit_is_scoped_to_its_own_prefix(tmp_path):
    """Body output must not read as a face cache hit, or vice versa."""
    display = tmp_path / "display"
    display.mkdir()
    (display / f"{BODY_PREFIX}1.jpg").write_bytes(b"x")
    _write_stamp(display, BODY_PREFIX)

    assert _cache_hit(display, BODY_PREFIX) is True
    assert _cache_hit(display, FACE_PREFIX) is False

    (display / f"{FACE_PREFIX}1.jpg").write_bytes(b"x")
    _write_stamp(display, FACE_PREFIX)
    assert _cache_hit(display, FACE_PREFIX) is True


def test_cache_hit_ignores_unrelated_files(tmp_path):
    display = tmp_path / "display"
    display.mkdir()
    (display / "video_1.mov").write_bytes(b"x")
    assert _cache_hit(display, BODY_PREFIX) is False
    assert _cache_hit(display, FACE_PREFIX) is False


def test_cache_hit_on_missing_dir(tmp_path):
    assert _cache_hit(tmp_path / "nothing", BODY_PREFIX) is False


def test_unstamped_slides_from_an_older_composer_are_discarded(tmp_path):
    """Slides cached by the previous composer (no stamp) must not be shown."""
    display = tmp_path / "display"
    display.mkdir()
    old = display / f"{BODY_PREFIX}1.jpg"
    old.write_bytes(b"old")
    video = display / "video_1.mov"
    video.write_bytes(b"v")
    assert _cache_hit(display, BODY_PREFIX) is False
    assert not old.exists()
    assert video.exists()                         # only this composer's files go


def test_compose_single_body_reuses_cached_output(tmp_path):
    scenes_dir, config_path = _write_scenes(tmp_path, {"a": {"scale": 0.03, "anchor": [200, 250]}})
    out = tmp_path / "display"
    cutout = _body(tmp_path)
    compose_single_body(cutout, scenes_dir, config_path, out, height_cm=172)
    marker = out / f"{BODY_PREFIX}a.jpg"
    marker.write_bytes(b"sentinel")
    assert compose_single_body(cutout, scenes_dir, config_path, out, height_cm=172) == str(out)
    assert marker.read_bytes() == b"sentinel"     # untouched => genuine cache hit


def test_compose_single_body_recomposes_when_height_changes(tmp_path):
    scenes_dir, config_path = _write_scenes(tmp_path, {"a": {"scale": 0.03, "anchor": [200, 250]}})
    out = tmp_path / "display"
    cutout = _body(tmp_path)
    compose_single_body(cutout, scenes_dir, config_path, out, height_cm=172)
    marker = out / f"{BODY_PREFIX}a.jpg"
    marker.write_bytes(b"sentinel")
    compose_single_body(cutout, scenes_dir, config_path, out, height_cm=160)
    assert marker.read_bytes() != b"sentinel"
    stamp = json.loads((out / f".{BODY_PREFIX}stamp.json").read_text())
    assert stamp == {"v": COMPOSE_VERSION, "height_cm": [160.0]}


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
        (["male"], ("male", "male"), True),
        (["male"], ("female", "female"), False),
        (["male"], ("male", "female"), False),
        (["female"], ("female", "female"), True),
        (["female"], ("male", "male"), False),
        (["female"], ("female", "male"), False),
        (["male", "female"], ("male", "male"), True),
        (["male", "female"], ("female", "female"), True),
        (["male", "female"], ("male", "female"), False),
        (["mixed"], ("male", "female"), True),
        (["mixed"], ("female", "male"), True),
        (["mixed"], ("male", "male"), False),
        (["mixed"], ("female", "female"), False),
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
    assert _duo_gender_allows(pair[0], pair[1], {"gender": ["mixed"]}) is True


def test_mixed_never_matches_a_single_person():
    assert _gender_allows("male", {"gender": ["mixed"]}) is False
    assert _gender_allows("female", {"gender": ["mixed"]}) is False


# ── scene names ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("4.png", "4"),
        ("data/base_scenes/fru_duo/4.png", "4"),
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
    assert scene_label("4", {"name": "4"}) == "4"
    assert scene_label("4", {"name": "Winter Market"}) == "4 (Winter Market)"


# ─────────────────────────────────────────────────────────────────────────
# Duo face: mixed pairs land on the figure drawn as their gender
# ─────────────────────────────────────────────────────────────────────────


def _duo_face_slide(tmp_path, first, second, anchor_genders=("male", "female")):
    """Compose (first, second) — each "male" or "female" — onto one mixed scene
    whose left figure is anchor_genders[0]; return the slide."""
    left, right = _anchor(mid=(100, 150), dist=40), _anchor(mid=(300, 150), dist=40)
    if anchor_genders:
        left["gender"], right["gender"] = anchor_genders
    scenes_dir, config_path = _write_scenes(
        tmp_path, {"4": {"gender": ["mixed"], "face_anchor_1": left, "face_anchor_2": right}}, "fru_duo")
    paths = []
    for i, gender in enumerate((first, second)):
        cut = _face_cutout()
        bgra = cut.bgra.copy()
        if gender == "female":            # female face red, male stays blue
            bgra[:, :, 0], bgra[:, :, 2] = bgra[:, :, 2].copy(), 255
        paths.append(_write_cutout(tmp_path / f"fru_{i}.png", bgra, cut.geometry))
    out = tmp_path / "display"
    assert bc.compose_duo_face(None, paths[0], paths[1], first, second, scenes_dir, config_path, out)
    return cv2.imread(str(out / f"{FACE_PREFIX}4.jpg"))


def _is_blue(px):
    return px[0] > 180 and px[2] < 100


@pytest.mark.parametrize("order", [("male", "female"), ("female", "male")])
def test_duo_face_mixed_pair_follows_anchor_gender_whatever_the_slug_order(tmp_path, order):
    slide = _duo_face_slide(tmp_path, *order)
    assert _is_blue(slide[180, 100]) and not _is_blue(slide[180, 300])   # man left, woman right


def test_duo_face_without_anchor_gender_keeps_slug_order(tmp_path):
    slide = _duo_face_slide(tmp_path, "female", "male", anchor_genders=None)
    assert not _is_blue(slide[180, 100]) and _is_blue(slide[180, 300])
