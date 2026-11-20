"""Character compose (FRU's character form) and character-art fetching.

What is pinned here: the art lands exactly in its crop box (and matches the
server's own blend), every scene a visitor has art for becomes a slide, duo
pairs follow the FRU duo gender rules including the mixed-scene swap, and
every bad input — missing art, wrong scene size, an older server, a broken
listing — degrades to fewer slides rather than an exception.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.modules.setdefault("picamera2", MagicMock())  # face_service imports it; no camera here

from wpu_client.services.face_recognition.asset_fetcher import (  # noqa: E402
    CHARACTER_INDEX_NAME,
    fetch_person_assets,
    load_cached_character_art,
)
from wpu_client.services.face_recognition.base_composer import (  # noqa: E402
    CHAR_PREFIX,
    compose_duo_character,
    compose_single_character,
    place_character,
)
from wpu_client.services.face_recognition.face_service import (  # noqa: E402
    _composed_inputs_changed,
    _has_composed_slides,
)

SCENE_W, SCENE_H = 120, 100
GREY = 200
BOX_1 = [10, 20, 40, 60]   # x0, y0, x1, y1 — figure 1 / single
BOX_2 = [70, 20, 100, 60]  # figure 2


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _art(path: Path, bgr, w=16, h=20, alpha=255) -> Path:
    """A flat-colour RGBA art. Written as PNG: OpenCV's WebP writer drops a
    fully opaque alpha channel (real art always has transparency; see
    test_server_webp_art_loads_with_alpha)."""
    path = path.with_suffix(".png")
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = bgr
    img[..., 3] = alpha
    assert cv2.imwrite(str(path), img)
    return path


def _item(cid, path, box=BOX_1, size=(SCENE_W, SCENE_H), registry="stock-v2"):
    return {"character_id": cid, "path": path,
            "geometry": {"crop_box": list(box), "scene_size": list(size), "registry_version": registry}}


def _scenes(tmp_path, stems):
    d = tmp_path / "stock"
    d.mkdir(exist_ok=True)
    for s in stems:
        cv2.imwrite(str(d / f"{s}.png"), np.full((SCENE_H, SCENE_W, 3), GREY, np.uint8))
    return d


def _config(tmp_path, name, data):
    p = tmp_path / name / "scenes_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


def _slide(out, scene_id):
    return cv2.imread(str(out / f"{CHAR_PREFIX}{scene_id}.jpg"))


def _mean_in(img, box):
    x0, y0, x1, y1 = box
    return img[y0 + 3:y1 - 3, x0 + 3:x1 - 3].reshape(-1, 3).mean(0)


RED, BLUE = (0, 0, 255), (255, 0, 0)  # BGR


# ─────────────────────────────────────────────────────────────────────────
# Placement
# ─────────────────────────────────────────────────────────────────────────


def test_art_fills_exactly_its_crop_box(tmp_path):
    scene = np.full((SCENE_H, SCENE_W, 4), GREY, np.uint8)
    art = cv2.imread(str(_art(tmp_path / "a.webp", RED)), cv2.IMREAD_UNCHANGED)
    assert place_character(scene, art, _item("M01", None)["geometry"])
    x0, y0, x1, y1 = BOX_1
    assert (scene[y0:y1, x0:x1, :3] == RED).all()
    outside = scene.copy()
    outside[y0:y1, x0:x1] = GREY
    assert (outside[..., :3] == GREY).all()


def test_server_webp_art_loads_with_alpha(tmp_path):
    """The server's art: WebP, lossy colour, lossless alpha, transparent
    outside the head."""
    from wpu_client.services.face_recognition.base_composer import load_character_art
    img = np.zeros((40, 32, 4), np.uint8)
    img[..., :3] = RED
    img[8:32, 6:26, 3] = 255
    p = tmp_path / "art.webp"
    assert cv2.imwrite(str(p), img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    art = load_character_art(p)
    assert art is not None and art.shape == (40, 32, 4)
    assert art[0, 0, 3] == 0 and art[20, 16, 3] == 255


def test_transparent_art_leaves_the_scene_untouched(tmp_path):
    scene = np.full((SCENE_H, SCENE_W, 4), GREY, np.uint8)
    art = cv2.imread(str(_art(tmp_path / "a.webp", RED, alpha=0)), cv2.IMREAD_UNCHANGED)
    place_character(scene, art, _item("M01", None)["geometry"])
    assert (scene[..., :3] == GREY).all()


@pytest.mark.parametrize("box", [[10, 5, 105, 95], [10, 20, 40, 60]], ids=["single-size", "duo-size"])
def test_matches_the_servers_own_compose(box):
    """sketch-service character_postprocess.compose (PIL Lanczos colour,
    bilinear alpha) is the reference. The cv2 port differs by resampling
    kernels only: on real art ~48 dB at single-scene scale, >=36 dB for the
    small heads of duo scenes. Smooth synthetic art stands in for it here."""
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    h, w = 120, 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    art = np.zeros((h, w, 4), np.uint8)
    for c in range(3):
        art[..., c] = (127 + 100 * np.sin(xx / (5 + 3 * c) + c) * np.cos(yy / 7)).astype(np.uint8)
    ell = ((xx - w / 2) / (w * 0.4)) ** 2 + ((yy - h / 2) / (h * 0.42)) ** 2
    art[..., 3] = cv2.GaussianBlur(np.where(ell < 1, 255, 0).astype(np.uint8), (9, 9), 0)
    scene_rgb = rng.integers(150, 256, (SCENE_H, SCENE_W, 3), dtype=np.uint8)
    x0, y0, x1, y1 = box
    size = (x1 - x0, y1 - y0)

    rgba = art[..., [2, 1, 0, 3]]
    ref_art = np.array(Image.fromarray(rgba[..., :3]).resize(size, Image.LANCZOS), np.float32)
    a = np.array(Image.fromarray(rgba[..., 3]).resize(size, Image.BILINEAR), np.float32)[..., None] / 255
    ref = scene_rgb.astype(np.float32)
    ref[y0:y1, x0:x1] = ref[y0:y1, x0:x1] * (1 - a) + ref_art * a

    scene = np.dstack([scene_rgb[..., ::-1], np.full((SCENE_H, SCENE_W), 255, np.uint8)]).copy()
    geometry = {"crop_box": box, "scene_size": [SCENE_W, SCENE_H]}
    assert place_character(scene, art, geometry)
    got = scene[y0:y1, x0:x1, :3][..., ::-1].astype(np.float32)
    mse = ((got - ref[y0:y1, x0:x1]) ** 2).mean()
    assert 10 * np.log10(255 ** 2 / max(mse, 1e-9)) >= 33


def test_wrong_scene_size_is_refused(tmp_path):
    scene = np.full((SCENE_H, SCENE_W, 4), GREY, np.uint8)
    art = cv2.imread(str(_art(tmp_path / "a.webp", RED)), cv2.IMREAD_UNCHANGED)
    with pytest.raises(ValueError):
        place_character(scene, art, _item("M01", None, size=(SCENE_W + 2, SCENE_H))["geometry"])


# ─────────────────────────────────────────────────────────────────────────
# Single
# ─────────────────────────────────────────────────────────────────────────


SINGLE = {
    "f1f": {"name": "Horse Rider", "gender": ["male"], "character": "M01"},
    "f2f": {"name": "Meditator", "gender": ["male"], "character": "M02"},
    "f6f": {"name": "Dancer", "gender": ["female"], "character": "F06"},
}


def test_every_scene_with_art_becomes_a_slide(tmp_path):
    scenes = _scenes(tmp_path, SINGLE)
    cfg = _config(tmp_path, "character_single", SINGLE)
    arts = [_item("M01", _art(tmp_path / "m1.webp", RED)), _item("M02", _art(tmp_path / "m2.webp", BLUE))]
    out = tmp_path / "display"
    assert compose_single_character(arts, "male", scenes, cfg, out) == str(out)
    slides = sorted(p.name for p in out.glob(f"{CHAR_PREFIX}*"))
    assert slides == [f"{CHAR_PREFIX}f1f.jpg", f"{CHAR_PREFIX}f2f.jpg"]
    assert np.allclose(_mean_in(_slide(out, "f1f"), BOX_1), RED, atol=6)
    assert np.allclose(_mean_in(_slide(out, "f2f"), BOX_1), BLUE, atol=6)


def test_no_art_means_no_slides(tmp_path):
    scenes = _scenes(tmp_path, SINGLE)
    cfg = _config(tmp_path, "character_single", SINGLE)
    out = tmp_path / "display"
    assert compose_single_character([], "male", scenes, cfg, out) is None
    assert not out.exists()


def test_gender_filter_still_applies(tmp_path):
    """Art for a scene of the other gender (a mis-sent item) is not shown."""
    scenes = _scenes(tmp_path, SINGLE)
    cfg = _config(tmp_path, "character_single", SINGLE)
    arts = [_item("F06", _art(tmp_path / "f6.webp", RED))]
    assert compose_single_character(arts, "male", scenes, cfg, tmp_path / "out") is None


def test_unplaceable_art_is_skipped_not_raised(tmp_path):
    scenes = _scenes(tmp_path, SINGLE)
    cfg = _config(tmp_path, "character_single", SINGLE)
    (tmp_path / "broken.webp").write_bytes(b"not an image")
    arts = [_item("M01", tmp_path / "broken.webp"),
            _item("M02", _art(tmp_path / "m2.webp", BLUE), size=(999, 999))]
    assert compose_single_character(arts, "male", scenes, cfg, tmp_path / "out") is None


def test_cache_hit_until_the_art_set_changes(tmp_path):
    scenes = _scenes(tmp_path, SINGLE)
    cfg = _config(tmp_path, "character_single", SINGLE)
    out = tmp_path / "display"
    m1 = _item("M01", _art(tmp_path / "m1.webp", RED))
    compose_single_character([m1], "male", scenes, cfg, out)
    first = (out / f"{CHAR_PREFIX}f1f.jpg").stat().st_mtime_ns
    compose_single_character([m1], "male", scenes, cfg, out)
    assert (out / f"{CHAR_PREFIX}f1f.jpg").stat().st_mtime_ns == first  # cached
    m2 = _item("M02", _art(tmp_path / "m2.webp", BLUE))
    compose_single_character([m1, m2], "male", scenes, cfg, out)
    assert (out / f"{CHAR_PREFIX}f2f.jpg").exists()  # new art -> recomposed


def test_character_slides_count_as_composed(tmp_path):
    d = tmp_path / "display"
    d.mkdir()
    (d / f"{CHAR_PREFIX}f1f.jpg").write_bytes(b"x")
    assert _has_composed_slides(d)


def test_character_art_change_invalidates_composed_slides():
    assert _composed_inputs_changed({"chr_M01"})
    assert _composed_inputs_changed({"fru"})
    assert not _composed_inputs_changed({"video_1"})
    assert not _composed_inputs_changed(set())


# ─────────────────────────────────────────────────────────────────────────
# Duo
# ─────────────────────────────────────────────────────────────────────────


DUO = {
    "d1f": {"name": "2 Horse Riders", "gender": ["male"], "character_1": "D1_1", "character_2": "D1_2"},
    "d3f": {"name": "2 Female Dancers", "gender": ["female"], "character_1": "D3_1", "character_2": "D3_2"},
    "d4f": {"name": "Male & Female Dancers", "gender": ["mixed"], "gender_1": "male",
            "character_1": "D4_1", "character_2": "D4_2"},
}
MALE_SLOTS, FEMALE_SLOTS = ("D1_1", "D1_2", "D4_1"), ("D3_1", "D3_2", "D4_2")


def _person(tmp_path, tag, colour, slots):
    out = []
    for cid in slots:
        box = BOX_1 if cid.endswith("_1") else BOX_2
        out.append(_item(cid, _art(tmp_path / f"{tag}_{cid}.webp", colour), box=box))
    return out


def test_two_men_get_the_male_duo_scene_a_left_b_right(tmp_path):
    scenes = _scenes(tmp_path, DUO)
    cfg = _config(tmp_path, "character_duo", DUO)
    a, b = _person(tmp_path, "a", RED, MALE_SLOTS), _person(tmp_path, "b", BLUE, MALE_SLOTS)
    out = tmp_path / "pair"
    assert compose_duo_character(a, b, "male", "male", scenes, cfg, out) == str(out)
    assert sorted(p.name for p in out.glob(f"{CHAR_PREFIX}*")) == [f"{CHAR_PREFIX}d1f.jpg"]
    img = _slide(out, "d1f")
    assert np.allclose(_mean_in(img, BOX_1), RED, atol=6)
    assert np.allclose(_mean_in(img, BOX_2), BLUE, atol=6)


def test_two_women_get_the_female_duo_scene(tmp_path):
    scenes = _scenes(tmp_path, DUO)
    cfg = _config(tmp_path, "character_duo", DUO)
    a, b = _person(tmp_path, "a", RED, FEMALE_SLOTS), _person(tmp_path, "b", BLUE, FEMALE_SLOTS)
    out = tmp_path / "pair"
    compose_duo_character(a, b, "female", "female", scenes, cfg, out)
    assert sorted(p.name for p in out.glob(f"{CHAR_PREFIX}*")) == [f"{CHAR_PREFIX}d3f.jpg"]


@pytest.mark.parametrize("woman_first", [False, True])
def test_mixed_pair_puts_the_man_on_the_male_figure(tmp_path, woman_first):
    scenes = _scenes(tmp_path, DUO)
    cfg = _config(tmp_path, "character_duo", DUO)
    man = ("male", _person(tmp_path, "m", RED, MALE_SLOTS))
    woman = ("female", _person(tmp_path, "w", BLUE, FEMALE_SLOTS))
    (ga, a), (gb, b) = (woman, man) if woman_first else (man, woman)
    out = tmp_path / "pair"
    compose_duo_character(a, b, ga, gb, scenes, cfg, out)
    assert sorted(p.name for p in out.glob(f"{CHAR_PREFIX}*")) == [f"{CHAR_PREFIX}d4f.jpg"]
    img = _slide(out, "d4f")
    assert np.allclose(_mean_in(img, BOX_1), RED, atol=6)   # man, figure 1
    assert np.allclose(_mean_in(img, BOX_2), BLUE, atol=6)  # woman, figure 2


def test_duo_needs_both_peoples_art_for_their_figures(tmp_path):
    """Visitors registered before duo art existed (single art only) get no duo slide."""
    scenes = _scenes(tmp_path, DUO)
    cfg = _config(tmp_path, "character_duo", DUO)
    a = _person(tmp_path, "a", RED, MALE_SLOTS)
    b = [_item("M01", _art(tmp_path / "b_m1.webp", BLUE))]
    assert compose_duo_character(a, b, "male", "male", scenes, cfg, tmp_path / "pair") is None


# ─────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────


RID = "reg-9"
OBJ = f"http://minio/visits/{RID}/wpu/character"


class CharHTTP:
    """wpu/images with an optional character_art list."""

    def __init__(self, items=None, field=True, etag='"v1"'):
        self.items = items if items is not None else ["M01", "M02"]
        self.field = field
        self.etag = etag
        self.downloads = []

    def get_json_once(self, url, params=None, headers=None):
        if "/sau/media" in url:
            return None
        body = {"signed_urls": []}
        if self.field:
            body["character_art"] = [
                {"character_id": c, "signed_url": f"{OBJ}/character_{c}.webp?sig=1",
                 "geometry": {"crop_box": BOX_1, "scene_size": [SCENE_W, SCENE_H],
                              "registry_version": "stock-v2"}}
                for c in self.items]
        return body

    def head(self, url, headers=None):
        return {"ETag": self.etag}

    def download_binary(self, url, headers=None):
        self.downloads.append(url)
        return b"ART-" + self.etag.encode()


def _fetch(http, assets_dir):
    return fetch_person_assets(http, "http://api/v1/wpu/images", "http://api/v1/sau/media", RID,
                               "sau_cutout.png", "fru_cutout.png", 0, assets_dir)


def test_character_art_is_downloaded_with_its_geometry(tmp_path):
    r = _fetch(CharHTTP(), tmp_path)
    assert [a["character_id"] for a in r["characters"]] == ["M01", "M02"]
    assert r["characters"][0]["geometry"]["crop_box"] == BOX_1
    assert (tmp_path / "raw" / "chr_M01.webp").read_bytes() == b'ART-"v1"'
    assert {"chr_M01", "chr_M02"} <= r["changed"]
    # and found again from disk, as after a restart
    assert [a["character_id"] for a in load_cached_character_art(tmp_path / "raw")] == ["M01", "M02"]


def test_repeat_fetch_is_a_cache_hit(tmp_path):
    http = CharHTTP()
    _fetch(http, tmp_path)
    r = _fetch(http, tmp_path)
    assert len(http.downloads) == 2 and not r["changed"]


def test_older_server_without_the_field_keeps_cached_art(tmp_path):
    _fetch(CharHTTP(), tmp_path)
    r = _fetch(CharHTTP(field=False), tmp_path)
    assert [a["character_id"] for a in r["characters"]] == ["M01", "M02"]
    assert not r["changed"]


def test_withdrawn_art_is_removed_and_reported(tmp_path):
    _fetch(CharHTTP(), tmp_path)
    r = _fetch(CharHTTP(items=["M01"]), tmp_path)
    assert [a["character_id"] for a in r["characters"]] == ["M01"]
    assert not (tmp_path / "raw" / "chr_M02.webp").exists()
    assert "chr_M02" in r["changed"]
    index = json.loads((tmp_path / "raw" / CHARACTER_INDEX_NAME).read_text())
    assert [e["character_id"] for e in index] == ["M01"]


def test_malformed_and_unsafe_entries_are_skipped(tmp_path):
    http = CharHTTP(items=["M01", "../evil", "ok_2"])
    r = _fetch(http, tmp_path)
    assert [a["character_id"] for a in r["characters"]] == ["M01", "ok_2"]
    assert not any(p.name.startswith("chr_..") for p in (tmp_path / "raw").iterdir())
