"""Scene source selection: compose on the Pi vs pre-composed scenes.

_ensure_single_assets tries the configured source first and, with the
fallback on, the other one — in both modes, with diagnostic mode never
touching the server. The composers and fetchers are faked; what is pinned
here is the ordering, the fallback, and that the two slide sets never mix.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("picamera2", MagicMock())

from wpu_client.config.settings import FaceRecognitionConfig  # noqa: E402
from wpu_client.services.face_recognition import face_service as fs  # noqa: E402
from wpu_client.services.face_recognition.asset_fetcher import PRECOMPOSED_PREFIX  # noqa: E402

RID = "reg-1"


class World:
    """Fakes for everything _ensure_single_assets calls out to."""

    def __init__(self, tmp_path, cutouts=True, scenes=True):
        self.tmp = tmp_path
        self.have_cutouts = cutouts
        self.have_scenes = scenes
        self.fetch_calls: list[bool] = []       # fetch_cutouts flag per call
        self.scene_fetches = 0
        self.composed = 0

    # fetch_person_assets(http, wpu, media, rid, sau_name, fru_name, count, assets_dir, fetch_cutouts=)
    def fetch_person_assets(self, *args, fetch_cutouts=True):
        self.fetch_calls.append(fetch_cutouts)
        raw = self.tmp / "raw"
        raw.mkdir(exist_ok=True)
        out = {"sau": None, "fru": None, "videos": [], "changed": set()}
        if fetch_cutouts and self.have_cutouts:
            for k in ("sau", "fru"):
                (raw / f"{k}.png").write_bytes(b"x")
                out[k] = raw / f"{k}.png"
        return out

    def fetch_precomposed_scenes(self, http, endpoint, rid, assets_dir):
        self.scene_fetches += 1
        if not self.have_scenes:
            return []
        d = self.tmp / "server_scenes"
        d.mkdir(exist_ok=True)
        (d / "sau_scene_1.jpg").write_bytes(b"s")
        return [d / "sau_scene_1.jpg"]

    def compose_single_body(self, sau, scenes_dir, cfg, out, height_cm=None):
        self.composed += 1
        if sau:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "sau_scene_1.jpg").write_bytes(b"c")

    def compose_single_face(self, detector, fru, gender, scenes_dir, cfg, out):
        if fru:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "fru_scene_1.jpg").write_bytes(b"c")


def _service(tmp_path, monkeypatch, world, diagnostic=False, gallery=None, **cfg):
    for name in ("fetch_person_assets", "fetch_precomposed_scenes",
                 "compose_single_body", "compose_single_face"):
        monkeypatch.setattr(fs, name, getattr(world, name))
    monkeypatch.setattr(fs, "enforce_disk_budget", lambda *a, **k: None)
    svc = object.__new__(fs.FaceRecognitionService)   # no camera / models / dataset dirs
    svc.config = FaceRecognitionConfig(diagnostic_mode=diagnostic, **cfg)
    svc._diagnostic_mode = diagnostic
    svc._gallery = gallery
    svc._assets_root = tmp_path / "assets"
    svc._duo_assets_root = tmp_path / "assets" / "duo"
    svc._single_cache, svc._duo_cache = {}, {}
    svc._yunet_detector = None
    svc._http_client = MagicMock()
    return svc


def _slides(sketch_dir):
    return sorted(p.name for p in Path(sketch_dir).iterdir() if not p.name.startswith("."))


# ── base (server) mode ──────────────────────────────────────────────────


def test_compose_first_composes_and_never_fetches_scenes(tmp_path, monkeypatch):
    w = World(tmp_path)
    svc = _service(tmp_path, monkeypatch, w)
    d = svc._ensure_single_assets(RID, "male")
    assert _slides(d) == ["fru_scene_1.jpg", "sau_scene_1.jpg"]
    assert w.fetch_calls == [True] and w.scene_fetches == 0


def test_compose_falls_back_to_precomposed(tmp_path, monkeypatch):
    w = World(tmp_path, cutouts=False)
    svc = _service(tmp_path, monkeypatch, w)
    d = svc._ensure_single_assets(RID, None)
    assert _slides(d) == [f"{PRECOMPOSED_PREFIX}sau_scene_1.jpg"]


def test_compose_without_fallback_shows_nothing(tmp_path, monkeypatch):
    w = World(tmp_path, cutouts=False)
    svc = _service(tmp_path, monkeypatch, w, scene_source_fallback=False)
    assert svc._ensure_single_assets(RID, None) is None
    assert w.scene_fetches == 0


def test_precomposed_first_skips_the_cutouts_and_composer(tmp_path, monkeypatch):
    w = World(tmp_path)
    svc = _service(tmp_path, monkeypatch, w, scene_source="precomposed")
    d = svc._ensure_single_assets(RID, None)
    assert _slides(d) == [f"{PRECOMPOSED_PREFIX}sau_scene_1.jpg"]
    assert w.fetch_calls == [False] and w.composed == 0


def test_precomposed_falls_back_to_composing(tmp_path, monkeypatch):
    """No server scenes: fetch the cutouts it skipped, then compose."""
    w = World(tmp_path, scenes=False)
    svc = _service(tmp_path, monkeypatch, w, scene_source="precomposed")
    d = svc._ensure_single_assets(RID, None)
    assert _slides(d) == ["fru_scene_1.jpg", "sau_scene_1.jpg"]
    assert w.fetch_calls == [False, True]


def test_legacy_valve_selects_precomposed(tmp_path, monkeypatch):
    assert FaceRecognitionConfig(use_legacy_final_images=True).scene_source == "precomposed"
    w = World(tmp_path)
    svc = _service(tmp_path, monkeypatch, w, use_legacy_final_images=True)
    assert _slides(svc._ensure_single_assets(RID, None)) == [f"{PRECOMPOSED_PREFIX}sau_scene_1.jpg"]


def test_switching_source_never_mixes_the_two_sets(tmp_path, monkeypatch):
    w = World(tmp_path)
    composed = _service(tmp_path, monkeypatch, w)._ensure_single_assets(RID, None)
    pre = _service(tmp_path, monkeypatch, w, scene_source="precomposed")._ensure_single_assets(RID, None)
    assert composed == pre
    assert _slides(pre) == [f"{PRECOMPOSED_PREFIX}sau_scene_1.jpg"]      # composed ones removed
    back = _service(tmp_path, monkeypatch, w)._ensure_single_assets(RID, None)
    assert _slides(back) == ["fru_scene_1.jpg", "sau_scene_1.jpg"]      # pre_ ones removed


# ── diagnostic (offline) mode ───────────────────────────────────────────


def _gallery(tmp_path, with_scenes=True):
    person = tmp_path / "embeddings" / "ekan"
    (person / "sketches").mkdir(parents=True)
    if with_scenes:
        (person / "scenes").mkdir()
        (person / "scenes" / "temple.jpg").write_bytes(b"p")
        (person / "scenes" / "notes.txt").write_text("not a slide")
    entry = SimpleNamespace(sketch_dir=str(person / "sketches"), meta={})
    return SimpleNamespace(get_entry=lambda slug: entry if slug == "ekan" else None)


def test_diagnostic_precomposed_reads_the_gallery_scenes_offline(tmp_path, monkeypatch):
    w = World(tmp_path)
    svc = _service(tmp_path, monkeypatch, w, diagnostic=True, gallery=_gallery(tmp_path),
                   scene_source="precomposed")
    monkeypatch.setattr(svc, "_cutouts_for", lambda rid: {"sau": None, "fru": None})
    d = svc._ensure_single_assets("ekan", None)
    assert _slides(d) == [f"{PRECOMPOSED_PREFIX}temple.jpg"]
    assert w.scene_fetches == 0 and w.fetch_calls == []                 # never the server


def test_diagnostic_compose_falls_back_to_gallery_scenes(tmp_path, monkeypatch):
    w = World(tmp_path)
    svc = _service(tmp_path, monkeypatch, w, diagnostic=True, gallery=_gallery(tmp_path))
    monkeypatch.setattr(svc, "_cutouts_for", lambda rid: {"sau": None, "fru": None})
    assert _slides(svc._ensure_single_assets("ekan", None)) == [f"{PRECOMPOSED_PREFIX}temple.jpg"]
    assert w.scene_fetches == 0


# ── duo ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg,composes", [
    ({}, True),
    ({"scene_source": "precomposed"}, True),                           # fallback composes
    ({"scene_source": "precomposed", "scene_source_fallback": False}, False),
])
def test_duo_is_composed_only_when_composing_is_allowed(tmp_path, monkeypatch, cfg, composes):
    svc = _service(tmp_path, monkeypatch, World(tmp_path), **cfg)
    calls = []
    monkeypatch.setattr(svc, "_cutouts_for", lambda rid: calls.append(rid) or {"sau": None, "fru": None})
    svc._build_duo_sketch_dir({}, {}, "a", "b")
    assert bool(calls) is composes
