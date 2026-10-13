"""Diagnostic gallery: nearest-regardless-of-threshold, and live reload."""

import json
import os
import time

import numpy as np

from wpu_client.services.face_recognition.diagnostic_gallery import DiagnosticGallery


def _seed(root, slug, vec, name=None):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embedding_mobilenet.npy", np.asarray(vec, np.float32))
    (d / "meta.json").write_text(json.dumps({"name": name or slug.title()}))
    return d


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_nearest_reports_beyond_threshold(tmp_path):
    _seed(tmp_path, "a", _unit([1, 0, 0]))
    _seed(tmp_path, "b", _unit([0, 1, 0]))
    g = DiagnosticGallery(str(tmp_path))
    g.load()
    probe = _unit([0.2, 1, 0.9])
    near = g.nearest(probe, "mobilenet")
    assert near.entry.slug == "b" and near.distance > 0.2
    assert g.match(probe, 0.2, "mobilenet") is None
    assert g.match(probe, 0.9, "mobilenet").entry.slug == "b"
    assert g.nearest(probe, "sface") is None


def test_reload_picks_up_added_changed_and_removed(tmp_path):
    _seed(tmp_path, "a", _unit([1, 0, 0]))
    g = DiagnosticGallery(str(tmp_path))
    g.load()
    assert g.reload_if_changed() == set()

    _seed(tmp_path, "b", _unit([0, 1, 0]))
    assert g.reload_if_changed() == {"b"} and len(g) == 2

    sketches = tmp_path / "a" / "sketches"
    sketches.mkdir()
    (sketches / "fru.png").write_bytes(b"x")
    future = time.time() + 5
    os.utime(sketches / "fru.png", (future, future))
    assert g.reload_if_changed() == {"a"}

    import shutil

    shutil.rmtree(tmp_path / "b")
    assert g.reload_if_changed() == {"b"} and len(g) == 1


def test_hidden_staging_folders_are_ignored(tmp_path):
    _seed(tmp_path, ".seeding_x", _unit([1, 0, 0]))
    _seed(tmp_path, "a", _unit([0, 1, 0]))
    g = DiagnosticGallery(str(tmp_path))
    assert g.load() == 1 and ".seeding_x" not in g.signature()
