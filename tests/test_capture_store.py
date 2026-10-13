"""Capture store: frame + face crops + meta on disk, index, pacing, budget."""

import json
import time

import numpy as np

from wpu_client.services.face_recognition.capture_store import CaptureStore, crop_quality, face_crop


def _frame(w=320, h=240):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _record(t, faces=1):
    return {
        "t": t,
        "faces": [
            {"i": i, "bbox": [40 + 80 * i, 40, 60, 70], "size": 60, "status": "recognized", "name": "Varun"}
            for i in range(faces)
        ],
    }


def _drain(store):
    store.stop()


def test_writes_frame_faces_meta_and_index(tmp_path):
    store = CaptureStore(tmp_path, min_interval_s=0)
    store.start()
    t = time.time()
    assert store.submit(_frame(), _record(t, faces=2))
    _drain(store)

    day = next(tmp_path.iterdir())
    index = [json.loads(line) for line in (day / "index.jsonl").read_text().splitlines()]
    assert len(index) == 1
    rec = index[0]
    cap = tmp_path / rec["dir"]
    assert (cap / "frame.jpg").is_file()
    assert {p.name for p in cap.glob("face_*.jpg")} == {"face_0.jpg", "face_1.jpg"}
    assert json.loads((cap / "meta.json").read_text())["faces"][0]["crop"] == "face_0.jpg"
    assert "sharpness" in rec["faces"][0] and "brightness" in rec["faces"][0]


def test_faces_only_and_frame_only(tmp_path):
    store = CaptureStore(tmp_path / "a", save_full_frame=False, min_interval_s=0)
    store.start()
    store.submit(_frame(), _record(time.time()))
    _drain(store)
    cap = next((tmp_path / "a").glob("*/*/"))
    assert not (cap / "frame.jpg").exists() and (cap / "face_0.jpg").exists()

    store = CaptureStore(tmp_path / "b", save_faces=False, min_interval_s=0)
    store.start()
    store.submit(_frame(), _record(time.time()))
    _drain(store)
    cap = next((tmp_path / "b").glob("*/*/"))
    assert (cap / "frame.jpg").exists() and not list(cap.glob("face_*.jpg"))


def test_pacing_faces_and_empty_frames():
    store = CaptureStore("unused", min_interval_s=1.0, save_empty_every_s=0)
    assert store.wants(True, now=100.0)
    store._last_face_save = 100.0
    assert not store.wants(True, now=100.5)
    assert store.wants(True, now=101.0)
    assert not store.wants(False, now=500.0)  # empty frames off by default
    store.save_empty_every_s = 60
    assert store.wants(False, now=500.0)


def test_full_queue_drops_instead_of_blocking(tmp_path):
    store = CaptureStore(tmp_path)  # not started: nothing drains the queue
    accepted = [store.submit(_frame(), _record(time.time())) for _ in range(20)]
    assert accepted.count(True) == 8 and store.dropped == 12


def test_budget_evicts_oldest(tmp_path):
    store = CaptureStore(tmp_path, min_interval_s=0, max_bytes=1)  # every write goes over budget
    store.start()
    t = time.time()
    for k in range(3):
        store.submit(_frame(), _record(t + k))
        time.sleep(0.2)
    _drain(store)
    remaining = list(tmp_path.glob("*/*/"))
    assert len(remaining) <= 1
    day = next(tmp_path.iterdir())
    assert len((day / "index.jsonl").read_text().splitlines()) == 3  # the record outlives the pixels


def test_crop_clips_to_frame_and_quality():
    frame = _frame(100, 80)
    crop = face_crop(frame, [90, 70, 30, 30], margin=0.5)
    assert crop.shape[0] <= 80 - 55 and crop.shape[1] <= 100 - 75
    assert face_crop(frame, [200, 200, 10, 10], 0.1) is None
    q = crop_quality(frame)
    assert q["sharpness"] > 0 and 0 <= q["brightness"] <= 255
