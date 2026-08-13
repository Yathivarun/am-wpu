"""Base-mode asset fetching / caching tests.

The behaviours worth pinning down here are the ones a kiosk depends on and
that are easy to regress silently:

  * a repeat visitor costs no downloads (cache hit),
  * a re-photographed person IS re-downloaded even though the server keeps
    the cutout at the same object key (ETag revalidation),
  * an offline Pi keeps showing what it already has instead of losing it,
  * the disk cache stays under its cap, evicting duo pairs before people.
"""

import json
import os
import time

import pytest

from wpu_client.services.face_recognition.asset_fetcher import (
    _dir_size,
    enforce_disk_budget,
    fetch_person_assets,
    link_videos_into,
)

RID = "reg-123"
IMAGES_URL = "http://api/wpu/images"
VIDEOS_URL = "http://api/wpu/videos"
OBJECT_BASE = f"http://minio/visits/{RID}/wpu"


class FakeHTTP:
    """Minimal stand-in for HTTPClient covering the three methods used."""

    def __init__(self, etag='"v1"', videos=False, offline=False):
        self.etag = etag
        self.videos = videos
        self.offline = offline
        self.downloads = 0
        self.heads = 0

    def get(self, url, params=None, headers=None):
        if self.offline:
            raise RuntimeError("network unreachable")
        if "videos" in url:
            if not self.videos:
                raise RuntimeError("404 — no such route")
            return {"signed_urls": [f"{OBJECT_BASE}/video_1.mov"]}
        return {
            "signed_urls": [
                f"{OBJECT_BASE}/sau_cutout.png",
                f"{OBJECT_BASE}/fru_cutout.png",
                # An unrelated object that must not match any configured name.
                f"{OBJECT_BASE}/scene_1.png",
            ]
        }

    def head(self, url, headers=None):
        self.heads += 1
        return None if self.offline else {"ETag": self.etag}

    def download_binary(self, url, headers=None):
        if self.offline:
            return None
        self.downloads += 1
        return b"BYTES-" + self.etag.encode()


def _fetch(http, assets_dir, videos=("video_1.mov",)):
    return fetch_person_assets(
        http, IMAGES_URL, VIDEOS_URL, RID,
        "sau_cutout.png", "fru_cutout.png", list(videos), assets_dir,
    )


def test_first_fetch_downloads_both_cutouts(tmp_path):
    http = FakeHTTP()
    result = _fetch(http, tmp_path)

    assert result["sau"].name == "sau.png"
    assert result["fru"].name == "fru.png"
    assert result["changed"] == {"sau", "fru"}
    # scene_1.png matches no configured filename and must be ignored.
    assert http.downloads == 2


def test_repeat_visit_is_a_cache_hit(tmp_path):
    http = FakeHTTP()
    _fetch(http, tmp_path)
    result = _fetch(http, tmp_path)

    assert http.downloads == 2  # unchanged — nothing re-downloaded
    assert result["changed"] == set()
    assert result["sau"].exists() and result["fru"].exists()


def test_changed_etag_forces_refetch(tmp_path):
    """The server overwrites cutouts at the SAME object key when a person is
    re-photographed, so the URL alone can never reveal the change."""
    http = FakeHTTP()
    _fetch(http, tmp_path)

    http.etag = '"v2"'
    result = _fetch(http, tmp_path)

    assert result["changed"] == {"sau", "fru"}
    assert http.downloads == 4
    assert (tmp_path / "raw" / "sau.png").read_bytes() == b'BYTES-"v2"'


def test_offline_keeps_previously_fetched_assets(tmp_path):
    """A flaky network must not cost a visitor their already-cached slides."""
    _fetch(FakeHTTP(), tmp_path)
    result = _fetch(FakeHTTP(offline=True), tmp_path)

    assert result["sau"] is not None
    assert result["fru"] is not None
    assert result["changed"] == set()


def test_offline_new_person_degrades_to_nothing(tmp_path):
    result = _fetch(FakeHTTP(offline=True), tmp_path)
    assert result["sau"] is None
    assert result["fru"] is None
    assert result["videos"] == []


def test_missing_videos_endpoint_is_not_an_error(tmp_path):
    """No video route exists server-side yet — this is the normal path."""
    result = _fetch(FakeHTTP(videos=False), tmp_path)
    assert result["videos"] == []
    assert result["sau"] is not None  # cutouts still fetched fine


def test_videos_fetched_when_available(tmp_path):
    result = _fetch(FakeHTTP(videos=True), tmp_path)
    assert [p.name for p in result["videos"]] == ["video_1.mov"]


def test_manifest_records_every_configured_asset(tmp_path):
    _fetch(FakeHTTP(), tmp_path, videos=("video_1.mov", "video_2.mov"))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert sorted(manifest) == ["fru", "sau", "video_1", "video_2"]
    assert manifest["sau"]["present"] is True
    assert manifest["video_1"]["present"] is False


def test_corrupt_manifest_is_survivable(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text("{ not json")
    result = _fetch(FakeHTTP(), tmp_path)
    assert result["sau"] is not None


def test_fetch_never_raises_on_a_broken_client(tmp_path):
    """This runs on the recognition loop — it must not take the loop down."""

    class Exploding:
        def get(self, *a, **k):
            raise RuntimeError("boom")

        def head(self, *a, **k):
            raise RuntimeError("boom")

        def download_binary(self, *a, **k):
            raise RuntimeError("boom")

    result = _fetch(Exploding(), tmp_path)
    assert result["sau"] is None and result["fru"] is None


# ─────────────────────────────────────────────────────────────────────────
# Videos into display/
# ─────────────────────────────────────────────────────────────────────────


def test_link_videos_into_display(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    video = raw / "video_1.mov"
    video.write_bytes(b"movie")

    display = tmp_path / "display"
    assert link_videos_into([video], display) == 1
    assert (display / "video_1.mov").read_bytes() == b"movie"
    # Hardlinked, not duplicated — same inode where the FS supports it.
    assert (display / "video_1.mov").stat().st_ino == video.stat().st_ino


def test_link_videos_is_idempotent(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    video = raw / "v.mov"
    video.write_bytes(b"movie")
    display = tmp_path / "display"

    assert link_videos_into([video], display) == 1
    assert link_videos_into([video], display) == 1
    assert len(list(display.iterdir())) == 1


# ─────────────────────────────────────────────────────────────────────────
# Disk budget
# ─────────────────────────────────────────────────────────────────────────


def _entry(root, rel, kb, age_s):
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    (path / "blob.bin").write_bytes(b"x" * (kb * 1024))
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_budget_noop_when_under_cap(tmp_path):
    _entry(tmp_path, "alice", 100, 10)
    assert enforce_disk_budget(tmp_path, 10 * 1024 * 1024) == 0
    assert (tmp_path / "alice").exists()


def test_budget_evicts_duo_pairs_before_people(tmp_path):
    """Pairs grow quadratically with visitor count and rebuild cheaply from
    the retained per-person cutouts, so they go first — even when they are
    the most recently used entries."""
    _entry(tmp_path, "alice", 500, 900)          # oldest person
    _entry(tmp_path, "bob", 500, 100)
    _entry(tmp_path, "duo/alice__bob", 500, 10)  # newest of all
    _entry(tmp_path, "duo/alice__carl", 500, 5)

    enforce_disk_budget(tmp_path, 1100 * 1024)

    assert (tmp_path / "alice").exists()
    assert (tmp_path / "bob").exists()
    assert not (tmp_path / "duo" / "alice__bob").exists()
    assert not (tmp_path / "duo" / "alice__carl").exists()
    assert _dir_size(tmp_path) <= 1100 * 1024


def test_budget_evicts_people_least_recently_used_first(tmp_path):
    _entry(tmp_path, "old", 500, 9000)
    _entry(tmp_path, "recent", 500, 5)

    enforce_disk_budget(tmp_path, 600 * 1024)

    assert not (tmp_path / "old").exists()
    assert (tmp_path / "recent").exists()


def test_budget_respects_protected_dirs(tmp_path):
    """Whatever is on screen right now must never be deleted underneath it."""
    old = _entry(tmp_path, "old", 500, 9000)
    _entry(tmp_path, "recent", 500, 5)

    enforce_disk_budget(tmp_path, 600 * 1024, protect={str(old)})

    assert old.exists()
    assert not (tmp_path / "recent").exists()


@pytest.mark.parametrize("cap", [0, -1])
def test_budget_disabled_when_cap_not_positive(tmp_path, cap):
    _entry(tmp_path, "alice", 500, 10)
    assert enforce_disk_budget(tmp_path, cap) == 0
    assert (tmp_path / "alice").exists()


def test_budget_on_missing_dir_is_noop(tmp_path):
    assert enforce_disk_budget(tmp_path / "absent", 1024) == 0
