"""Base-mode asset fetching — download a person's raw SAU/FRU cutouts and
their optional videos from the server, cached on disk.

Base (server-recognition) mode composes each visitor's slides locally from
two raw alpha cutouts instead of downloading pre-composed final images. This
module owns the *fetching* half of that: getting the cutouts (and any videos)
onto local disk and keeping them fresh. base_composer.py owns the compositing
half.

Layout per person (assets_dir = data/base_assets/<registration_id>/):
    raw/            sau.png, fru.png, video_1.mov, ...  (whichever fetched)
    manifest.json   per-asset fetch state — see _Manifest below
    display/        composed slides + hardlinked videos (written by the caller)

Everything here is BEST-EFFORT by design. A missing cutout, an unreachable
server, a 404 on the (not-yet-existing) videos endpoint — all are normal
outcomes that are logged and skipped, never raised. A visitor with no usable
assets simply produces no composed slides, and the caller falls back.

Cache freshness
───────────────
The server writes each cutout to a FIXED object key
(`<registration_id>/wpu/sau_cutout.png`), so re-photographing a person
overwrites the same path — a filename/URL comparison alone can never notice.
Each cached asset therefore records the ETag/Last-Modified it was fetched
with, and a cheap HEAD revalidates before reuse. If the validators moved, the
asset is re-downloaded and reported in the returned `changed` set so the
caller knows to discard any slides it composed from the old copy.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

from wpu_client.utils.http import HTTPClient

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
RAW_DIR_NAME = "raw"
DISPLAY_DIR_NAME = "display"

# Local filenames the fetched cutouts are stored under, independent of
# whatever the server happens to call them (those are configurable).
SAU_LOCAL_NAME = "sau.png"
FRU_LOCAL_NAME = "fru.png"


def _load_manifest(manifest_path: Path) -> dict:
    """Read the on-disk manifest, or {} if absent/unreadable/corrupt."""
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        logger.warning(f"Unreadable asset manifest {manifest_path} ({e}) — refetching")
        return {}


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Write the manifest. A failure here only costs a redundant refetch."""
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except OSError as e:
        logger.warning(f"Could not write asset manifest {manifest_path}: {e}")


def _fetch_signed_urls(http_client: HTTPClient, endpoint: str, registration_id: str) -> list[str]:
    """GET one signed_urls list. Returns [] on any failure.

    Both the images and the videos endpoint are expected to return the same
    `{"signed_urls": [...]}` shape. The videos endpoint does not exist
    server-side yet, so failing quietly here is the normal path for videos.
    """
    if not endpoint:
        return []
    try:
        data = http_client.get(endpoint, params={"registration_id": registration_id})
    except Exception as e:
        logger.info(f"No assets from {endpoint} for {registration_id}: {e}")
        return []
    urls = data.get("signed_urls", []) if isinstance(data, dict) else []
    return [u for u in urls if isinstance(u, str)]


def _match_url(urls: list[str], filename: str) -> str | None:
    """Find the URL whose path ends with `filename`.

    Safe because the server returns raw object URLs
    (`http://<endpoint>/<registration_id>/wpu/<filename>`) rather than
    opaque signed keys, so the original filename survives in the path. If
    the backend ever switches to hashed keys this is the assumption that
    breaks, and the server would need to return a {filename: url} mapping
    instead of a flat list.
    """
    if not filename:
        return None
    for url in urls:
        if urlparse(url).path.endswith(filename):
            return url
    return None


def _validators(headers: dict | None) -> dict:
    """Extract the cache validators worth storing from response headers."""
    if not headers:
        return {}
    lowered = {k.lower(): v for k, v in headers.items()}
    out = {}
    if "etag" in lowered:
        out["etag"] = lowered["etag"]
    if "last-modified" in lowered:
        out["last_modified"] = lowered["last-modified"]
    return out


def _is_still_fresh(http_client: HTTPClient, url: str, record: dict) -> bool:
    """Whether a cached asset can be reused without re-downloading.

    Revalidates with a HEAD and compares ETag/Last-Modified against what was
    stored at fetch time. Conservative on both edges: if the server offers no
    validators, or the HEAD itself fails (offline, endpoint down), the cached
    copy is kept rather than re-downloaded — a stale slide beats no slide on
    a kiosk with flaky networking.
    """
    stored = {k: record.get(k) for k in ("etag", "last_modified") if record.get(k)}
    if not stored:
        # Nothing to compare against (fetched before validators were stored,
        # or the server never sent any) — trust the cache.
        return True

    headers = http_client.head(url)
    if headers is None:
        logger.debug(f"Could not revalidate {url} — keeping cached copy")
        return True

    current = _validators(headers)
    if not current:
        return True

    if "etag" in stored and "etag" in current:
        return stored["etag"] == current["etag"]
    if "last_modified" in stored and "last_modified" in current:
        return stored["last_modified"] == current["last_modified"]
    return True


def _fetch_one(
    http_client: HTTPClient,
    urls: list[str],
    server_filename: str,
    local_name: str,
    raw_dir: Path,
    manifest: dict,
    key: str,
) -> tuple[Path | None, bool]:
    """Resolve one asset to a local file.

    Returns (local_path_or_None, changed) where `changed` is True only when
    fresh bytes were just written — the signal the caller needs to throw away
    anything composed from the previous copy.
    """
    record = manifest.get(key) or {}
    local_path = raw_dir / local_name
    url = _match_url(urls, server_filename)

    if url is None:
        # Server didn't offer this asset this time. Keep a previously-fetched
        # copy if we have one — an empty/failed listing shouldn't wipe a
        # working cache — otherwise record it as absent.
        if local_path.exists():
            logger.debug(f"{key}: not listed by server, reusing cached copy")
            return local_path, False
        manifest[key] = {"present": False, "url_path": None, "checked_at": time.time()}
        return None, False

    url_path = urlparse(url).path
    if (
        local_path.exists()
        and record.get("present")
        and record.get("url_path") == url_path
        and _is_still_fresh(http_client, url, record)
    ):
        logger.debug(f"{key}: cache hit ({local_path})")
        return local_path, False

    content = http_client.download_binary(url)
    if content is None:
        # Download failed. A previously-cached copy is better than nothing.
        if local_path.exists():
            logger.warning(f"{key}: refetch failed, keeping cached copy")
            return local_path, False
        manifest[key] = {"present": False, "url_path": url_path, "checked_at": time.time()}
        return None, False

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
    except OSError as e:
        logger.warning(f"{key}: could not write {local_path}: {e}")
        return (local_path, False) if local_path.exists() else (None, False)

    record = {
        "present": True,
        "url_path": url_path,
        "checked_at": time.time(),
        **_validators(http_client.head(url)),
    }
    manifest[key] = record
    logger.info(f"{key}: fetched {len(content)} bytes -> {local_path}")
    return local_path, True


def fetch_person_assets(
    http_client: HTTPClient,
    wpu_endpoint: str,
    wpu_videos_endpoint: str,
    registration_id: str,
    sau_filename: str,
    fru_filename: str,
    video_filenames: list[str],
    assets_dir: Path,
) -> dict:
    """Best-effort fetch of this person's SAU/FRU cutouts + videos.

    Downloads into `assets_dir/raw/`, cached via `assets_dir/manifest.json`.
    Never raises — every failure mode degrades to a None/absent entry.

    Returns:
        {
            "sau": Path | None,
            "fru": Path | None,
            "videos": list[Path],
            "changed": set[str],   # asset keys whose bytes were just refreshed
        }
    """
    result: dict = {"sau": None, "fru": None, "videos": [], "changed": set()}
    try:
        raw_dir = assets_dir / RAW_DIR_NAME
        manifest_path = assets_dir / MANIFEST_NAME
        manifest = _load_manifest(manifest_path)

        image_urls = _fetch_signed_urls(http_client, wpu_endpoint, registration_id)
        # Only worth asking for videos if any are configured.
        video_urls = (
            _fetch_signed_urls(http_client, wpu_videos_endpoint, registration_id)
            if video_filenames
            else []
        )

        for key, server_filename, local_name in (
            ("sau", sau_filename, SAU_LOCAL_NAME),
            ("fru", fru_filename, FRU_LOCAL_NAME),
        ):
            path, changed = _fetch_one(
                http_client, image_urls, server_filename, local_name,
                raw_dir, manifest, key,
            )
            result[key] = path
            if changed:
                result["changed"].add(key)

        for index, server_filename in enumerate(video_filenames, start=1):
            key = f"video_{index}"
            # Keep the server's own extension so the slideshow's
            # extension-based video detection still recognises the file.
            suffix = os.path.splitext(server_filename)[1] or ".mov"
            path, changed = _fetch_one(
                http_client, video_urls, server_filename, f"{key}{suffix}",
                raw_dir, manifest, key,
            )
            if path is not None:
                result["videos"].append(path)
            if changed:
                result["changed"].add(key)

        _save_manifest(manifest_path, manifest)
    except Exception as e:
        # Belt-and-braces: this function is called from the recognition loop
        # and must never take it down.
        logger.error(f"Asset fetch failed for {registration_id}: {e}", exc_info=True)

    logger.info(
        f"Assets for {registration_id}: sau={'y' if result['sau'] else 'n'} "
        f"fru={'y' if result['fru'] else 'n'} videos={len(result['videos'])} "
        f"changed={sorted(result['changed']) or 'none'}"
    )
    return result


def link_videos_into(videos: list[Path], display_dir: Path) -> int:
    """Hardlink fetched videos into a person's display/ dir.

    Hardlinked rather than copied so the slideshow's existing
    glob-a-single-directory mechanism picks up composed images and videos
    together without storing every video twice. Falls back to a real copy
    when the link can't be made (cross-device, or a filesystem without
    hardlinks). Returns how many videos ended up in display/.
    """
    linked = 0
    for video in videos:
        target = display_dir / video.name
        if target.exists():
            linked += 1
            continue
        try:
            display_dir.mkdir(parents=True, exist_ok=True)
            os.link(video, target)
            linked += 1
        except OSError:
            try:
                shutil.copy2(video, target)
                linked += 1
            except OSError as e:
                logger.warning(f"Could not place video {video.name} in {display_dir}: {e}")
    return linked


# ─────────────────────────────────────────────────────────────────────────
# Disk budget
# ─────────────────────────────────────────────────────────────────────────

def _dir_size(path: Path) -> int:
    """Total bytes under `path`. Unreadable entries count as 0."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def touch(path: Path) -> None:
    """Mark a cached dir as just-used, for LRU ordering.

    Explicit mtime bump rather than relying on atime, which is unreliable on
    the noatime/relatime mounts typical of an SD-card kiosk.
    """
    try:
        os.utime(path, None)
    except OSError:
        pass


def enforce_disk_budget(
    base_assets_dir: Path, max_bytes: int, protect: set[str] | None = None
) -> int:
    """Evict cached assets until `base_assets_dir` fits in `max_bytes`.

    Composed slides are always regenerable, so eviction is safe — an evicted
    visitor simply pays the compose cost again on their next visit.

    Eviction order is deliberate: duo pairs go first, because the number of
    pairs grows quadratically with visitor count (they are the term that
    actually blows up) and each rebuilds cheaply from the per-person cutouts
    that are kept. Within each group, least-recently-used first.

    `protect` holds directory paths that must not be evicted — whatever is
    on screen right now. Returns the number of bytes reclaimed.
    """
    protect = protect or set()
    if max_bytes <= 0 or not base_assets_dir.is_dir():
        return 0

    total = _dir_size(base_assets_dir)
    if total <= max_bytes:
        return 0

    duo_root = base_assets_dir / "duo"

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def candidates(root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        try:
            entries = [p for p in root.iterdir() if p.is_dir() and p != duo_root]
        except OSError:
            return []
        entries.sort(key=_mtime)
        return entries

    # Duo pairs first, then whole person dirs; LRU within each group.
    ordered = candidates(duo_root) + candidates(base_assets_dir)

    reclaimed = 0
    for entry in ordered:
        if total - reclaimed <= max_bytes:
            break
        if str(entry) in protect:
            continue
        size = _dir_size(entry)
        try:
            shutil.rmtree(entry)
        except OSError as e:
            logger.warning(f"Could not evict {entry}: {e}")
            continue
        reclaimed += size
        logger.info(f"Evicted cached assets: {entry} ({size / 1e6:.1f} MB)")

    if reclaimed:
        logger.info(
            f"Base asset cache trimmed: {reclaimed / 1e6:.1f} MB reclaimed, "
            f"now {(total - reclaimed) / 1e6:.1f} MB / {max_bytes / 1e6:.1f} MB"
        )
    return reclaimed
