"""Base-mode asset fetching — download a person's raw SAU/FRU cutouts and
their optional videos from the server, cached on disk.

Base (server-recognition) mode composes each visitor's slides locally from
two raw alpha cutouts instead of downloading pre-composed final images. This
module owns the *fetching* half of that: getting the cutouts (and any videos)
onto local disk and keeping them fresh. base_composer.py owns the compositing
half.

Layout per person (assets_dir = data/base_assets/<registration_id>/):
    raw/            sau.png, fru.png, video_1.mov, ...  (whichever fetched)
    manifest.json   per-asset fetch state
    display/        composed slides + hardlinked videos (written by the caller)

Everything here is BEST-EFFORT by design. A missing cutout, an unreachable
server, a registration with no SAU video — all are normal outcomes that are
logged and skipped, never raised. A visitor with no usable assets simply
produces no composed slides, and the caller falls back.

Two endpoints, two very different identification schemes
────────────────────────────────────────────────────────
Cutouts come from `GET <wpu_endpoint>?registration_id=…`, which labels them
(`sau_signed_url` / `fru_signed_url`) and also returns the older flat
`signed_urls` list. Labels win; the flat list is matched by filename suffix
only as a fallback, which works because the server writes cutouts to fixed,
named keys (`<registration_id>/wpu/sau_cutout.png`).

Videos come from `GET <sau_media_endpoint>/<registration_id>` — a PATH
param, a different response shape, and crucially **positional**: the server
stores every upload under a generated `{uuid}{ext}` key, so the original
filename is gone by the time a URL reaches us and there is nothing to match
on. The capture station uploads the composited clip first, so `video_urls[0]`
(== the legacy scalar `video_url`) is the one worth showing.

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

# Videos are stored locally as video_<n><suffix>. The suffix is taken from the
# server's own object key so the slideshow's extension-based video detection
# still recognises the file; anything it wouldn't play falls back to .mp4
# (the content type the upload route defaults to).
VIDEO_LOCAL_PREFIX = "video_"
VIDEO_SUFFIXES = (".mov", ".mp4")
DEFAULT_VIDEO_SUFFIX = ".mp4"


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


def _get_json(http_client: HTTPClient, url: str, params: dict | None = None) -> dict:
    """Single-attempt GET of a JSON object. Returns {} on any failure.

    Deliberately not http_client.get(): that retries, and these listings run
    on the recognition thread where a definitive 404 must not cost
    max_retries × timeout of stalled recognition.
    """
    try:
        data = http_client.get_json_once(url, params=params)
    except Exception as e:  # a broken/legacy client must not take the loop down
        logger.info(f"Asset listing failed at {url}: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def _str_or_none(value) -> str | None:
    """A non-empty string, or None — JSON fields here are all nullable."""
    return value if isinstance(value, str) and value else None


def _match_url(urls: list[str], filename: str) -> str | None:
    """Find the URL whose path ends with `filename`.

    Fallback for the older unlabelled `signed_urls` list. Safe for cutouts
    because the server writes those to fixed named keys
    (`<registration_id>/wpu/<filename>`) and returns raw object URLs, so the
    filename survives in the path. It is NOT usable for videos, whose keys
    are generated UUIDs — see `_fetch_video_urls`.
    """
    if not filename:
        return None
    for url in urls:
        if urlparse(url).path.endswith(filename):
            return url
    return None


def _fetch_cutout_urls(
    http_client: HTTPClient,
    endpoint: str,
    registration_id: str,
    sau_filename: str,
    fru_filename: str,
) -> dict:
    """Resolve this person's SAU/FRU cutout URLs. Returns {"sau": …, "fru": …}.

    Prefers the server's explicit labels over guessing from the flat list:
    they are unambiguous, and a registration with only one of the two cutouts
    yields a one-entry `signed_urls` that position alone can't disambiguate.
    """
    urls: dict = {"sau": None, "fru": None}
    if not endpoint:
        return urls

    data = _get_json(http_client, endpoint, params={"registration_id": registration_id})
    if not data:
        return urls

    urls["sau"] = _str_or_none(data.get("sau_signed_url"))
    urls["fru"] = _str_or_none(data.get("fru_signed_url"))
    if urls["sau"] and urls["fru"]:
        return urls

    listed = [u for u in (data.get("signed_urls") or []) if isinstance(u, str)]
    urls["sau"] = urls["sau"] or _match_url(listed, sau_filename)
    urls["fru"] = urls["fru"] or _match_url(listed, fru_filename)
    return urls


def _fetch_video_urls(
    http_client: HTTPClient, endpoint: str, registration_id: str, limit: int
) -> list[str]:
    """Resolve this person's SAU video URLs, in capture order.

    `endpoint` is the *base* — the registration id goes in the PATH
    (`/api/v1/sau/media/<id>`), not a query string. The response carries
    `video_urls` (all captured videos, index 0 = the composited clip) plus
    `video_url`, the pre-multi-video scalar kept for older clients; the
    scalar is only read when the list is absent, so a server predating the
    list still works.

    Only the first `limit` are taken. Videos are picked BY POSITION because
    the server's object keys are generated UUIDs with no trace of the
    original filename.
    """
    if not endpoint or limit <= 0:
        return []

    data = _get_json(http_client, f"{endpoint.rstrip('/')}/{registration_id}")
    if not data:
        return []

    listed = [u for u in (data.get("video_urls") or []) if _str_or_none(u)]
    if not listed:
        single = _str_or_none(data.get("video_url"))
        listed = [single] if single else []
    return listed[:limit]


def _video_local_name(index: int, url: str) -> str:
    """Local filename for the nth video, keeping the server's extension."""
    suffix = os.path.splitext(urlparse(url).path)[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = DEFAULT_VIDEO_SUFFIX
    return f"{VIDEO_LOCAL_PREFIX}{index}{suffix}"


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
    url: str | None,
    local_name: str,
    raw_dir: Path,
    manifest: dict,
    key: str,
) -> tuple[Path | None, bool]:
    """Download one already-resolved asset URL to a local file.

    The caller resolves the URL — by label for cutouts, by position for
    videos — so this stays a pure fetch-and-cache step with one policy for
    both. `url=None` means the server didn't offer this asset.

    Returns (local_path_or_None, changed) where `changed` is True only when
    fresh bytes were just written — the signal the caller needs to throw away
    anything composed from the previous copy.
    """
    record = manifest.get(key) or {}
    local_path = raw_dir / local_name

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


def _is_video_file(name: str) -> bool:
    """Whether a filename is one of ours in raw/ or display/."""
    return name.startswith(VIDEO_LOCAL_PREFIX) and name.lower().endswith(VIDEO_SUFFIXES)


def _cached_videos(raw_dir: Path) -> list[Path]:
    """Previously-fetched videos still on disk, in stable order."""
    try:
        return sorted(p for p in raw_dir.iterdir() if p.is_file() and _is_video_file(p.name))
    except OSError:
        return []


def _prune_videos(raw_dir: Path, keep: set[str], manifest: dict) -> None:
    """Delete cached videos the server no longer offers under that name."""
    for stale in _cached_videos(raw_dir):
        if stale.name in keep:
            continue
        try:
            stale.unlink()
            logger.info(f"Removed superseded video {stale.name}")
        except OSError as e:
            logger.warning(f"Could not remove superseded video {stale}: {e}")
            continue
        manifest.pop(os.path.splitext(stale.name)[0], None)


def fetch_person_assets(
    http_client: HTTPClient,
    wpu_endpoint: str,
    sau_media_endpoint: str,
    registration_id: str,
    sau_filename: str,
    fru_filename: str,
    video_count: int,
    assets_dir: Path,
) -> dict:
    """Best-effort fetch of this person's SAU/FRU cutouts + videos.

    Downloads into `assets_dir/raw/`, cached via `assets_dir/manifest.json`.
    Never raises — every failure mode degrades to a None/absent entry.

    Args:
        wpu_endpoint: cutout listing URL, queried with ?registration_id=…
        sau_media_endpoint: BASE of the SAU media route; the registration id
            is appended as a path segment.
        video_count: how many of the returned videos to keep, from the front.
            1 takes just the composited clip.

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

        cutout_urls = _fetch_cutout_urls(
            http_client, wpu_endpoint, registration_id, sau_filename, fru_filename
        )
        video_urls = _fetch_video_urls(
            http_client, sau_media_endpoint, registration_id, video_count
        )

        for key, local_name in (("sau", SAU_LOCAL_NAME), ("fru", FRU_LOCAL_NAME)):
            path, changed = _fetch_one(
                http_client, cutout_urls[key], local_name, raw_dir, manifest, key,
            )
            result[key] = path
            if changed:
                result["changed"].add(key)

        keep_video_names = set()
        for index, url in enumerate(video_urls, start=1):
            key = f"{VIDEO_LOCAL_PREFIX}{index}"
            local_name = _video_local_name(index, url)
            keep_video_names.add(local_name)
            path, changed = _fetch_one(
                http_client, url, local_name, raw_dir, manifest, key,
            )
            if path is not None:
                result["videos"].append(path)
            if changed:
                result["changed"].add(key)

        if video_urls or video_count <= 0:
            # A re-upload can change a video's extension, and video_count can
            # be lowered (to 0, disabling videos entirely) — either way the
            # superseded file would otherwise linger in raw/ forever, charged
            # against the disk budget.
            _prune_videos(raw_dir, keep_video_names, manifest)
        else:
            # Videos are wanted but the listing came back empty, which an
            # unreachable endpoint and a genuinely video-less registration
            # produce alike. Reuse whatever is already cached rather than let
            # a network blip cost a visitor the video they had last time.
            result["videos"] = _cached_videos(raw_dir)

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

    Videos the caller no longer lists are removed from display/ first, so a
    re-uploaded or dropped clip stops appearing in the slideshow without
    having to discard the composed slides alongside it. Only `video_*` names
    are touched — composed slides use their own prefixes.
    """
    keep = {video.name for video in videos}
    try:
        for existing in display_dir.iterdir():
            if existing.is_file() and _is_video_file(existing.name) and existing.name not in keep:
                existing.unlink()
                logger.info(f"Removed stale video slide {existing.name}")
    except OSError:
        pass

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
    """Total bytes actually occupied under `path`. Unreadable entries count as 0.

    Videos are hardlinked from raw/ into display/, so the same blocks appear
    under two names. Counting each inode once keeps the budget honest instead
    of double-charging every video and evicting sooner than necessary.
    """
    total = 0
    seen: set = set()
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                stat = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.st_nlink > 1:
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += stat.st_size
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

    `protect` holds paths that must not be evicted — whatever is on screen
    right now. An entry is spared when it is, or contains, a protected path,
    so callers can pass either the person/pair dir or the display/ dir inside
    it without having to know which granularity eviction works at.

    Returns the number of bytes reclaimed.
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

    def _is_protected(entry: Path) -> bool:
        """True if `entry` is, or contains, anything the caller pinned."""
        entry_str = str(entry)
        return any(p == entry_str or p.startswith(entry_str + os.sep) for p in protect)

    reclaimed = 0
    for entry in ordered:
        if total - reclaimed <= max_bytes:
            break
        if _is_protected(entry):
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
