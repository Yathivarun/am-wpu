"""Pre-flight checks — `python main.py --check`.

Answers "is this unit actually going to work?" without starting anything, and
says so in one screen. Built for two callers with the same needs:

  a person bringing up a Pi, who wants the reason it is showing stock images
  a fleet tool (Ansible, monitoring), which wants an exit code and JSON

Every check is therefore non-destructive, bounded in time, and degrades to a
report rather than a traceback: an import that fails is a FAIL line, not a
stack trace, because the checks most worth running are exactly the ones you
run when the environment is broken. Nothing here may import picamera2, cv2 or
onnxruntime at module scope for the same reason.

Statuses, and what a fleet should do with them:

    ok      nothing to do
    warn    works, but not how you probably intended — a Pi running on
            built-in defaults, or scene art that is only partly present
    fail    this unit will not do its job

The exit code is 1 if anything FAILed, else 0. Warnings never fail a run;
a fleet that wants them to can read the JSON.
"""

import json
import os
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from wpu_client.paths import DATA_DIR, MODELS_DIR

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Every unit that holds the camera and the screen. Used to decide whether the
# camera check may safely open the camera.
UNITS = (
    "slideshow-server.service",
    "slideshow-diagnostic.service",
    "slideshow-only.service",
)

REQUIRED_MODELS = (
    "mobilefacenet.onnx",
    "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx",
)

# (import name, human name). Version is read from __version__ where present.
REQUIRED_MODULES = (
    ("numpy", "numpy"),
    ("cv2", "opencv"),
    ("onnxruntime", "onnxruntime"),
    ("yaml", "pyyaml"),
    ("pydantic", "pydantic"),
    ("scipy", "scipy"),
    ("httpx", "httpx"),
    ("psutil", "psutil"),
)

SCENE_DIRS = ("sau_single", "sau_duo", "fru_single", "fru_duo")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

LOG_DIR = Path("/var/log/wpu-client")

# Long enough to cross a congested Pi LAN, short enough that a checked fleet
# of 50 units does not take an hour when the server is down.
PROBE_TIMEOUT = 5.0

# A composed slide is a few hundred KB and the cache is size-capped, but a
# full disk breaks logging and the dataset writer too.
MIN_FREE_BYTES = 500 * 1024 * 1024


@dataclass
class Result:
    """One check's outcome."""

    name: str
    status: str
    detail: str


def _version(module) -> str:
    for attr in ("__version__", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    return "?"


def _unit_active(unit: str) -> bool:
    try:
        done = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            timeout=5, capture_output=True,
        )
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def active_units() -> list[str]:
    """Which slideshow units are currently running, if systemd is available."""
    if shutil.which("systemctl") is None:
        return []
    return [u for u in UNITS if _unit_active(u)]


# ── individual checks ───────────────────────────────────────────────────


def check_dependencies() -> Result:
    """Every runtime import, and the NumPy ABI pin picamera2 depends on."""
    import importlib

    missing, versions = [], []
    for import_name, label in REQUIRED_MODULES:
        try:
            module = importlib.import_module(import_name)
        except Exception as e:
            missing.append(f"{label} ({type(e).__name__})")
            continue
        versions.append(f"{label} {_version(module)}")

    if missing:
        return Result("deps", FAIL, f"missing: {', '.join(missing)}")

    import numpy

    if numpy.__version__.startswith("2."):
        return Result(
            "deps", FAIL,
            f"numpy {numpy.__version__} breaks picamera2's C ABI — "
            f"pip install --force-reinstall 'numpy==1.26.4'",
        )
    return Result("deps", OK, ", ".join(versions))


def check_picamera() -> Result:
    """picamera2 and GTK come from apt, not the venv — a venv built without
    --system-site-packages resolves neither, and the failure only shows up
    when a service starts."""
    problems = []
    try:
        import picamera2  # noqa: F401
    except Exception as e:
        problems.append(f"picamera2 ({type(e).__name__})")
    try:
        import gi

        gi.require_version("Gtk", "4.0")
    except Exception as e:
        problems.append(f"gtk4 ({type(e).__name__})")

    if problems:
        return Result(
            "system", FAIL,
            f"{', '.join(problems)} — venv needs --system-site-packages, "
            f"and apt needs python3-picamera2 python3-gi gir1.2-gtk-4.0",
        )
    return Result("system", OK, "picamera2 + gtk4 import")


def check_models() -> Result:
    """A missing recogniser aborts the recognition thread at startup and the
    kiosk shows stock images forever, with nothing on screen to say why."""
    missing = [m for m in REQUIRED_MODELS if not (MODELS_DIR / m).is_file()]
    if missing:
        return Result("models", FAIL, f"missing: {', '.join(missing)}")
    total = sum((MODELS_DIR / m).stat().st_size for m in REQUIRED_MODELS)
    return Result("models", OK, f"{len(REQUIRED_MODELS)}/3 present ({total // 1048576} MB)")


def check_config(settings, config_path: Path) -> Result:
    """Report what this unit will actually talk to.

    A missing config is a warning, not a failure: the client falls back to
    built-in defaults and still runs. On a fleet that is nearly always wrong
    though — the defaults name one hardcoded host — so it is called out.
    """
    face = settings.services.face_recognition
    host = urlparse(face.api_endpoint).netloc or "?"
    summary = (
        f"host={host} model={face.model} "
        f"scale_mode={settings.services.slideshow.scale_mode}"
    )
    if not config_path.is_file():
        return Result(
            "config", WARN,
            f"no {config_path} — running on built-in defaults ({summary})",
        )
    if face.model != "mobilenet":
        return Result(
            "config", WARN,
            f"{summary} — 'sface' queries the 128-D gallery, which is empty "
            f"in production and matches nobody",
        )
    return Result("config", OK, summary)


def check_scenes() -> Result:
    """Scene art is what composition draws onto. Without it a recognised
    visitor composes nothing and silently falls back."""
    scenes_root = DATA_DIR / "base_scenes"
    if not scenes_root.is_dir():
        return Result("scenes", FAIL, f"{scenes_root} does not exist")

    counts, empty, unmatched = [], [], 0
    for name in SCENE_DIRS:
        scene_dir = scenes_root / name
        images = [
            p for p in scene_dir.iterdir()
            if scene_dir.is_dir() and p.suffix.lower() in IMAGE_SUFFIXES
        ] if scene_dir.is_dir() else []
        counts.append(f"{name} {len(images)}")
        if not images:
            empty.append(name)

        config_path = scene_dir / "scenes_config.json"
        if config_path.is_file():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                stems = {p.stem for p in images}
                unmatched += len([k for k in config if k not in stems])
            except (OSError, ValueError):
                return Result("scenes", FAIL, f"unreadable {config_path}")

    if len(empty) == len(SCENE_DIRS):
        return Result("scenes", FAIL, "no scene art at all — nothing can be composed")
    if empty:
        return Result("scenes", FAIL, f"no art in: {', '.join(empty)}")
    detail = ", ".join(counts)
    if unmatched:
        return Result("scenes", WARN, f"{detail} — {unmatched} config key(s) have no image")
    return Result("scenes", OK, detail)


def check_gallery(settings) -> Result:
    """Diagnostic mode only: seeded people, and whether their cutouts resolve.

    A gallery folder holding finished slides rather than the two raw cutouts
    is the common mistake, and it composes nothing.
    """
    from wpu_client.services.face_recognition.diagnostic_gallery import DiagnosticGallery
    from wpu_client.services.face_recognition.local_assets import resolve_local_cutouts

    gallery = DiagnosticGallery(settings.services.face_recognition.diagnostic_gallery_dir)
    loaded = gallery.load()
    if not loaded:
        return Result(
            "gallery", FAIL,
            f"no people in {settings.services.face_recognition.diagnostic_gallery_dir} — "
            f"seed with scripts/seed_face.py",
        )

    incomplete = []
    for entry in gallery.entries():
        cutouts = resolve_local_cutouts(entry.sketch_dir, entry.meta)
        have = [k for k in ("sau", "fru") if cutouts[k]]
        if len(have) < 2:
            incomplete.append(f"{entry.slug}({'+'.join(have) or 'none'})")

    if incomplete:
        return Result(
            "gallery", WARN,
            f"{loaded} person(s), incomplete cutouts: {', '.join(incomplete)}",
        )
    return Result("gallery", OK, f"{loaded} person(s), all with sau+fru cutouts")


def check_server(settings) -> Result:
    """Base mode only: can this unit reach the registration server?

    Any HTTP response counts — /identify is POST-only and answers a GET with
    405, which still proves the host is up and routable. Only a transport
    failure is a FAIL.
    """
    import httpx

    endpoint = settings.services.face_recognition.api_endpoint
    host = urlparse(endpoint).netloc or endpoint
    try:
        response = httpx.get(endpoint, timeout=PROBE_TIMEOUT)
        return Result("server", OK, f"{host} responded HTTP {response.status_code}")
    except httpx.HTTPError as e:
        return Result("server", FAIL, f"{host} unreachable ({type(e).__name__}: {e})")


def check_camera() -> Result:
    """Can the camera be claimed?

    Skipped while a slideshow unit is running — that unit holds the camera
    exclusively, and opening it here would either fail spuriously or, worse,
    take it away from a live kiosk.
    """
    running = active_units()
    if running:
        return Result("camera", OK, f"held by {running[0]} (running) — not probed")

    try:
        from picamera2 import Picamera2
    except Exception as e:
        return Result("camera", FAIL, f"picamera2 unavailable ({type(e).__name__})")

    try:
        camera = Picamera2()
    except Exception as e:
        return Result("camera", FAIL, f"cannot open camera ({type(e).__name__}: {e})")
    try:
        models = getattr(camera, "camera_properties", {}).get("Model", "camera")
        return Result("camera", OK, f"{models} claimable")
    finally:
        try:
            camera.close()
        except Exception:
            pass


def check_log_dir() -> Result:
    """main.py writes app.log here at import; an unwritable dir stops the
    service before any of its own logging can report why."""
    if not LOG_DIR.is_dir():
        return Result("logs", FAIL, f"{LOG_DIR} missing — sudo mkdir -p {LOG_DIR}")
    if not os.access(LOG_DIR, os.W_OK):
        return Result(
            "logs", FAIL,
            f"{LOG_DIR} not writable by {os.getenv('USER', 'this user')} — "
            f"sudo chown $(id -un) {LOG_DIR}",
        )
    return Result("logs", OK, str(LOG_DIR))


def check_disk() -> Result:
    """Composed slides, the dataset writer and the log all need headroom."""
    try:
        usage = shutil.disk_usage(DATA_DIR)
    except OSError as e:
        return Result("disk", WARN, f"could not stat {DATA_DIR} ({e})")
    free_mb = usage.free // 1048576
    detail = f"{free_mb} MB free of {usage.total // 1048576} MB"
    if usage.free < MIN_FREE_BYTES:
        return Result("disk", FAIL, detail)
    return Result("disk", OK, detail)


# ── driver ──────────────────────────────────────────────────────────────


def run_checks(settings, diagnostic: bool, config_path: Path) -> list[Result]:
    """Every check applicable to this unit, in report order.

    Mode decides two of them: diagnostic mode never talks to a server, and
    only diagnostic mode has a gallery to inspect. Running the wrong one
    would report a failure that does not matter on this unit.
    """
    results = [
        check_dependencies(),
        check_picamera(),
        check_models(),
        check_config(settings, config_path),
        check_scenes(),
    ]
    if diagnostic:
        results.append(check_gallery(settings))
    else:
        results.append(check_server(settings))
    results += [check_camera(), check_log_dir(), check_disk()]
    return results


def render_text(results: list[Result], diagnostic: bool) -> str:
    """One aligned line per check, plus a verdict."""
    width = max(len(r.name) for r in results)
    lines = [f"wpu-client pre-flight — {socket.gethostname()} — "
             f"{'diagnostic' if diagnostic else 'base'} mode", ""]
    for r in results:
        lines.append(f"  {r.name.ljust(width)}  {r.status.upper():<4}  {r.detail}")

    failed = [r.name for r in results if r.status == FAIL]
    warned = [r.name for r in results if r.status == WARN]
    lines.append("")
    if failed:
        lines.append(f"FAIL — {len(failed)} check(s) failed: {', '.join(failed)}")
    elif warned:
        lines.append(f"OK with warnings — review: {', '.join(warned)}")
    else:
        lines.append("OK — this unit is ready to run.")
    return "\n".join(lines)


def render_json(results: list[Result], diagnostic: bool) -> str:
    failed = [r.name for r in results if r.status == FAIL]
    return json.dumps(
        {
            "host": socket.gethostname(),
            "mode": "diagnostic" if diagnostic else "base",
            "status": FAIL if failed else OK,
            "failed": failed,
            "warned": [r.name for r in results if r.status == WARN],
            "checks": [asdict(r) for r in results],
        },
        indent=2,
    )


def exit_code(results: list[Result]) -> int:
    """1 if anything failed. Warnings never fail a run — a fleet that wants
    them to can read `warned` out of the JSON."""
    return 1 if any(r.status == FAIL for r in results) else 0
