"""Pi-side plumbing: systemd units, the app's config overlay, its logs, and
the capture store's records.

The kit never edits config/config.yaml. `app start` writes a copy with the
test's settings into the session folder and points the diagnostic unit at it
through a systemd drop-in (ExecStart override); `app restore` removes the
drop-in, which puts the unit back exactly as setup.sh installed it.
"""

import copy
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from wpu_client.paths import CONFIG_DIR, DATA_DIR, PROJECT_ROOT

UNITS = {
    "diagnostic": "slideshow-diagnostic.service",
    "server": "slideshow-server.service",
    "only": "slideshow-only.service",
}
DROPIN_NAME = "sitetest.conf"
APP_LOG = Path("/var/log/wpu-client/app.log")


# ── shell ───────────────────────────────────────────────────────────────


def run(cmd: list[str], check: bool = False, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def sudo(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run as root. Interactive sudo is fine: the kit is driven from a terminal."""
    prefix = [] if _is_root() else ["sudo"]
    return subprocess.run(prefix + cmd, text=True, check=check)


def _is_root() -> bool:
    import os

    return hasattr(os, "geteuid") and os.geteuid() == 0


def python_exe() -> str:
    venv = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


# ── units ───────────────────────────────────────────────────────────────


def unit_prop(unit: str, prop: str) -> str:
    out = run(["systemctl", "show", "-p", prop, "--value", unit]).stdout.strip()
    return out


def unit_active(unit: str) -> bool:
    return run(["systemctl", "is-active", unit]).stdout.strip() == "active"


def active_unit() -> Optional[str]:
    for unit in UNITS.values():
        if unit_active(unit):
            return unit
    return None


def unit_state(unit: str) -> dict:
    return {
        "unit": unit,
        "active": unit_prop(unit, "ActiveState"),
        "sub": unit_prop(unit, "SubState"),
        "main_pid": int(unit_prop(unit, "MainPID") or 0),
        "restarts": int(unit_prop(unit, "NRestarts") or 0),
        "since": unit_prop(unit, "ActiveEnterTimestamp"),
        "enabled": run(["systemctl", "is-enabled", unit]).stdout.strip(),
        "dropin": dropin_path(unit).exists(),
    }


def dropin_path(unit: str) -> Path:
    return Path(f"/etc/systemd/system/{unit}.d/{DROPIN_NAME}")


def write_dropin(unit: str, exec_start: str) -> None:
    body = (
        f"# Written by sitetest.py — remove with: python sitetest.py app restore\n"
        f"[Service]\nExecStart=\nExecStart={exec_start}\n"
    )
    tmp = DATA_DIR / "sitetest" / f".{unit}.dropin"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(body)
    path = dropin_path(unit)
    sudo(["mkdir", "-p", str(path.parent)])
    sudo(["install", "-m", "644", str(tmp), str(path)])
    tmp.unlink(missing_ok=True)
    sudo(["systemctl", "daemon-reload"])


def remove_dropin(unit: str) -> bool:
    path = dropin_path(unit)
    if not path.exists():
        return False
    sudo(["rm", "-f", str(path)])
    sudo(["systemctl", "daemon-reload"])
    return True


def restart(unit: str) -> None:
    sudo(["systemctl", "restart", unit])


def stop_all() -> None:
    for unit in UNITS.values():
        sudo(["systemctl", "stop", unit], check=False)


def wait_active(unit: str, timeout: float = 60) -> Optional[float]:
    """Seconds until `unit` is active, or None on timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if unit_active(unit):
            return time.time() - start
        time.sleep(0.5)
    return None


# ── config overlay ──────────────────────────────────────────────────────


def base_config() -> dict:
    for path in (CONFIG_DIR / "config.yaml", CONFIG_DIR / "config.yaml.example"):
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    return {}


def overlay_config(
    base: dict,
    *,
    scene_source: str,
    threshold: Optional[float],
    capture: bool,
    empty_every: float,
    min_face: Optional[int],
    log_level: Optional[str],
    capture_dir: Optional[str] = None,
    fallback: Optional[bool] = None,
) -> dict:
    cfg = copy.deepcopy(base)
    fr = cfg.setdefault("services", {}).setdefault("face_recognition", {})
    fr["diagnostic_mode"] = True
    fr["scene_source"] = scene_source
    fr.pop("use_legacy_final_images", None)
    if fallback is not None:
        fr["scene_source_fallback"] = fallback
    if threshold is not None:
        fr["diagnostic_match_threshold"] = threshold
    if min_face is not None:
        fr["min_face_size"] = min_face
    fr["diagnostic_gallery_reload_s"] = fr.get("diagnostic_gallery_reload_s", 5) or 5
    cs = fr.setdefault("capture_store", {})
    cs["enabled"] = capture
    cs["save_empty_every_s"] = empty_every
    if capture_dir:
        cs["dir"] = capture_dir
    if log_level:
        cfg["log_level"] = log_level
    return cfg


def effective(cfg: dict) -> dict:
    """The face-recognition settings a test result should carry."""
    from wpu_client.config.settings import Settings

    fr = Settings(**cfg).services.face_recognition
    return {
        "scene_source": fr.scene_source,
        "scene_source_fallback": fr.scene_source_fallback,
        "threshold": fr.diagnostic_match_threshold,
        "min_face_size": fr.min_face_size,
        "person_timeout": fr.person_timeout,
        "detection_interval": fr.detection_interval,
        "model": fr.model,
        "capture": fr.capture_store.enabled,
        "capture_dir": fr.capture_store.dir,
        "gallery_dir": fr.diagnostic_gallery_dir,
    }


def gallery_dir(cfg: Optional[dict] = None) -> Path:
    fr = ((cfg or base_config()).get("services") or {}).get("face_recognition") or {}
    path = Path(fr.get("diagnostic_gallery_dir") or "data/embeddings")
    return path if path.is_absolute() else PROJECT_ROOT / path


def captures_dir(cfg: Optional[dict] = None) -> Path:
    fr = ((cfg or base_config()).get("services") or {}).get("face_recognition") or {}
    path = Path((fr.get("capture_store") or {}).get("dir") or "data/captures")
    return path if path.is_absolute() else PROJECT_ROOT / path


# ── logs → events ───────────────────────────────────────────────────────

LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - (?P<dev>\S+) - (?P<logger>\S+) - "
    r"(?P<level>[A-Z]+) - (?P<msg>.*)$"
)

# (kind, pattern) — first match wins. Group names become event fields.
PATTERNS = [
    ("match", re.compile(r"^Local match: (?P<name>.+?) dist=(?P<dist>[\d.]+) conf=")),
    (
        "nomatch",
        re.compile(
            r"^No local match within threshold \((?P<thr>[\d.]+) cosine\) — not recognized"
            r"(?: \(nearest (?P<name>.+?) dist=(?P<dist>[\d.]+)\))?"
        ),
    ),
    (
        "detected",
        re.compile(
            r"^Emitted person\.detected: (?P<name>.+?) \(registration_id: (?P<rid>[^,]+), "
            r"sketches: (?P<sketches>.*)\)$"
        ),
    ),
    ("left", re.compile(r"^Emitted person\.left: (?P<name>.+?) \(registration_id: (?P<rid>[^)]+)\)$")),
    ("timeout", re.compile(r"^Person timeout: (?P<name>.+?) not seen for (?P<secs>[\d.]+)s")),
    ("slides", re.compile(r"^Loaded (?P<n>\d+) local slide\(s\)/video\(s\) from (?P<dir>.+)$")),
    ("no_slides", re.compile(r"^No sketches to show for (?P<name>.+?) \(registration_id: (?P<rid>[^)]+)\)")),
    ("display", re.compile(r"^Updated image list: (?P<n>\d+) images, mode=(?P<mode>\w+)")),
    ("stock", re.compile(r"^Switching to stock mode")),
    ("duo_fallback", re.compile(r"^Duo composition unavailable")),
    ("gallery_reload", re.compile(r"^Gallery reloaded: (?P<n>\d+) person")),
    ("loop_start", re.compile(r"^Face recognition loop started")),
    ("capture_drop", re.compile(r"^Capture store falling behind — (?P<n>\d+)")),
]


@dataclass
class LogEvent:
    t: float
    kind: str
    level: str
    msg: str
    fields: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"t": self.t, "kind": self.kind, "level": self.level, "msg": self.msg[:300], **self.fields}


def parse_line(line: str) -> Optional[LogEvent]:
    m = LINE.match(line.rstrip("\n"))
    if not m:
        return None
    try:
        t = time.mktime(time.strptime(m["ts"], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None
    msg, level = m["msg"], m["level"]
    for kind, pattern in PATTERNS:
        hit = pattern.match(msg)
        if hit:
            fields = {k: v for k, v in hit.groupdict().items() if v is not None}
            return LogEvent(t, kind, level, msg, fields)
    if level in ("ERROR", "CRITICAL"):
        return LogEvent(t, "error", level, msg)
    if level == "WARNING":
        return LogEvent(t, "warning", level, msg)
    return None


def app_log_lines(since: float, until: float, unit: Optional[str] = None) -> list[str]:
    """The app's own log lines in [since, until].

    Reads the journal (it also holds what the app could not log itself:
    tracebacks on stderr, restarts); falls back to app.log off-systemd.
    """
    unit = unit or UNITS["diagnostic"]
    cmd = [
        "journalctl",
        "-u",
        unit,
        "--since",
        f"@{int(since) - 1}",
        "--until",
        f"@{int(until) + 1}",
        "-o",
        "cat",
        "--no-pager",
    ]
    if shutil.which("journalctl"):
        try:
            res = run(cmd, timeout=60)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.splitlines()
        except subprocess.TimeoutExpired:
            pass
    lines: list[str] = []
    for path in (APP_LOG.with_name(APP_LOG.name + ".1"), APP_LOG):
        try:
            lines += path.read_text(errors="replace").splitlines()
        except OSError:
            pass
    return lines


def events(since: float, until: float, unit: Optional[str] = None) -> list[LogEvent]:
    out = []
    for line in app_log_lines(since, until, unit):
        ev = parse_line(line)
        if ev and since - 1 <= ev.t <= until + 1:
            out.append(ev)
    return out


# ── capture records ─────────────────────────────────────────────────────


def captures(since: float, until: float, root: Optional[Path] = None) -> list[dict]:
    """Capture-store records with t in [since, until], oldest first."""
    root = root or captures_dir()
    out: list[dict] = []
    if not root.is_dir():
        return out
    days = {time.strftime("%Y%m%d", time.localtime(t)) for t in (since, until)}
    for day in sorted(days):
        index = root / day / "index.jsonl"
        try:
            with open(index) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if since <= rec.get("t", 0) <= until:
                        rec["_root"] = str(root)
                        out.append(rec)
        except OSError:
            continue
    return sorted(out, key=lambda r: r["t"])


def latest_captures(seconds: float, root: Optional[Path] = None) -> list[dict]:
    now = time.time()
    return captures(now - seconds, now, root)
