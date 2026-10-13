"""Telemetry: one CSV row every few seconds for the whole site day.

SoC temperature, the firmware's throttling flags, CPU clock, CPU and memory
load, the app's own memory (leak check), disk, fan speed where there is one,
and the active unit with its restart count. `start` detaches a sampler that
outlives the terminal; a reboot ends it (`fault power --verify` restarts it).
"""

import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

from wpu_client.sitetest import system
from wpu_client.sitetest.session import Session

FIELDS = [
    "t",
    "local",
    "temp_c",
    "throttled_hex",
    "under_voltage",
    "freq_capped",
    "throttled",
    "soft_temp_limit",
    "uv_occurred",
    "throttle_occurred",
    "arm_mhz",
    "cpu_pct",
    "load1",
    "mem_used_pct",
    "app_rss_mb",
    "app_cpu_pct",
    "disk_free_gb",
    "fan_rpm",
    "unit",
    "restarts",
]

# vcgencmd get_throttled bits
BITS = {
    "under_voltage": 0,
    "freq_capped": 1,
    "throttled": 2,
    "soft_temp_limit": 3,
    "uv_occurred": 16,
    "throttle_occurred": 18,
}


def decode_throttled(value: int) -> dict:
    return {name: int(bool(value >> bit & 1)) for name, bit in BITS.items()}


def _vcgencmd(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_temp() -> Optional[float]:
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except (OSError, ValueError):
        out = _vcgencmd("measure_temp")  # temp=48.3'C
        try:
            return float(out.split("=")[1].split("'")[0]) if out else None
        except (IndexError, ValueError):
            return None


def read_throttled() -> Optional[int]:
    out = _vcgencmd("get_throttled")  # throttled=0x50000
    try:
        return int(out.split("=")[1], 16) if out else None
    except (IndexError, ValueError):
        return None


def read_arm_mhz() -> Optional[int]:
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text()) // 1000
    except (OSError, ValueError):
        return None


def read_fan() -> Optional[int]:
    for path in Path("/sys/devices/platform").glob("cooling_fan/hwmon/*/fan1_input"):
        try:
            return int(path.read_text())
        except (OSError, ValueError):
            pass
    return None


class _AppProc:
    """The app's main process, re-found whenever the unit restarts."""

    def __init__(self):
        self.pid = 0
        self.proc: Optional[psutil.Process] = None

    def sample(self, unit: Optional[str]) -> tuple[Optional[float], Optional[float]]:
        pid = int(system.unit_prop(unit, "MainPID") or 0) if unit else 0
        if pid != self.pid:
            self.pid = pid
            self.proc = psutil.Process(pid) if pid else None
            if self.proc:
                self.proc.cpu_percent(None)
        if not self.proc:
            return None, None
        try:
            return round(self.proc.memory_info().rss / 1e6, 1), self.proc.cpu_percent(None)
        except psutil.Error:
            self.pid, self.proc = 0, None
            return None, None


def sample_row(app: _AppProc) -> dict:
    unit = system.active_unit()
    thr = read_throttled()
    rss, app_cpu = app.sample(unit)
    row = {
        "t": round(time.time(), 1),
        "local": time.strftime("%H:%M:%S"),
        "temp_c": read_temp(),
        "throttled_hex": hex(thr) if thr is not None else "",
        "arm_mhz": read_arm_mhz(),
        "cpu_pct": psutil.cpu_percent(None),
        "load1": round(os.getloadavg()[0], 2),
        "mem_used_pct": psutil.virtual_memory().percent,
        "app_rss_mb": rss,
        "app_cpu_pct": app_cpu,
        "disk_free_gb": round(psutil.disk_usage(str(system.PROJECT_ROOT)).free / 1e9, 2),
        "fan_rpm": read_fan(),
        "unit": (unit or "none").replace(".service", ""),
        "restarts": int(system.unit_prop(unit, "NRestarts") or 0) if unit else "",
    }
    row.update(decode_throttled(thr) if thr is not None else {k: "" for k in BITS})
    return row


def run_sampler(session: Session, interval: float) -> None:
    """Foreground loop (what `start` detaches)."""
    path = session.telemetry_csv
    new = not path.exists()
    app = _AppProc()
    psutil.cpu_percent(None)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        while not stop["flag"]:
            try:
                writer.writerow(sample_row(app))
                f.flush()
            except Exception as e:  # a bad sample must not end a day-long log
                print(f"telemetry sample failed: {e}", file=sys.stderr)
            time.sleep(interval)


def pidfile(session: Session) -> Path:
    return session.file("telemetry.pid")


def running_pid(session: Session) -> Optional[int]:
    try:
        pid = int(pidfile(session).read_text())
        return pid if psutil.pid_exists(pid) else None
    except (OSError, ValueError):
        return None


def start(session: Session, interval: float, entry: Path) -> int:
    pid = running_pid(session)
    if pid:
        return pid
    log = open(session.file("telemetry.log"), "a")
    proc = subprocess.Popen(
        [sys.executable, str(entry), "telemetry", "run", "--interval", str(interval)],
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(system.PROJECT_ROOT),
    )
    pidfile(session).write_text(str(proc.pid))
    return proc.pid


def stop(session: Session) -> bool:
    pid = running_pid(session)
    if not pid:
        return False
    os.kill(pid, signal.SIGTERM)
    pidfile(session).unlink(missing_ok=True)
    return True


# ── reading it back ─────────────────────────────────────────────────────


def load(session: Session, since: float = 0, until: float = float("inf")) -> list[dict]:
    try:
        with open(session.telemetry_csv, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    out = []
    for r in rows:
        try:
            t = float(r["t"])
        except (KeyError, ValueError):
            continue
        if since <= t <= until:
            out.append(r)
    return out


def _num(rows: list[dict], key: str) -> list[float]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (KeyError, ValueError, TypeError):
            pass
    return vals


def summarise(rows: list[dict]) -> dict:
    """Peaks and flags over a window of telemetry rows."""
    if not rows:
        return {"samples": 0}
    temps, mhz, rss = _num(rows, "temp_c"), _num(rows, "arm_mhz"), _num(rows, "app_rss_mb")
    flags = {
        k: any(r.get(k) == "1" for r in rows)
        for k in ("under_voltage", "throttled", "freq_capped", "soft_temp_limit")
    }
    restarts = _num(rows, "restarts")
    return {
        "samples": len(rows),
        "minutes": round((float(rows[-1]["t"]) - float(rows[0]["t"])) / 60, 1),
        "temp_c": {
            "min": min(temps, default=None),
            "max": max(temps, default=None),
            "mean": round(sum(temps) / len(temps), 1) if temps else None,
        },
        "arm_mhz": {"min": min(mhz, default=None), "max": max(mhz, default=None)},
        "cpu_pct_mean": round(sum(_num(rows, "cpu_pct")) / len(rows), 1),
        "app_rss_mb": {
            "first": rss[0] if rss else None,
            "last": rss[-1] if rss else None,
            "max": max(rss, default=None),
        },
        "disk_free_gb_min": min(_num(rows, "disk_free_gb"), default=None),
        "flags_seen": flags,
        "restarts_delta": (max(restarts) - min(restarts)) if restarts else None,
    }


def verdict(summary: dict) -> tuple[str, list[str]]:
    """PASS / WARN / FAIL for a telemetry window, with reasons."""
    if not summary.get("samples"):
        return "FAIL", ["no telemetry samples — is `telemetry start` running?"]
    rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    state = {"level": "PASS"}
    reasons: list[str] = []

    def flag(level: str, reason: str) -> None:
        reasons.append(reason)
        if rank[level] > rank[state["level"]]:
            state["level"] = level

    tmax = (summary["temp_c"] or {}).get("max")
    flags = summary["flags_seen"]
    if flags.get("under_voltage"):
        flag("FAIL", "under-voltage reported — PSU or cable too weak")
    if flags.get("throttled") or flags.get("freq_capped"):
        flag("FAIL", "CPU throttled / frequency capped")
    if tmax is not None and tmax >= 80:
        flag("FAIL", f"temperature reached {tmax} °C (throttling starts at 80-85)")
    elif tmax is not None and tmax >= 75:
        flag("WARN", f"temperature reached {tmax} °C — close to the throttle point")
    if flags.get("soft_temp_limit"):
        flag("WARN", "soft temperature limit hit")
    rss = summary.get("app_rss_mb") or {}
    if rss.get("first") and rss.get("last") and summary.get("minutes", 0) >= 60:
        growth = rss["last"] - rss["first"]
        if growth > 150:
            flag("WARN", f"app memory grew {growth:.0f} MB over {summary['minutes']} min — possible leak")
    if summary.get("restarts_delta"):
        flag("WARN", f"app restarted {summary['restarts_delta']:.0f} time(s) in the window")
    if not reasons:
        reasons.append("temperature, power and clock within limits")
    return state["level"], reasons
