"""Fault and load tests: what happens when things go wrong on site.

kill     SIGKILL the app; systemd must bring it back and recognition resume.
disk     fill the disk to N% for a while; the app must keep running. The
         balloon file is always removed, also on Ctrl+C (`fault disk --cleanup`
         removes a leftover one).
display  guided HDMI unplug/replug; the slideshow must come back by itself.
power    two steps around pulling the plug: `--arm` before, `--verify` after
         the Pi boots; checks it came back in the right mode, how fast, and
         whether the filesystem complained.
stress   hold every core at 100% while the app runs; telemetry judges the
         temperature, throttling and recognition slowdown.

Camera unplug is deliberately not here: the CSI ribbon is not hot-pluggable
and pulling it live can damage the camera or the Pi.
"""

import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import psutil

from wpu_client.sitetest import scenarios, system, telemetry
from wpu_client.sitetest.session import Session

APP_UNIT = system.UNITS["diagnostic"]
BALLOON = system.DATA_DIR / "sitetest" / ".disk_balloon"


def _first_capture_after(t: float, timeout: float) -> float | None:
    """Seconds after `t` until the capture store wrote a record, or None."""
    end = time.time() + timeout
    while time.time() < end:
        if system.captures(t, time.time() + 1):
            return round(min(r["t"] for r in system.captures(t, time.time() + 1)) - t, 1)
        time.sleep(1)
    return None


def _first_event_after(kind: str, t: float, timeout: float) -> float | None:
    end = time.time() + timeout
    while time.time() < end:
        hits = [e for e in system.events(t, time.time()) if e.kind == kind]
        if hits:
            return round(hits[0].t - t, 1)
        time.sleep(2)
    return None


# ── kill ────────────────────────────────────────────────────────────────


def kill(session: Session, timeout: float = 90) -> None:
    before = system.unit_state(APP_UNIT)
    if before["active"] != "active" or not before["main_pid"]:
        raise SystemExit(f"{APP_UNIT} is not running — start it first: python sitetest.py app start")
    t0 = time.time()
    system.sudo(["kill", "-9", str(before["main_pid"])])
    print(f"killed pid {before['main_pid']} — waiting for systemd to restart it ...")
    back = None
    while time.time() - t0 < timeout:
        st = system.unit_state(APP_UNIT)
        if st["active"] == "active" and st["main_pid"] not in (0, before["main_pid"]):
            back = round(time.time() - t0, 1)
            break
        time.sleep(0.5)
    loop = _first_event_after("loop_start", t0, timeout) if back is not None else None
    after = system.unit_state(APP_UNIT)
    details = {"before": before, "after": after, "unit_back_s": back, "recognition_loop_s": loop}
    if back is None:
        session.record(
            "fault-kill",
            "FAIL",
            f"app not back within {timeout:.0f} s",
            details,
            reasons=["systemd did not restart the unit — check Restart= in the unit file"],
            started=t0,
        )
    elif loop is None:
        session.record(
            "fault-kill",
            "FAIL",
            "unit restarted but recognition never started",
            details,
            reasons=[f"unit active after {back} s, no 'Face recognition loop started'"],
            started=t0,
        )
    else:
        verdict = "PASS" if loop <= 30 else "WARN"
        session.record(
            "fault-kill",
            verdict,
            f"recovered: unit {back} s, recognition {loop} s",
            details,
            reasons=[f"systemd restart in {back} s", f"recognition loop running {loop} s after kill"],
            started=t0,
        )


# ── disk ────────────────────────────────────────────────────────────────


def disk_cleanup() -> bool:
    if BALLOON.exists():
        BALLOON.unlink()
        return True
    return False


def disk(session: Session, fill_to: float, seconds: int) -> None:
    usage = shutil.disk_usage(system.PROJECT_ROOT)
    target_used = usage.total * fill_to / 100
    need = int(target_used - usage.used)
    if need <= 0:
        raise SystemExit(f"disk already {usage.used / usage.total:.0%} full — nothing to fill")
    if fill_to > 98:
        raise SystemExit("refusing to fill beyond 98% — the OS itself needs room to recover")
    BALLOON.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        print(f"allocating {need / 1e9:.1f} GB to reach {fill_to:.0f}% ...")
        res = subprocess.run(["fallocate", "-l", str(need), str(BALLOON)], capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"fallocate failed: {res.stderr.strip()}")
        print(f"holding for {seconds} s — keep a known person in front of the display")
        time.sleep(seconds)
        full = shutil.disk_usage(system.PROJECT_ROOT)
        st = system.unit_state(APP_UNIT)
    finally:
        disk_cleanup()
    evs = [e.as_dict() for e in system.events(t0, time.time())]
    errors = [e["msg"] for e in evs if e["kind"] == "error"]
    details = {
        "filled_to_pct": round(full.used / full.total * 100, 1),
        "seconds": seconds,
        "unit_after": st,
        "errors": errors[:30],
        "recognitions": sum(e["kind"] == "match" for e in evs),
    }
    reasons = []
    verdict = "PASS"
    if st["active"] != "active":
        verdict = "FAIL"
        reasons.append("app stopped while the disk was full")
    if errors:
        verdict = "WARN" if verdict == "PASS" else verdict
        reasons.append(f"{len(errors)} error line(s) while full (first: {errors[0][:160]})")
    if not reasons:
        reasons.append("app kept running with the disk full")
    session.record(
        "fault-disk", verdict, reasons[0], details, label=f"{fill_to:.0f}pct", reasons=reasons, started=t0
    )


# ── display ─────────────────────────────────────────────────────────────


def display(session: Session) -> None:
    before = system.unit_state(APP_UNIT)
    input("Unplug the display's HDMI cable, then press Enter ...")
    t0 = time.time()
    print("waiting 15 s unplugged ...")
    time.sleep(15)
    input("Plug the HDMI cable back in, then press Enter ...")
    t1 = time.time()
    print("waiting 20 s for the slideshow to come back ...")
    time.sleep(20)
    answer = input("Is the slideshow back on screen, filling the panel correctly? [y/n] ").strip().lower()
    after = system.unit_state(APP_UNIT)
    evs = [e.as_dict() for e in system.events(t0, time.time())]
    errors = [e["msg"] for e in evs if e["kind"] == "error"]
    details = {
        "before": before,
        "after": after,
        "replugged_at": t1,
        "operator_says_back": answer == "y",
        "errors": errors[:30],
    }
    reasons = []
    verdict = "PASS"
    if answer != "y":
        verdict = "FAIL"
        reasons.append("operator: slideshow did not come back correctly")
    if after["restarts"] != before["restarts"]:
        reasons.append(f"app restarted {after['restarts'] - before['restarts']} time(s) during the unplug")
        verdict = "WARN" if verdict == "PASS" else verdict
    if after["active"] != "active":
        verdict = "FAIL"
        reasons.append("app not running after replug")
    if errors:
        reasons.append(f"{len(errors)} error line(s)")
    if not reasons:
        reasons.append("display came back by itself, app untouched")
    session.record("fault-display", verdict, reasons[0], details, reasons=reasons, started=t0)


# ── power ───────────────────────────────────────────────────────────────


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def power_arm(session: Session) -> None:
    state = {
        "armed_at": time.time(),
        "boot_id": _boot_id(),
        "unit": system.active_unit(),
        "enabled": {
            u: system.run(["systemctl", "is-enabled", u]).stdout.strip() for u in system.UNITS.values()
        },
    }
    session.file("power_arm.json").write_text(json.dumps(state, indent=2))
    subprocess.run(["sync"])
    print(
        "Armed. Now pull the power plug (no shutdown), wait 10 s, plug it back in.\n"
        "After boot: python sitetest.py fault power --verify"
    )


def power_verify(session: Session, entry: Path, interval: float) -> None:
    try:
        armed = json.loads(session.file("power_arm.json").read_text())
    except (OSError, ValueError):
        raise SystemExit("not armed — run `fault power --arm` before pulling the plug") from None
    if _boot_id() == armed["boot_id"]:
        raise SystemExit("still the same boot — pull the power first, then verify after it restarts")
    boot_t = psutil.boot_time()
    expected = next((u for u, s in armed["enabled"].items() if s == "enabled"), None)
    active = system.active_unit()
    since = system.unit_prop(active, "ActiveEnterTimestampMonotonic") if active else ""
    unit_up_s = round(int(since) / 1e6, 1) if since.isdigit() else None
    first_cap = system.captures(boot_t, time.time())
    fs_errors = (
        system.run(["journalctl", "-k", "-b", "--no-pager", "-g", "EXT4-fs error|I/O error|corrupt"])
        .stdout.strip()
        .splitlines()
    )
    if not telemetry.running_pid(session):
        telemetry.start(session, interval, entry)
    details = {
        "armed": armed,
        "boot_time": boot_t,
        "expected_unit": expected,
        "active_unit": active,
        "unit_active_s_after_boot": unit_up_s,
        "first_capture_s_after_boot": round(first_cap[0]["t"] - boot_t, 1) if first_cap else None,
        "fs_errors": fs_errors[:20],
        "telemetry_restarted": True,
    }
    reasons, verdict = [], "PASS"
    if active != expected:
        verdict = "FAIL"
        reasons.append(f"came back as {active}, expected {expected} (the unit enabled for boot)")
    if fs_errors and fs_errors != ["-- No entries --"]:
        verdict = "WARN" if verdict == "PASS" else verdict
        reasons.append(f"kernel reported filesystem/IO errors: {fs_errors[0][:160]}")
    if unit_up_s is not None:
        reasons.append(f"app active {unit_up_s} s after boot")
    if not reasons:
        reasons.append("came back in the expected mode")
    session.record("fault-power", verdict, reasons[0], details, reasons=reasons, started=armed["armed_at"])
    session.file("power_arm.json").unlink(missing_ok=True)


# ── stress ──────────────────────────────────────────────────────────────


def _burn(stop_at: float) -> None:
    x = 0
    while time.time() < stop_at:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def stress(session: Session, minutes: float, workers: int, baseline_s: float = 120) -> None:
    if not telemetry.running_pid(session):
        raise SystemExit("telemetry is not running — `python sitetest.py telemetry start` first")
    t0 = time.time()
    stop_at = t0 + minutes * 60
    procs = [multiprocessing.Process(target=_burn, args=(stop_at,), daemon=True) for _ in range(workers)]
    for p in procs:
        p.start()
    print(f"{workers} worker(s) at 100% for {minutes:g} min — keep a known person passing by now and then")
    try:
        while time.time() < stop_at:
            rows = telemetry.load(session, time.time() - 20)
            if rows:
                r = rows[-1]
                print(
                    f"  {r['local']}  {r['temp_c']} °C  {r['arm_mhz']} MHz  throttled={r['throttled_hex']}",
                    flush=True,
                )
            time.sleep(15)
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        for p in procs:
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
    t1 = time.time()
    time.sleep(3)
    load = telemetry.summarise(telemetry.load(session, t0, t1))
    base_caps = system.captures(t0 - baseline_s, t0)
    load_caps = system.captures(t0, t1)
    base_obs = scenarios.observe(base_caps, [], [], t0 - baseline_s, t0)
    load_obs = scenarios.observe(load_caps, [], [], t0, t1)
    verdict, reasons = telemetry.verdict(load)
    for key in ("detect_ms_median", "embed_ms_median"):
        b, l_ = base_obs[key], load_obs[key]
        if b and l_:
            reasons.append(f"{key.replace('_median', '')}: {b} → {l_} ms under load")
    details = {
        "minutes": round((t1 - t0) / 60, 1),
        "workers": workers,
        "telemetry": load,
        "baseline": {k: base_obs[k] for k in ("detect_ms_median", "embed_ms_median", "frames_saved")},
        "under_load": {k: load_obs[k] for k in ("detect_ms_median", "embed_ms_median", "frames_saved")},
    }
    session.record(
        "stress",
        verdict,
        reasons[0],
        details,
        label=f"{workers}x{minutes:g}min",
        reasons=reasons,
        started=t0,
        ended=t1,
    )
