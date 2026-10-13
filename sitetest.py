#!/usr/bin/env python3
"""WPU site test kit — the single entry point. See SITETEST.md.

    python sitetest.py session start --name <site>
    python sitetest.py app start --scene-source compose|precomposed
    python sitetest.py telemetry start
    python sitetest.py scenario single --expect varun
    python sitetest.py report

Every test writes a result (.json + .md) into data/sitetest/<session>/results/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ENTRY = Path(__file__).resolve()
sys.path.insert(0, str(ENTRY.parent))

from wpu_client.sitetest import session as sessions  # noqa: E402
from wpu_client.sitetest import system  # noqa: E402


def _session(args):
    return sessions.by_name(args.session) if getattr(args, "session", None) else sessions.current()


def _say(verdict: str, summary: str, path=None, reasons=()):
    print(f"\n  {verdict}  {summary}")
    for r in reasons:
        print(f"    {r}")
    if path:
        print(f"  → {path}")


# ── session ─────────────────────────────────────────────────────────────


def cmd_session(args):
    if args.action == "start":
        s = sessions.start(args.name, args.note or "")
        print(f"session {s.name} started → {s.path}")
        print("next: python sitetest.py app start   and   python sitetest.py telemetry start")
    elif args.action == "status":
        cmd_status(args)
    elif args.action == "end":
        s = sessions.current()
        from wpu_client.sitetest import telemetry

        telemetry.stop(s)
        sessions.CURRENT.unlink(missing_ok=True)
        print(f"session {s.name} closed (telemetry stopped). Run `report --session {s.name}` any time.")


def cmd_status(args):
    s = sessions.current(required=False)
    print(f"session   : {s.name if s else 'none — python sitetest.py session start --name <site>'}")
    if s:
        print(f"app       : {s.app_settings() or 'not started via sitetest'}")
        from wpu_client.sitetest import telemetry

        pid = telemetry.running_pid(s)
        rows = telemetry.load(s, time.time() - 30)
        last = rows[-1] if rows else {}
        print(
            f"telemetry : {'running pid ' + str(pid) if pid else 'NOT running'}"
            + (
                f" — {last.get('temp_c')} °C, {last.get('arm_mhz')} MHz, "
                f"throttled {last.get('throttled_hex')}"
                if last
                else ""
            )
        )
        print(f"results   : {len(s.all_results())} in {s.results}")
    for unit in system.UNITS.values():
        st = system.unit_state(unit)
        print(
            f"{unit:30s} {st['active']:9s} boot={st['enabled']:9s} restarts={st['restarts']}"
            + ("  [sitetest drop-in]" if st["dropin"] else "")
        )
    gdir = system.gallery_dir()
    people = (
        [p.name for p in gdir.iterdir() if p.is_dir() and not p.name.startswith(".")] if gdir.is_dir() else []
    )
    print(f"gallery   : {len(people)} seeded ({', '.join(sorted(people))})")
    croot = system.captures_dir()
    today = croot / time.strftime("%Y%m%d")
    n = sum(1 for _ in open(today / "index.jsonl")) if (today / "index.jsonl").exists() else 0
    import shutil

    free = shutil.disk_usage(system.PROJECT_ROOT).free / 1e9
    print(f"captures  : {n} today in {today}   disk free {free:.1f} GB")


# ── app ─────────────────────────────────────────────────────────────────


def cmd_app(args):
    s = sessions.current()
    unit = system.UNITS["diagnostic"]
    if args.action == "start":
        cfg = system.overlay_config(
            system.base_config(),
            scene_source=args.scene_source,
            threshold=args.threshold,
            capture=not args.no_capture,
            empty_every=args.empty_every,
            min_face=args.min_face,
            log_level=args.log_level,
            fallback=False if args.strict else None,
        )
        path = s.file(f"app_config_{time.strftime('%H%M%S')}.yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        system.require_unit(unit)
        system.write_dropin(unit, _app_exec(path))
        t0 = time.time()
        system.restart(unit)
        up = system.wait_active(unit, 60)
        state = dict(system.effective(cfg), config=str(path), started=t0)
        s.app_state.write_text(json.dumps(state, indent=2))
        verdict = "INFO" if up is not None else "FAIL"
        summary = (
            (
                f"diagnostic app running: scene_source={state['scene_source']}, "
                f"threshold={state['threshold']}, "
                f"captures={'on' if state['capture'] else 'off'}"
            )
            if up is not None
            else "app did not start"
        )
        p = s.record(
            "app-start",
            verdict,
            summary,
            {"unit_active_s": up, **state},
            label=state["scene_source"],
            started=t0,
        )
        _say(verdict, summary, p)
    elif args.action == "stop":
        system.stop_all()
        print("all slideshow units stopped (camera and display free)")
    elif args.action == "restore":
        removed = [u for u in system.UNITS.values() if system.remove_dropin(u)]
        print(f"removed sitetest drop-ins from: {', '.join(removed) or 'none'} — units are back as installed")
        if system.unit_active(unit):
            system.restart(unit)
            print(f"{unit} restarted with its own config")
    elif args.action == "status":
        print(json.dumps({"state": s.app_settings(), "unit": system.unit_state(unit)}, indent=2))


# ── check ───────────────────────────────────────────────────────────────


def cmd_check(args):
    s = sessions.current()
    res = system.run(
        [system.python_exe(), str(system.PROJECT_ROOT / "main.py"), "--check", "--diagnostic", "--json"],
        timeout=180,
    )
    try:
        doc = json.loads(res.stdout)
    except ValueError:
        doc = {"raw": res.stdout[-4000:], "stderr": res.stderr[-4000:]}
    checks = doc.get("checks") or doc.get("results") or []
    failed = [c for c in checks if str(c.get("status", "")).upper() == "FAIL"]
    warned = [c for c in checks if str(c.get("status", "")).upper() == "WARN"]
    verdict = "FAIL" if res.returncode else ("WARN" if warned else "PASS")
    reasons = [f"{c.get('name')}: {c.get('status')} {c.get('detail', '')}" for c in failed + warned]
    summary = "pre-flight " + (
        "failed: " + ", ".join(c.get("name", "?") for c in failed)
        if failed
        else "ok" + (f" ({len(warned)} warning(s))" if warned else "")
    )
    p = s.record("check", verdict, summary, doc, reasons=reasons)
    _say(verdict, summary, p, reasons)


# ── telemetry ───────────────────────────────────────────────────────────


def cmd_telemetry(args):
    from wpu_client.sitetest import telemetry

    s = _session(args)
    if args.action == "run":
        telemetry.run_sampler(s, args.interval)
    elif args.action == "start":
        print(f"telemetry running (pid {telemetry.start(s, args.interval, ENTRY)}) → {s.telemetry_csv}")
    elif args.action == "stop":
        print("stopped" if telemetry.stop(s) else "was not running")
    elif args.action in ("status", "check"):
        since = time.time() - args.minutes * 60 if args.minutes else 0
        summ = telemetry.summarise(telemetry.load(s, since))
        verdict, reasons = telemetry.verdict(summ)
        if args.action == "status":
            print(json.dumps(summ, indent=2))
            _say(verdict, reasons[0], None, reasons[1:])
        else:
            p = s.record(
                "telemetry",
                verdict,
                reasons[0],
                summ,
                label=f"{args.minutes}min" if args.minutes else "session",
                reasons=reasons,
                started=since or None,
            )
            _say(verdict, reasons[0], p, reasons[1:])


# ── gallery / seeding ───────────────────────────────────────────────────


def cmd_seed(args):
    from wpu_client.sitetest import seed

    s = sessions.current()
    source = "capture" if args.from_capture else "camera" if args.camera else "photo"
    t0 = time.time()
    threshold = (s.app_settings() or {}).get("threshold") or 0.5
    details = seed.seed(
        args.name,
        args.gender,
        args.height,
        source,
        photo=args.photo,
        within_s=args.within,
        sau=args.sau,
        fru=args.fru,
        proxy_from=args.proxy_from,
        scenes=args.scenes,
        force=args.force,
        min_face=args.min_face,
        threshold=threshold,
    )
    reasons = [
        f"face {details['face_px'][2]}x{details['face_px'][3]} px, quality {details['face_quality']}",
        f"cutouts {details['cutouts']}"
        + (f" (proxy of {details['proxy_of']})" if details["proxy_of"] else ""),
        f"closest other person: {details['closest_other']}",
    ]
    verdict = "PASS"
    if details["confusable"]:
        verdict = "WARN"
        reasons.insert(
            0,
            f"looks like {details['closest_other']['name']} to the model "
            f"(distance {details['closest_other']['distance']} < threshold {threshold}) — expect mix-ups",
        )
    if not all(details["cutouts"].values()) and not details["scenes"]:
        verdict = "WARN"
        reasons.append(
            "no SAU+FRU cutouts and no scenes — recognised but nothing to show; "
            "add --sau/--fru, --proxy-from auto or --scenes"
        )
    p = s.record(
        "seed",
        verdict,
        f"seeded {args.name} ({source})",
        details,
        label=details["slug"],
        reasons=reasons,
        started=t0,
    )
    _say(verdict, f"seeded {args.name} as '{details['slug']}'", p, reasons)
    print("  the running app picks this up within ~5 s; the visitor may need to step out of view and back")


def cmd_gallery(args):
    from wpu_client.services.face_recognition.diagnostic_gallery import DiagnosticGallery
    from wpu_client.services.face_recognition.local_assets import resolve_local_cutouts

    g = DiagnosticGallery(str(system.gallery_dir()))
    g.load()
    print(
        f"{'slug':16s} {'name':16s} {'gender':7s} {'cm':>4s} {'models':18s} "
        f"{'sau':4s} {'fru':4s} {'scenes':>6s} proxy"
    )
    for e in g.entries():
        cut = resolve_local_cutouts(e.sketch_dir, e.meta)
        scenes = Path(e.sketch_dir).parent / "scenes"
        n_scenes = len(list(scenes.iterdir())) if scenes.is_dir() else 0
        print(
            f"{e.slug:16s} {e.name[:16]:16s} {str(e.gender or '-'):7s} "
            f"{str(e.meta.get('height_cm', '-')):>4s} "
            f"{'/'.join(sorted(e.vectors)):18s} {'yes' if cut['sau'] else 'NO':4s} "
            f"{'yes' if cut['fru'] else 'NO':4s} "
            f"{n_scenes:6d} {e.meta.get('proxy_of', '')}"
        )


# ── scenarios ───────────────────────────────────────────────────────────


def _sample_frames(records: list[dict], obs: dict, limit: int = 6) -> list[Path]:
    with_faces = [r for r in records if r.get("faces") and r.get("frame")]
    picks = []
    odd = [
        r
        for r in with_faces
        if any(
            f.get("status") in ("unrecognized", "too_small")
            or f.get("name") in obs.get("unexpected_recognised", {})
            for f in r["faces"]
        )
    ]
    for r in [*(with_faces[:1]), *odd[:3], *(with_faces[len(with_faces) // 2 :][:1]), *(with_faces[-1:])]:
        if r not in picks:
            picks.append(r)
    return [Path(r["_root"]) / r["dir"] / r["frame"] for r in picks[:limit]]


def cmd_scenario(args):
    from wpu_client.sitetest import scenarios

    if args.name == "list":
        for sc in scenarios.CATALOGUE.values():
            print(f"{sc.name:8s} {sc.seconds:3d}s  expect={sc.expect:8s} {sc.instructions}")
        return
    s = sessions.current()
    sc = scenarios.CATALOGUE[args.name]
    expect = [e.strip() for e in (args.expect or "").split(",") if e.strip()]
    err = scenarios.check_expect(args.name, expect)
    if err:
        raise SystemExit(err)
    if sc.needs_label and not args.label:
        raise SystemExit(f"'{args.name}' needs --label (e.g. --label 2m-centre)")
    state = s.app_settings()
    if not state:
        raise SystemExit("start the app through the kit first: python sitetest.py app start")
    if not state.get("capture"):
        raise SystemExit("captures are off — scenarios are judged from them: app start without --no-capture")
    unit = system.UNITS["diagnostic"]
    if not system.unit_active(unit):
        raise SystemExit(f"{unit} is not running — python sitetest.py app start")
    seconds = args.seconds or sc.seconds
    print(f"\n[{sc.name}{' / ' + args.label if args.label else ''}]  {seconds}s  expect: {expect or '—'}")
    print(f"  {sc.instructions}\n")
    if not args.now:
        input("Press Enter when people are in position (Ctrl+C during the run ends it early) ...")
    start = time.time()
    try:
        while time.time() - start < seconds:
            left = seconds - (time.time() - start)
            print(f"  {left:4.0f}s left", end="\r", flush=True)
            time.sleep(min(1.0, left))
    except KeyboardInterrupt:
        print("\n  ended early")
    end = time.time()
    print("\n  evaluating ...")
    time.sleep(3)  # let the capture writer flush
    root = system.captures_dir(
        yaml.safe_load(Path(state["config"]).read_text()) if state.get("config") else None
    )
    records = system.captures(start, end, root)
    evs = [e.as_dict() for e in system.events(start, end + 2)]
    verdict, summary, reasons, obs = scenarios.evaluate(
        args.name,
        records,
        evs,
        expect,
        start,
        end,
        label=args.label or "",
        scene_source=state.get("scene_source"),
        threshold=state.get("threshold"),
        interval=float(state.get("detection_interval") or 1),
    )
    obs["events"] = evs[:400]
    obs["expect"] = expect
    p = s.record(
        args.name,
        verdict,
        summary,
        obs,
        label=args.label or ",".join(expect),
        reasons=reasons,
        started=start,
        ended=end,
        artifacts=_sample_frames(records, obs),
    )
    _say(verdict, summary, p, reasons)


# ── faults / stress ─────────────────────────────────────────────────────


def cmd_fault(args):
    from wpu_client.sitetest import faults

    s = sessions.current()
    if args.kind == "kill":
        faults.kill(s)
    elif args.kind == "disk":
        if args.cleanup:
            print("balloon removed" if faults.disk_cleanup() else "no balloon present")
            return
        faults.disk(s, args.fill_to, args.seconds)
    elif args.kind == "display":
        faults.display(s)
    elif args.kind == "power":
        if args.verify:
            faults.power_verify(s, ENTRY, args.interval)
        else:
            faults.power_arm(s)
        return
    last = s.all_results()[-1]
    _say(last["verdict"], last["summary"], None, last["reasons"])


def cmd_stress(args):
    from wpu_client.sitetest import faults

    s = sessions.current()
    faults.stress(s, args.minutes, args.workers)
    last = s.all_results()[-1]
    _say(last["verdict"], last["summary"], None, last["reasons"])


# ── display patterns, notes, report, collect ────────────────────────────


def _app_exec(config_path) -> str:
    return f'"{system.python_exe()}" main.py --service all --diagnostic --config "{config_path}"'


def cmd_patterns(args):
    """Patterns run on the diagnostic unit (slideshow only, pattern folder as
    its images) — the one unit every kit Pi has. `stop` puts the app back with
    the settings of the last `app start`."""
    from wpu_client.sitetest import patterns

    s = sessions.current()
    unit = system.require_unit(system.UNITS["diagnostic"])
    if args.action == "start":
        out = s.file("patterns")
        patterns.write_all(out)
        cfg = system.base_config()
        sl = cfg.setdefault("services", {}).setdefault("slideshow", {})
        sl["image_directory"] = str(out)
        sl["advance_time"] = args.seconds
        path = s.file("patterns_config.yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        system.write_dropin(unit, f'"{system.python_exe()}" main.py --service slideshow --config "{path}"')
        system.restart(unit)
        print(
            f"patterns cycling every {args.seconds}s on the panel (camera off). Judge each on screen, "
            "then record it, e.g.\n"
            '  python sitetest.py note --test display --label geometry --verdict pass "circle round"\n'
            "then: python sitetest.py patterns stop"
        )
        return
    state = s.app_settings()
    if state.get("config") and Path(state["config"]).exists():
        system.write_dropin(unit, _app_exec(state["config"]))
        system.restart(unit)
        print(f"patterns stopped — app back on its last settings (scene_source={state.get('scene_source')})")
    else:
        system.remove_dropin(unit)
        system.sudo(["systemctl", "stop", unit], check=False)
        print("patterns stopped — start the app: python sitetest.py app start ...")


def cmd_note(args):
    s = sessions.current()
    p = s.record(args.test, args.verdict.upper(), args.text, {"note": args.text}, label=args.label or "")
    _say(args.verdict.upper(), args.text, p)


def cmd_report(args):
    from wpu_client.sitetest import report

    s = _session(args)
    print(f"report → {report.build(s)}")


def cmd_collect(args):
    from wpu_client.sitetest import collect

    s = _session(args)
    out = Path(args.out) if args.out else system.DATA_DIR / "sitetest" / "bundles"
    print("bundling (captures make this slow) ...")
    path = collect.bundle(s, out, with_captures=not args.no_captures, with_dataset=args.with_dataset)
    print(f"bundle → {path} ({path.stat().st_size / 1e6:.0f} MB)")


# ── argument parsing ────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sitetest.py", description="WPU site test kit (see SITETEST.md)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("session", help="start / status / end a test session")
    a.add_argument("action", choices=["start", "status", "end"])
    a.add_argument("--name", default="site")
    a.add_argument("--note")
    a.set_defaults(fn=cmd_session)

    sub.add_parser("status", help="everything at a glance").set_defaults(fn=cmd_status)
    sub.add_parser("check", help="pre-flight (main.py --check --diagnostic) as a result").set_defaults(
        fn=cmd_check
    )

    a = sub.add_parser("app", help="run the diagnostic app with test settings")
    a.add_argument("action", choices=["start", "stop", "restore", "status"])
    a.add_argument("--scene-source", choices=["compose", "precomposed"], default="compose")
    a.add_argument("--strict", action="store_true", help="no fallback to the other scene source")
    a.add_argument("--threshold", type=float, help="diagnostic match threshold (cosine distance)")
    a.add_argument("--min-face", type=int, help="min face size in px")
    a.add_argument("--no-capture", action="store_true", help="don't store frames (scenarios need them)")
    a.add_argument("--empty-every", type=float, default=60, help="also store an empty frame every N s")
    a.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"])
    a.set_defaults(fn=cmd_app)

    a = sub.add_parser("telemetry", help="temperature/throttle/memory sampler")
    a.add_argument("action", choices=["start", "stop", "status", "check", "run"])
    a.add_argument("--interval", type=float, default=5)
    a.add_argument("--minutes", type=float, default=0, help="status/check: last N minutes (0 = session)")
    a.add_argument("--session")
    a.set_defaults(fn=cmd_telemetry)

    a = sub.add_parser("seed", help="seed a visitor into the local gallery, live")
    a.add_argument("--name", required=True)
    a.add_argument("--gender", choices=["male", "female"])
    a.add_argument("--height", type=int, help="height in cm (body scaling)")
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-capture", action="store_true", help="newest unrecognised face the app captured")
    src.add_argument("--photo", help="image file with the face")
    src.add_argument("--camera", action="store_true", help="briefly take the camera from the app")
    a.add_argument("--within", type=float, default=120, help="--from-capture: look back N seconds")
    a.add_argument("--sau", help="body cutout PNG")
    a.add_argument("--fru", help="face cutout PNG")
    a.add_argument("--proxy-from", help="borrow cutouts (and scenes) from a seeded slug, or 'auto'")
    a.add_argument("--scenes", help="folder of pre-composed scenes")
    a.add_argument("--min-face", type=int, default=80)
    a.add_argument("--force", action="store_true", help="re-seed an existing person")
    a.set_defaults(fn=cmd_seed)

    sub.add_parser("gallery", help="list seeded people and their art").set_defaults(fn=cmd_gallery)

    a = sub.add_parser("scenario", help="run one timed scenario ('scenario list' for all)")
    a.add_argument(
        "name",
        choices=[
            "list",
            "empty",
            "single",
            "unknown",
            "duo",
            "mixed",
            "crowd",
            "return",
            "walkby",
            "queue",
            "spot",
            "spoof",
        ],
    )
    a.add_argument("--expect", help="seeded slug(s)/name(s), comma-separated")
    a.add_argument("--label", help="variant or spot name, e.g. glasses, 2m-centre")
    a.add_argument("--seconds", type=int)
    a.add_argument("--now", action="store_true", help="start without waiting for Enter")
    a.set_defaults(fn=cmd_scenario)

    a = sub.add_parser("fault", help="fault tests: kill | disk | display | power")
    a.add_argument("kind", choices=["kill", "disk", "display", "power"])
    a.add_argument("--fill-to", type=float, default=95, help="disk: target used %%")
    a.add_argument("--seconds", type=int, default=120, help="disk: how long to hold it full")
    a.add_argument("--cleanup", action="store_true", help="disk: remove a leftover balloon file")
    a.add_argument("--arm", action="store_true", help="power: before pulling the plug")
    a.add_argument("--verify", action="store_true", help="power: after the Pi is back")
    a.add_argument("--interval", type=float, default=5)
    a.set_defaults(fn=cmd_fault)

    a = sub.add_parser("stress", help="all cores at 100%% while the app runs")
    a.add_argument("--minutes", type=float, default=20)
    a.add_argument("--workers", type=int, default=4)
    a.set_defaults(fn=cmd_stress)

    a = sub.add_parser("patterns", help="display test patterns on the panel")
    a.add_argument("action", choices=["start", "stop"])
    a.add_argument("--seconds", type=int, default=8, help="seconds per pattern")
    a.set_defaults(fn=cmd_patterns)

    a = sub.add_parser("note", help="record a manual observation as a result")
    a.add_argument("text")
    a.add_argument("--test", default="note", help="e.g. display, mount, lighting")
    a.add_argument("--label")
    a.add_argument("--verdict", choices=["pass", "fail", "warn", "info"], default="info")
    a.set_defaults(fn=cmd_note)

    a = sub.add_parser("report", help="build report.html for a session")
    a.add_argument("--session", help="session folder name or path (default: current)")
    a.set_defaults(fn=cmd_report)

    a = sub.add_parser("collect", help="bundle a session for the laptop")
    a.add_argument("--session")
    a.add_argument("--out", help="output folder, e.g. a USB stick")
    a.add_argument("--no-captures", action="store_true", help="indexes only, no images")
    a.add_argument("--with-dataset", action="store_true", help="also data/dataset")
    a.set_defaults(fn=cmd_collect)
    return p


def main():
    # The report also runs on a Windows laptop, whose console can't print every symbol used here.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
