"""Site test kit: evaluation, log parsing, telemetry verdicts, results, report.

All off-Pi: synthetic capture records and log events in, verdicts out.
"""

import json
import time

import pytest

from wpu_client.sitetest import report, scenarios, session, system, telemetry

T0 = 1_700_000_000.0


def face(status, name=None, rid=None, dist=0.3, size=150, nearest=None):
    f = {
        "i": 0,
        "bbox": [0, 0, size, size],
        "size": size,
        "status": status,
        "embed_ms": 70.0,
        "sharpness": 120.0,
        "brightness": 110.0,
    }
    if status == "recognized":
        f.update(name=name, registration_id=rid or name.lower(), distance=dist)
        f["nearest"] = {"slug": rid or name.lower(), "name": name, "distance": dist}
    if status == "unrecognized":
        f.update(name="Not recognized", registration_id="__unrecognized__x")
        f["nearest"] = nearest or {"slug": "varun", "name": "Varun", "distance": 0.7}
    return f


def rec(t, *faces):
    return {
        "t": T0 + t,
        "faces": list(faces),
        "dir": f"d/{t}",
        "frame": "frame.jpg",
        "detect_ms": 40.0,
        "_root": "/tmp/x",
    }


def ev(t, kind, **fields):
    return {"t": T0 + t, "kind": kind, **fields}


def run(name, records, events, expect, secs=30, **kw):
    return scenarios.evaluate(name, records, events, expect, T0, T0 + secs, threshold=0.5, **kw)


# ── scenarios ───────────────────────────────────────────────────────────


def test_single_pass_with_latency_and_scene_kind(tmp_path):
    display = tmp_path / "display"
    display.mkdir()
    (display / "sau_scene_1.jpg").write_bytes(b"x")
    records = [rec(i, face("recognized", "Varun")) for i in range(1, 20)]
    events = [
        ev(1, "match", name="Varun"),
        ev(2, "detected", name="Varun", rid="varun", sketches=str(display)),
        ev(2, "slides", n="6", dir=str(display)),
        ev(3, "display", n="6", mode="visitor"),
    ]
    verdict, summary, reasons, o = run("single", records, events, ["varun"], scene_source="compose")
    assert verdict == "PASS", reasons
    p = o["per_person"]["varun"]
    assert p["rate"] == 1.0 and p["latency_to_display_s"] == 2.0 and p["slide_kinds"] == {"composed": 1}


def test_single_flags_fallback_to_other_scene_source(tmp_path):
    display = tmp_path / "display"
    display.mkdir()
    (display / "fru_scene_1.jpg").write_bytes(b"x")
    records = [rec(i, face("recognized", "Varun")) for i in range(1, 10)]
    events = [ev(2, "detected", name="Varun", rid="varun"), ev(2, "slides", n="1", dir=str(display))]
    verdict, _, reasons, _ = run("single", records, events, ["varun"], scene_source="precomposed")
    assert verdict == "WARN" and any("fallback" in r for r in reasons)


def test_single_fails_when_never_recognised_or_no_slides():
    records = [rec(i, face("unrecognized")) for i in range(1, 10)]
    assert run("single", records, [], ["varun"])[0] == "FAIL"
    records = [rec(i, face("recognized", "Varun")) for i in range(1, 10)]
    verdict, _, reasons, _ = run("single", records, [ev(2, "detected", name="Varun", rid="varun")], ["varun"])
    assert verdict == "FAIL" and any("no slides" in r for r in reasons)


def test_low_recognition_rate_warns_then_fails():
    mixed = [rec(i, face("recognized", "Varun") if i % 3 else face("unrecognized")) for i in range(1, 31)]
    events = [ev(1, "detected", name="Varun", rid="varun"), ev(1, "slides", n="3", dir="/nope")]
    assert run("single", mixed, events, ["varun"])[0] == "WARN"  # ~67%
    poor = [rec(i, face("recognized", "Varun") if i % 4 == 0 else face("unrecognized")) for i in range(1, 31)]
    assert run("single", poor, events, ["varun"])[0] == "FAIL"  # ~23%


def test_false_match_fails_unknown_and_single():
    records = [rec(1, face("recognized", "Asha"))]
    assert run("unknown", records, [], [])[0] == "FAIL"
    records = [rec(i, face("recognized", "Varun")) for i in range(5)] + [rec(9, face("recognized", "Asha"))]
    events = [ev(1, "detected", name="Varun", rid="varun"), ev(1, "slides", n="3", dir="/nope")]
    verdict, _, reasons, o = run("single", records, events, ["varun"])
    assert verdict == "FAIL" and o["unexpected_recognised"] == {"Asha": 1}


def test_unknown_margin_warning():
    close = {"slug": "varun", "name": "Varun", "distance": 0.52}
    records = [rec(i, face("unrecognized", nearest=close)) for i in range(5)]
    verdict, _, reasons, o = run("unknown", records, [], [])
    assert verdict == "WARN" and o["impostor_distances"][0] == 0.52 and o["impostor_margin"] == 0.02


def test_empty_scenario():
    assert run("empty", [], [], [], secs=60)[0] == "PASS"
    assert run("empty", [rec(3, face("unrecognized"))], [], [], secs=60)[0] == "WARN"


def test_duo_pass_and_fallback():
    events = [ev(3, "detected", name="Varun & Asha", rid="duo::asha::varun")]
    assert run("duo", [], events, ["varun", "asha"])[0] == "PASS"
    events = [ev(3, "duo_fallback")]
    records = [rec(1, face("recognized", "Varun"), face("recognized", "Asha"))]
    assert run("duo", records, events, ["varun", "asha"])[0] == "WARN"


def test_return_needs_two_detections():
    once = [ev(2, "detected", name="Varun", rid="varun")]
    assert run("return", [rec(1, face("recognized", "Varun"))], once, ["varun"], secs=60)[0] == "FAIL"
    twice = once + [ev(15, "left", name="Varun", rid="varun"), ev(25, "detected", name="Varun", rid="varun")]
    assert run("return", [rec(1, face("recognized", "Varun"))], twice, ["varun"], secs=60)[0] == "PASS"


def test_spot_too_small_and_crowd_flicker():
    small = [rec(1, face("too_small", size=60))]
    verdict, _, reasons, _ = run("spot", small, [], ["varun"], secs=15, label="4m")
    assert verdict == "FAIL" and "too small" in reasons[0]
    flicker = [ev(i, "detected", name=f"P{i}", rid=f"p{i}") for i in range(10)]
    assert run("crowd", [rec(1, face("unrecognized"))], flicker, [], secs=60)[0] == "WARN"


def test_spoof_is_info():
    records = [rec(1, face("recognized", "Varun"))]
    verdict, summary, _, _ = run("spoof", records, [], ["varun"])
    assert verdict == "INFO" and "ACCEPTED" in summary


def test_errors_in_window_warn():
    records = [rec(i, face("unrecognized")) for i in range(3)]
    events = [ev(1, "error", msg="Asset preparation failed")]
    assert run("unknown", records, events, [])[0] == "WARN"


def test_expect_validation():
    assert scenarios.check_expect("single", []) is not None
    assert scenarios.check_expect("duo", ["a"]) is not None
    assert scenarios.check_expect("unknown", ["a"]) is not None
    assert scenarios.check_expect("queue", ["a", "b", "c"]) is None


# ── logs ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "msg,kind,field,value",
    [
        ("Local match: Varun dist=0.2311 conf=76.9% cutouts=/x", "match", "dist", "0.2311"),
        (
            "No local match within threshold (0.5 cosine) — not recognized (nearest Asha dist=0.6123)",
            "nomatch",
            "dist",
            "0.6123",
        ),
        (
            "Emitted person.detected: Varun (registration_id: varun, sketches: /a/b)",
            "detected",
            "rid",
            "varun",
        ),
        (
            "Emitted person.left: Varun & Asha (registration_id: duo::asha::varun)",
            "left",
            "rid",
            "duo::asha::varun",
        ),
        ("Loaded 6 local slide(s)/video(s) from /data/x/display", "slides", "n", "6"),
        ("Updated image list: 6 images, mode=visitor", "display", "mode", "visitor"),
        (
            "Gallery reloaded: 4 person(s); changed: asha; released 1 unrecognized face(s)",
            "gallery_reload",
            "n",
            "4",
        ),
    ],
)
def test_parse_line(msg, kind, field, value):
    line = f"2026-10-12 14:03:22 - pi-1 - wpu_client.services.x - INFO - {msg}"
    e = system.parse_line(line)
    assert e.kind == kind and e.fields[field] == value


def test_parse_line_levels_and_garbage():
    assert system.parse_line("2026-10-12 14:03:22 - pi-1 - x - ERROR - boom").kind == "error"
    assert system.parse_line("2026-10-12 14:03:22 - pi-1 - x - DEBUG - chatter") is None
    assert system.parse_line("Traceback (most recent call last):") is None


# ── telemetry ───────────────────────────────────────────────────────────


def test_decode_throttled():
    assert telemetry.decode_throttled(0x50005) == {
        "under_voltage": 1,
        "freq_capped": 0,
        "throttled": 1,
        "soft_temp_limit": 0,
        "uv_occurred": 1,
        "throttle_occurred": 1,
    }


def _rows(temps, **extra):
    return [
        {
            "t": str(T0 + 5 * i),
            "temp_c": str(tc),
            "arm_mhz": "1800",
            "cpu_pct": "40",
            "app_rss_mb": "300",
            "disk_free_gb": "20",
            "restarts": "0",
            "under_voltage": "0",
            "throttled": "0",
            "freq_capped": "0",
            "soft_temp_limit": "0",
            **extra,
        }
        for i, tc in enumerate(temps)
    ]


def test_telemetry_verdicts():
    assert telemetry.verdict(telemetry.summarise(_rows([55, 60, 62])))[0] == "PASS"
    assert telemetry.verdict(telemetry.summarise(_rows([70, 77])))[0] == "WARN"
    assert telemetry.verdict(telemetry.summarise(_rows([70, 82])))[0] == "FAIL"
    assert telemetry.verdict(telemetry.summarise(_rows([50], under_voltage="1")))[0] == "FAIL"
    assert telemetry.verdict(telemetry.summarise([]))[0] == "FAIL"


# ── sessions, config overlay, report ────────────────────────────────────


@pytest.fixture
def tmp_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "ROOT", tmp_path)
    monkeypatch.setattr(session, "CURRENT", tmp_path / "CURRENT")
    return session.start("unit test")


def test_record_writes_json_md_index_and_artifacts(tmp_session, tmp_path):
    frame = tmp_path / "cap" / "frame.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"jpg")
    p = tmp_session.record(
        "single",
        "PASS",
        "fine",
        {"x": 1},
        label="glasses",
        reasons=["[ok] y"],
        artifacts=[frame, tmp_path / "missing.jpg"],
    )
    assert p.name == "001_single_glasses_PASS.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["artifacts"] == ["results/001_single_glasses_PASS/00_cap_frame.jpg"]
    assert (tmp_session.path / doc["artifacts"][0]).read_bytes() == b"jpg"
    assert "**PASS** — fine" in p.with_suffix(".md").read_text(encoding="utf-8")
    assert session.current().path == tmp_session.path
    assert tmp_session.record("note", "INFO", "x").name.startswith("002_")


def test_overlay_config_never_drops_base_settings():
    base = {
        "log_level": "INFO",
        "services": {
            "slideshow": {"scale_mode": "fill"},
            "face_recognition": {"model": "mobilenet", "use_legacy_final_images": True},
        },
    }
    cfg = system.overlay_config(
        base,
        scene_source="precomposed",
        threshold=0.45,
        capture=True,
        empty_every=60,
        min_face=90,
        log_level="DEBUG",
        fallback=False,
    )
    fr = cfg["services"]["face_recognition"]
    assert cfg["services"]["slideshow"]["scale_mode"] == "fill" and fr["model"] == "mobilenet"
    assert fr["scene_source"] == "precomposed" and "use_legacy_final_images" not in fr
    assert fr["capture_store"]["enabled"] and fr["diagnostic_match_threshold"] == 0.45
    assert base["services"]["face_recognition"]["use_legacy_final_images"]  # base untouched
    eff = system.effective(cfg)
    assert eff["scene_source"] == "precomposed" and eff["threshold"] == 0.45 and eff["capture"]


def test_threshold_advice():
    adv = report.threshold_advice([0.2, 0.25, 0.3, 0.35], [0.6, 0.65, 0.7], 0.5)
    assert adv["false_reject_at_current"] == 0 and adv["false_accept_at_current"] == 0
    assert adv["separable"] and 0.35 < adv["suggested"] < 0.6


def test_report_builds_from_results(tmp_session):
    records = [rec(i, face("recognized", "Varun")) for i in range(1, 10)]
    events = [ev(1, "detected", name="Varun", rid="varun"), ev(1, "slides", n="3", dir="/nope")]
    v, s, r, o = run("single", records, events, ["varun"])
    tmp_session.record("single", v, s, o, reasons=r, started=T0, ended=T0 + 30)
    v, s, r, o = run("unknown", [rec(1, face("unrecognized"))], [], [])
    tmp_session.record("unknown", v, s, o, reasons=r)
    tmp_session.telemetry_csv.write_text(
        "t,temp_c,arm_mhz,app_rss_mb,cpu_pct\n"
        + "".join(f"{time.time() + i},{50 + i},1800,300,20\n" for i in range(5))
    )
    path = report.build(tmp_session)
    page = path.read_text(encoding="utf-8")
    assert "Site test" in page and "<svg" in page and "single" in page
    summary = json.loads(tmp_session.file("summary.json").read_text())
    assert summary["threshold"]["genuine_n"] == 9 and summary["threshold"]["impostor_n"] == 1
