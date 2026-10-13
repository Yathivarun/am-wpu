"""Scenario catalogue and evaluation.

A scenario is a timed observation window: people do what the instructions
say, the running app does its normal job, and afterwards the window is judged
from two sources only — the capture store's per-frame records (what the camera
saw and what recognition made of each face) and the app's log events (what was
displayed, when). Nothing here touches the camera, so every scenario tests the
real production path, not a re-implementation of it.

`evaluate()` is pure (records + events in, verdict out) so it is unit-tested
off the Pi.
"""

import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from wpu_client.services.face_recognition.asset_fetcher import PRECOMPOSED_PREFIX
from wpu_client.services.face_recognition.base_composer import BODY_PREFIX, FACE_PREFIX

MATCH_RATE_PASS = 0.8  # share of frames a known face must be recognised in
LATENCY_WARN_S = 5.0  # first face seen -> their slides on screen
SPOT_RATE_PASS = 0.5  # placement spots are judged more leniently
IMPOSTOR_MARGIN_WARN = 0.05
FLICKER_PER_MIN_WARN = 6


@dataclass
class Scenario:
    name: str
    seconds: int
    expect: str  # none | one | two | list | optional
    instructions: str
    needs_label: bool = False


CATALOGUE: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario(
            "empty",
            60,
            "none",
            "Nobody in front of the camera. Leave posters/screens in view as they will be on site.",
        ),
        Scenario(
            "single",
            30,
            "one",
            "One seeded person walks up, stops facing the display ~1.5 m away, stays until the end. "
            "Use --label for variants: glasses, cap, mask, side-light, backlight, child, tall.",
        ),
        Scenario(
            "unknown", 30, "none", "One or more UNSEEDED people stand in front of the display, facing it."
        ),
        Scenario(
            "duo", 40, "two", "Two seeded people stand side by side facing the display; nobody else in view."
        ),
        Scenario(
            "mixed", 40, "one", "One seeded person and one unseeded person stand together facing the display."
        ),
        Scenario(
            "crowd",
            60,
            "optional",
            "Three to five people (any mix) in view, moving and swapping places naturally.",
        ),
        Scenario(
            "return",
            60,
            "one",
            "Seeded person stands ~10 s, walks fully out of view for at least 6 s, then comes back "
            "and stays until the end.",
        ),
        Scenario(
            "walkby",
            30,
            "optional",
            "People walk past the display at normal pace without stopping or looking at it.",
        ),
        Scenario(
            "queue",
            90,
            "list",
            "The listed seeded people take turns: each steps up alone for ~10 s, then leaves; "
            "next one steps up. Order as listed.",
        ),
        Scenario(
            "spot",
            15,
            "one",
            "Seeded person stands on the marked floor spot facing the display. --label names the "
            "spot, e.g. 2m-centre, 3m-left30.",
            needs_label=True,
        ),
        Scenario(
            "spoof",
            30,
            "one",
            "Hold a phone screen or printed photo of a seeded person in front of the camera; "
            "nobody real in view.",
        ),
    ]
}


# ── observation ─────────────────────────────────────────────────────────


def _is(face: dict, who: str) -> bool:
    who = who.lower()
    return face.get("status") == "recognized" and who in (
        str(face.get("registration_id", "")).lower(),
        str(face.get("name", "")).lower(),
    )


def _event_is(ev: dict, who: str) -> bool:
    who = who.lower()
    return who in (str(ev.get("rid", "")).lower(), str(ev.get("name", "")).lower())


def _median(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def slide_kinds(sketch_dir: str) -> dict:
    """How many composed vs pre-composed slides sit in a display folder."""
    kinds = Counter()
    try:
        for p in Path(sketch_dir).iterdir():
            if p.name.startswith((BODY_PREFIX, FACE_PREFIX)):
                kinds["composed"] += 1
            elif p.name.startswith(PRECOMPOSED_PREFIX):
                kinds["precomposed"] += 1
            elif p.is_file():
                kinds["other"] += 1
    except OSError:
        pass
    return dict(kinds)


def observe(
    records: list[dict],
    events: list[dict],
    expect: list[str],
    start: float,
    end: float,
    interval: float = 1.0,
) -> dict:
    """Everything the evaluators judge on, from one window."""
    faces = [
        dict(f, t=r["t"], dir=r.get("dir"), root=r.get("_root")) for r in records for f in r.get("faces", [])
    ]
    face_frames = [r for r in records if r.get("faces")]
    processed = [f for f in faces if f.get("status") in ("recognized", "unrecognized")]
    processed_frames = {f["t"] for f in processed}
    recognised = Counter(f.get("name") for f in faces if f.get("status") == "recognized")
    unexpected = Counter(
        f.get("name") for f in faces if f.get("status") == "recognized" and not any(_is(f, e) for e in expect)
    )
    first_face_t = min((f["t"] for f in processed), default=None)

    def ev(kind):
        return [e for e in events if e.get("kind") == kind]

    per_person = {}
    for who in expect:
        mine = [f for f in faces if _is(f, who)]
        frames_as = {f["t"] for f in mine}
        detected = [
            e for e in ev("detected") if _event_is(e, who) or who.lower() in str(e.get("rid", "")).lower()
        ]
        det_t = detected[0]["t"] if detected else None
        shown = [
            e for e in ev("display") if e.get("mode") == "visitor" and det_t is not None and e["t"] >= det_t
        ]
        slides = [e for e in ev("slides") if det_t is not None and e["t"] >= det_t]
        own_first = min(
            (
                f["t"]
                for f in faces
                if f.get("status") in ("recognized", "unrecognized")
                and (
                    _is(f, who)
                    or (f.get("nearest") or {}).get("slug", "").lower() == who.lower()
                    or (f.get("nearest") or {}).get("name", "").lower() == who.lower()
                )
            ),
            default=first_face_t,
        )
        per_person[who] = {
            "frames_recognised": len(frames_as),
            "rate": round(len(frames_as) / len(processed_frames), 3) if processed_frames else 0.0,
            "first_recognised_t": min(frames_as, default=None),
            "detected_events": len(detected),
            "detected_t": det_t,
            "detected_rids": sorted({e.get("rid") for e in detected}),
            "latency_to_detected_s": round(det_t - own_first, 1) if det_t and own_first else None,
            "latency_to_display_s": round(shown[0]["t"] - own_first, 1) if shown and own_first else None,
            "slides_loaded": int(slides[0]["n"]) if slides else None,
            "slides_dir": slides[0].get("dir") if slides else None,
            "slide_kinds": slide_kinds(slides[0]["dir"]) if slides and slides[0].get("dir") else {},
            "genuine_distance_median": _median([(f.get("nearest") or {}).get("distance") for f in mine]),
            "face_size_median": _median([f.get("size") for f in mine]),
        }

    switches = ev("detected")
    minutes = max((end - start) / 60, 1 / 60)
    return {
        "window_s": round(end - start, 1),
        "frames_saved": len(records),
        "frames_with_faces": len(face_frames),
        "detect_rate": round(len(face_frames) / max(1.0, (end - start) / interval), 3),
        "faces_total": len(faces),
        "faces_processed": len(processed),
        "faces_too_small": sum(1 for f in faces if f.get("status") == "too_small"),
        "faces_not_processed": sum(1 for f in faces if f.get("status") == "not_processed"),
        "max_faces_in_frame": max((len(r.get("faces", [])) for r in records), default=0),
        "recognised": dict(recognised),
        "unexpected_recognised": dict(unexpected),
        "unrecognised_faces": sum(1 for f in faces if f.get("status") == "unrecognized"),
        "genuine_distances": sorted(
            (f.get("nearest") or {}).get("distance")
            for f in faces
            if f.get("nearest")
            and any(
                (
                    f["nearest"].get("slug", "").lower() == e.lower()
                    or f["nearest"].get("name", "").lower() == e.lower()
                )
                for e in expect
            )
        ),
        "impostor_distances": sorted(
            (f.get("nearest") or {}).get("distance")
            for f in faces
            if f.get("status") == "unrecognized" and f.get("nearest")
        ),
        "face_size_median": _median([f.get("size") for f in processed]),
        "face_size_max": max((f.get("size") or 0 for f in faces), default=0),
        "sharpness_median": _median([f.get("sharpness") for f in processed]),
        "brightness_median": _median([f.get("brightness") for f in processed]),
        "detect_ms_median": _median([r.get("detect_ms") for r in records]),
        "embed_ms_median": _median([f.get("embed_ms") for f in processed]),
        "per_person": per_person,
        "detected_events": [{"t": e["t"], "name": e.get("name"), "rid": e.get("rid")} for e in switches],
        "switches_per_min": round(len(switches) / minutes, 1),
        "left_events": len(ev("left")),
        "duo_fallbacks": len(ev("duo_fallback")),
        "no_slides": [e.get("name") for e in ev("no_slides")],
        "errors": [e.get("msg") for e in ev("error")][:20],
        "warnings": [e.get("msg") for e in ev("warning")][:20],
        "capture_drops": len(ev("capture_drop")),
    }


# ── evaluators ──────────────────────────────────────────────────────────


class Judge:
    RANK = {"INFO": 0, "PASS": 1, "WARN": 2, "FAIL": 3}

    def __init__(self, start: str = "PASS"):
        self.level = start
        self.reasons: list[str] = []

    def add(self, level: str, reason: str) -> None:
        self.reasons.append(f"[{level}] {reason}")
        if self.RANK[level] > self.RANK[self.level]:
            self.level = level

    def ok(self, reason: str) -> None:
        self.reasons.append(f"[ok] {reason}")


def _common(j: Judge, o: dict) -> None:
    if o["errors"]:
        j.add("WARN", f"{len(o['errors'])} error line(s) in the app log during the window")
    if o["capture_drops"]:
        j.add("WARN", "capture store dropped frames — Pi too busy to keep every frame")


def _known_person(j: Judge, o: dict, who: str, scene_source: Optional[str], rate_pass: float) -> None:
    p = o["per_person"][who]
    if p["frames_recognised"] == 0:
        j.add(
            "FAIL",
            f"{who} was never recognised"
            + (f" (faces processed: {o['faces_processed']}, too small: {o['faces_too_small']})"),
        )
        return
    if p["rate"] < rate_pass:
        j.add(
            "FAIL" if p["rate"] < rate_pass / 2 else "WARN",
            f"{who} recognised in only {p['rate']:.0%} of frames with a processed face "
            f"(want ≥ {rate_pass:.0%})",
        )
    else:
        j.ok(f"{who} recognised in {p['rate']:.0%} of frames")
    if not p["detected_events"]:
        j.add("FAIL", f"{who} recognised but never sent to the display (no person.detected)")
        return
    if p["slides_loaded"] in (None, 0):
        j.add("FAIL", f"{who} displayed but no slides loaded — missing cutouts/scenes or gender mismatch")
    else:
        j.ok(f"{p['slides_loaded']} slide(s) loaded for {who}")
    lat = p["latency_to_display_s"] or p["latency_to_detected_s"]
    if lat is not None:
        (
            j.add("WARN", f"{lat:.1f} s from first frame to slides on screen")
            if lat > LATENCY_WARN_S
            else j.ok(f"{lat:.1f} s from first frame to slides on screen")
        )
    kinds = p["slide_kinds"]
    if scene_source and kinds:
        shown = "precomposed" if kinds.get("precomposed") and not kinds.get("composed") else "compose"
        if shown != scene_source:
            j.add(
                "WARN",
                f"asked for scene_source={scene_source} but {who}'s slides are {shown} (fallback) — {kinds}",
            )
        else:
            j.ok(f"slides are {shown} as requested ({kinds})")


def _unexpected(j: Judge, o: dict) -> None:
    if o["unexpected_recognised"]:
        j.add("FAIL", f"recognised someone not expected (false match): {o['unexpected_recognised']}")


def eval_empty(o, expect, label, scene_source):
    j = Judge()
    if o["recognised"]:
        j.add("FAIL", f"recognised people with nobody in view: {o['recognised']}")
    elif o["frames_with_faces"]:
        j.add(
            "WARN",
            f"faces detected with nobody in view ({o['frames_with_faces']} frame(s)) — "
            "posters, screens or reflections? see sample frames",
        )
    else:
        j.ok("no faces, no triggers")
    return j


def eval_single(o, expect, label, scene_source):
    j = Judge()
    _known_person(j, o, expect[0], scene_source, MATCH_RATE_PASS)
    _unexpected(j, o)
    return j


def eval_spot(o, expect, label, scene_source):
    j = Judge()
    who = expect[0]
    if o["faces_processed"] == 0:
        if o["faces_too_small"]:
            j.add("FAIL", f"face detected but too small to identify (largest {o['face_size_max']} px)")
        else:
            j.add("FAIL", "no face detected at this spot")
        return j
    p = o["per_person"][who]
    if p["rate"] >= SPOT_RATE_PASS:
        j.ok(f"{who} recognised in {p['rate']:.0%} of frames, face {o['face_size_median']} px")
    elif p["frames_recognised"]:
        j.add("WARN", f"{who} recognised in only {p['rate']:.0%} of frames")
    else:
        j.add("FAIL", f"face found ({o['face_size_median']} px) but {who} not recognised")
    _unexpected(j, o)
    return j


def eval_unknown(o, expect, label, scene_source):
    j = Judge()
    if o["recognised"]:
        j.add("FAIL", f"unseeded person matched as {o['recognised']} (false accept)")
    elif not o["unrecognised_faces"]:
        j.add("WARN", "no face was processed — nobody stood close enough?")
    else:
        j.ok(f"{o['unrecognised_faces']} face observation(s), none matched")
    d = o["impostor_distances"]
    if d and o.get("threshold") is not None:
        margin = round(d[0] - o["threshold"], 3)
        o["impostor_margin"] = margin
        if margin < IMPOSTOR_MARGIN_WARN and not o["recognised"]:
            j.add("WARN", f"closest impostor distance {d[0]} is only {margin} above the threshold")
    return j


def eval_duo(o, expect, label, scene_source):
    j = Judge()
    a, b = (e.lower() for e in expect[:2])
    duo = [
        e
        for e in o["detected_events"]
        if str(e.get("rid", "")).startswith("duo::") and a in e["rid"].lower() and b in e["rid"].lower()
    ]
    if duo:
        j.ok(f"duo display for {expect[0]} & {expect[1]}")
    elif o["duo_fallbacks"]:
        j.add("WARN", "both recognised but duo composition unavailable — fell back to one person")
    else:
        for who in expect[:2]:
            if not o["per_person"][who]["frames_recognised"]:
                j.add("FAIL", f"{who} never recognised")
        j.add("FAIL", "no duo display")
    _unexpected(j, o)
    return j


def eval_mixed(o, expect, label, scene_source):
    j = Judge()
    _known_person(j, o, expect[0], scene_source, MATCH_RATE_PASS)
    if any(str(e.get("rid", "")).startswith("duo::") for e in o["detected_events"]):
        j.add("FAIL", "duo shown although one person is unseeded")
    if not o["unrecognised_faces"]:
        j.add("WARN", "the unseeded person's face was never processed")
    _unexpected(j, o)
    return j


def eval_crowd(o, expect, label, scene_source):
    j = Judge()
    j.ok(
        f"up to {o['max_faces_in_frame']} faces in a frame; {o['faces_not_processed']} face observation(s) "
        f"skipped by the 2-per-frame cap"
    )
    if o["switches_per_min"] > FLICKER_PER_MIN_WARN:
        j.add("WARN", f"display switched {o['switches_per_min']}/min — flickering between people")
    for who in expect:
        if not o["per_person"][who]["detected_events"]:
            j.add("WARN", f"{who} was in the crowd but never shown")
    _unexpected(j, o)
    return j


def eval_return(o, expect, label, scene_source):
    j = Judge()
    who = expect[0]
    p = o["per_person"][who]
    det = [e for e in o["detected_events"] if _event_is(e, who)]
    if len(det) < 2:
        j.add(
            "FAIL",
            f"{who} shown {len(det)} time(s); expected again after returning"
            + ("" if o["left_events"] else " (never registered as left — out of view long enough?)"),
        )
        return j
    gap_t = det[1]["t"]
    o["redetect_t"] = gap_t
    j.ok(f"{who} shown, left ({o['left_events']} left event(s)), shown again")
    if p["frames_recognised"] and p["rate"] < MATCH_RATE_PASS / 2:
        j.add("WARN", f"recognition rate {p['rate']:.0%}")
    return j


def eval_walkby(o, expect, label, scene_source):
    j = Judge("INFO")
    j.ok(
        f"triggered display: {'yes' if o['detected_events'] else 'no'}; "
        f"faces in {o['frames_with_faces']} frame(s), largest {o['face_size_max']} px"
    )
    _unexpected(j, o)
    return j


def eval_queue(o, expect, label, scene_source):
    j = Judge()
    for who in expect:
        p = o["per_person"][who]
        if not p["detected_events"]:
            j.add("FAIL", f"{who} never shown")
        else:
            j.ok(f"{who} shown (rate {p['rate']:.0%}, latency {p['latency_to_detected_s']} s)")
    _unexpected(j, o)
    return j


def eval_spoof(o, expect, label, scene_source):
    j = Judge("INFO")
    who = expect[0]
    if o["per_person"][who]["frames_recognised"]:
        j.ok(f"photo/screen of {who} ACCEPTED — no liveness check (known limitation)")
    else:
        j.ok(f"photo/screen of {who} not accepted (face size {o['face_size_median']} px)")
    return j


EVALUATORS: dict[str, Callable] = {
    "empty": eval_empty,
    "single": eval_single,
    "unknown": eval_unknown,
    "duo": eval_duo,
    "mixed": eval_mixed,
    "crowd": eval_crowd,
    "return": eval_return,
    "walkby": eval_walkby,
    "queue": eval_queue,
    "spot": eval_spot,
    "spoof": eval_spoof,
}


def check_expect(name: str, expect: list[str]) -> Optional[str]:
    """An error message when --expect doesn't fit the scenario."""
    need = CATALOGUE[name].expect
    if need == "one" and len(expect) != 1:
        return f"'{name}' needs exactly one --expect <seeded person>"
    if need == "two" and len(expect) != 2:
        return f"'{name}' needs --expect A,B (two seeded people)"
    if need == "list" and not expect:
        return f"'{name}' needs --expect A,B,C (the people in the queue)"
    if need == "none" and expect:
        return f"'{name}' takes no --expect (nobody seeded should be in view)"
    return None


def evaluate(
    name: str,
    records: list[dict],
    events: list[dict],
    expect: list[str],
    start: float,
    end: float,
    label: str = "",
    scene_source: Optional[str] = None,
    threshold: Optional[float] = None,
    interval: float = 1.0,
) -> tuple[str, str, list[str], dict]:
    """Judge one window. Returns (verdict, one-line summary, reasons, observation)."""
    o = observe(records, events, expect, start, end, interval)
    o["threshold"] = threshold
    j = EVALUATORS[name](o, expect, label, scene_source)
    _common(j, o)
    summary = next(
        (r.split("] ", 1)[1] for r in j.reasons if r.startswith(("[FAIL]", "[WARN]"))),
        j.reasons[0].split("] ", 1)[1] if j.reasons else "",
    )
    return j.level, summary, j.reasons, o
