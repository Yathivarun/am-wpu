"""Session report: one self-contained HTML page (no network, opens anywhere)
plus summary.json, built from the session's result files and telemetry.

Runs on the Pi or on the laptop against an unpacked bundle
(`python sitetest.py report --session <path>`).
"""

import html
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from wpu_client.sitetest import telemetry
from wpu_client.sitetest.session import Session

GENUINE_TESTS = {"single", "spot", "return", "queue", "duo", "mixed", "crowd"}
IMPOSTOR_TESTS = {"unknown"}

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f1f0ed;--text-primary:#0b0b0b;
--text-secondary:#52514e;--grid:#e3e2de;--series-1:#2a78d6;--series-2:#eb6834;
--pass:#1d7a3a;--warn:#9a6700;--fail:#b42318;--info:#52514e}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#242423;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#3a3a38;
--series-1:#3987e5;--series-2:#d95926;--pass:#5cc27a;--warn:#e0b040;--fail:#f07a6e;--info:#c3c2b7}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242423;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#3a3a38;--series-1:#3987e5;--series-2:#d95926;
--pass:#5cc27a;--warn:#e0b040;--fail:#f07a6e;--info:#c3c2b7}
body{margin:0;background:var(--surface-1)}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 system-ui,sans-serif;
max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}
.muted{color:var(--text-secondary)}
table{border-collapse:collapse;width:100%;margin:8px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text-secondary);font-weight:600}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.tile{background:var(--surface-2);border-radius:8px;padding:10px 14px;min-width:90px}
.tile b{display:block;font-size:24px}
.v{font-weight:700}.v-PASS{color:var(--pass)}.v-WARN{color:var(--warn)}.v-FAIL{color:var(--fail)}
.v-INFO{color:var(--info)}
svg text{fill:var(--text-secondary);font-size:11px}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:16px;color:var(--text-secondary)}
.legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
.scroll{overflow-x:auto}
"""


def _e(x) -> str:
    return html.escape(str(x if x is not None else "–"))


def _verdict(v: str) -> str:
    return f'<span class="v v-{v}">{v}</span>'


# ── charts (inline SVG) ─────────────────────────────────────────────────


def line_chart(
    points: list[tuple[float, float]],
    title: str,
    unit: str,
    ref: Optional[float] = None,
    ref_label: str = "",
    width: int = 1000,
    height: int = 220,
) -> str:
    """One series over time, one axis. Native <title> tooltips on sampled points."""
    if len(points) < 2:
        return f'<p class="muted">{_e(title)}: not enough samples.</p>'
    pad_l, pad_r, pad_t, pad_b = 48, 12, 12, 26
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    lo, hi = min(ys + ([ref] if ref else [])), max(ys + ([ref] if ref else []))
    span = (hi - lo) or 1
    lo, hi = lo - span * 0.08, hi + span * 0.08

    def X(x):
        return pad_l + (x - x0) / ((x1 - x0) or 1) * (width - pad_l - pad_r)

    def Y(y):
        return pad_t + (hi - y) / (hi - lo) * (height - pad_t - pad_b)

    grid = []
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        grid.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/><text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" '
            f'text-anchor="end">{v:.0f}</text>'
        )
    for i in range(6):
        t = x0 + (x1 - x0) * i / 5
        grid.append(
            f'<text x="{X(t):.1f}" y="{height - 6}" text-anchor="middle">'
            f"{time.strftime('%H:%M', time.localtime(t))}</text>"
        )
    path = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in points)
    step = max(1, len(points) // 120)
    hits = "".join(
        f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="transparent">'
        f"<title>{time.strftime('%H:%M:%S', time.localtime(x))}  {y:g} {unit}</title></circle>"
        for x, y in points[::step]
    )
    refline = ""
    if ref is not None:
        refline = (
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" '
            f'stroke="var(--fail)" stroke-dasharray="4 4" stroke-width="1"/>'
            f'<text x="{width - pad_r}" y="{Y(ref) - 4:.1f}" text-anchor="end">{_e(ref_label)}</text>'
        )
    return (
        f'<h3 style="font-size:14px;margin:16px 0 4px">{_e(title)}</h3><div class="scroll">'
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_e(title)}">'
        f'{"".join(grid)}{refline}<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2" '
        f'stroke-linejoin="round"/>{hits}</svg></div>'
    )


def histogram(
    genuine: list[float],
    impostor: list[float],
    threshold: Optional[float],
    suggested: Optional[float],
    width: int = 1000,
    height: int = 240,
) -> str:
    """Genuine vs impostor nearest-gallery distances, grouped bars per bin."""
    if not genuine and not impostor:
        return '<p class="muted">No distance data yet — run single/spot and unknown scenarios.</p>'
    bins = [i / 20 for i in range(21)]  # 0.00 .. 1.00 step 0.05

    def counts(vals):
        c = [0] * 20
        for v in vals:
            c[min(19, max(0, int(v * 20)))] += 1
        return [x / max(1, len(vals)) for x in c]

    g, im = counts(genuine), counts(impostor)
    pad_l, pad_r, pad_t, pad_b = 48, 12, 12, 28
    top = max(g + im) or 1
    bw = (width - pad_l - pad_r) / 20

    def Y(v):
        return pad_t + (1 - v / top) * (height - pad_t - pad_b)

    parts = []
    for i in range(5):
        v = top * i / 4
        parts.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
            f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>'
        )
    for i in range(20):
        x = pad_l + i * bw
        for j, (vals, color, name) in enumerate(
            ((g, "var(--series-1)", "genuine"), (im, "var(--series-2)", "impostor"))
        ):
            h = (height - pad_b) - Y(vals[i])
            if vals[i] > 0:
                bx = x + 2 + j * (bw - 4) / 2
                parts.append(
                    f'<rect x="{bx:.1f}" y="{Y(vals[i]):.1f}" width="{(bw - 6) / 2:.1f}" '
                    f'height="{max(h, 1):.1f}" rx="2" fill="{color}"><title>{name} '
                    f"{bins[i]:.2f}–{bins[i + 1]:.2f}: {vals[i]:.0%}</title></rect>"
                )
        if i % 2 == 0:
            parts.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle">{bins[i]:.1f}</text>')

    def marker(v, label, dash):
        x = pad_l + v * 20 * bw
        return (
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{height - pad_b}" '
            'stroke="var(--text-primary)" '
            f'stroke-dasharray="{dash}" stroke-width="1"/>'
            f'<text x="{x + 4:.1f}" y="{pad_t + 10}">{_e(label)}</text>'
        )

    if threshold is not None:
        parts.append(marker(threshold, f"threshold {threshold}", "4 4"))
    if suggested is not None and suggested != threshold:
        parts.append(marker(suggested, f"suggested {suggested}", "1 3"))
    legend = (
        '<div class="legend"><span><i style="background:var(--series-1)"></i>genuine (seeded person vs '
        f'own entry, n={len(genuine)})</span><span><i style="background:var(--series-2)"></i>impostor '
        f"(unseeded vs nearest entry, n={len(impostor)})</span></div>"
    )
    return (
        f'{legend}<div class="scroll"><svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="distance histogram">{"".join(parts)}</svg></div>'
    )


# ── numbers ─────────────────────────────────────────────────────────────


def _pct(vals: list[float], q: float) -> Optional[float]:
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return round(vals[k], 3)


def threshold_advice(genuine: list[float], impostor: list[float], current: Optional[float]) -> dict:
    out = {"genuine_n": len(genuine), "impostor_n": len(impostor), "current": current}
    if current is not None:
        out["false_reject_at_current"] = (
            round(sum(g >= current for g in genuine) / len(genuine), 3) if genuine else None
        )
        out["false_accept_at_current"] = (
            round(sum(i < current for i in impostor) / len(impostor), 3) if impostor else None
        )
    g95, i05 = _pct(genuine, 0.95), _pct(impostor, 0.05)
    out["genuine_p95"], out["impostor_p05"] = g95, i05
    if g95 is not None and i05 is not None:
        out["suggested"] = round((g95 + i05) / 2, 2)
        out["separable"] = g95 < i05
    return out


def build(session: Session) -> Path:
    results = session.all_results()
    meta = session.meta()
    counts = Counter(r["verdict"] for r in results)
    tele = telemetry.load(session)
    tsum = telemetry.summarise(tele)
    tverdict, treasons = telemetry.verdict(tsum)

    genuine, impostor = [], []
    threshold = None
    for r in results:
        d = r.get("details") or {}
        threshold = d.get("threshold", threshold)
        if r["test"] in GENUINE_TESTS:
            genuine += [v for v in d.get("genuine_distances", []) if v is not None]
        if r["test"] in IMPOSTOR_TESTS:
            impostor += [v for v in d.get("impostor_distances", []) if v is not None]
    if threshold is None:
        threshold = (session.app_settings() or {}).get("threshold")
    advice = threshold_advice(genuine, impostor, threshold)

    spots = [r for r in results if r["test"] == "spot"]
    timed = [r for r in results if r["test"] in GENUINE_TESTS and r.get("details", {}).get("per_person")]

    out = []
    out.append(f"<h1>Site test — {_e(meta.get('name', session.name))}</h1>")
    out.append(
        f'<p class="muted">{_e(meta.get("started_local"))} → {_e(time.strftime("%Y-%m-%d %H:%M"))} · '
        f"session {_e(session.name)} · app {_e(session.app_settings())}</p>"
    )
    out.append(
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><b class="v-{v}">{counts.get(v, 0)}</b>{v}</div>'
            for v in ("PASS", "WARN", "FAIL", "INFO")
        )
        + f'<div class="tile"><b class="v-{tverdict}">{tverdict}</b>thermal / power</div></div>'
    )

    out.append(
        "<h2>Results</h2><div class='scroll'><table><tr><th>#</th><th>Test</th><th>Label</th><th>Verdict</th>"
        "<th>Summary</th><th>When</th></tr>"
    )
    for r in results:
        out.append(
            f"<tr><td>{r['seq']}</td><td>{_e(r['test'])}</td><td>{_e(r['label'])}</td>"
            f"<td>{_verdict(r['verdict'])}</td><td>{_e(r['summary'])}</td>"
            f"<td class='muted'>{_e((r.get('started_local') or r.get('ended_local') or '')[11:])}</td></tr>"
        )
    out.append("</table></div>")

    out.append("<h2>Thermal, power and stability</h2>")
    out.append(f"<p>{_verdict(tverdict)} — " + "; ".join(_e(x) for x in treasons) + "</p>")
    out.append(
        f'<p class="muted">{tsum.get("samples", 0)} samples over {tsum.get("minutes", 0)} min · '
        f"temp {_e(tsum.get('temp_c'))} · clock {_e(tsum.get('arm_mhz'))} MHz · "
        f"app memory {_e(tsum.get('app_rss_mb'))} MB · "
        f"min disk free {_e(tsum.get('disk_free_gb_min'))} GB</p>"
    )
    pts = [(float(r["t"]), float(r["temp_c"])) for r in tele if r.get("temp_c") not in ("", None)]
    out.append(line_chart(pts, "SoC temperature (°C)", "°C", ref=80, ref_label="throttle 80 °C"))
    pts = [(float(r["t"]), float(r["app_rss_mb"])) for r in tele if r.get("app_rss_mb") not in ("", None)]
    out.append(line_chart(pts, "App memory (MB) — a steady climb means a leak", "MB"))
    pts = [(float(r["t"]), float(r["arm_mhz"])) for r in tele if r.get("arm_mhz") not in ("", None)]
    out.append(line_chart(pts, "CPU clock (MHz) — dips mean throttling", "MHz"))

    out.append("<h2>Recognition threshold</h2>")
    out.append(histogram(genuine, impostor, threshold, advice.get("suggested")))
    out.append(
        f"<p>Current threshold {_e(threshold)}: false rejects {_e(advice.get('false_reject_at_current'))}, "
        f"false accepts {_e(advice.get('false_accept_at_current'))} (share of observations). "
        f"Genuine p95 {_e(advice.get('genuine_p95'))}, impostor p5 {_e(advice.get('impostor_p05'))}"
        + (
            f" → suggested <b>{advice['suggested']}</b>"
            + (
                ""
                if advice.get("separable")
                else " (classes overlap — a threshold alone can't separate them)"
            )
            if advice.get("suggested") is not None
            else ""
        )
        + "</p>"
    )

    out.append("<h2>Placement / range</h2>")
    if spots:
        out.append(
            "<div class='scroll'><table><tr><th>Spot</th><th>Verdict</th><th>Face px</th><th>Recognised</th>"
            "<th>Distance</th><th>Sharpness</th><th>Brightness</th></tr>"
        )
        for r in spots:
            d = r["details"]
            pp = next(iter((d.get("per_person") or {}).values()), {})
            out.append(
                f"<tr><td>{_e(r['label'])}</td><td>{_verdict(r['verdict'])}</td>"
                f"<td>{_e(d.get('face_size_median'))}</td><td>{_e(pp.get('rate'))}</td>"
                f"<td>{_e(pp.get('genuine_distance_median'))}</td><td>{_e(d.get('sharpness_median'))}</td>"
                f"<td>{_e(d.get('brightness_median'))}</td></tr>"
            )
        out.append("</table></div>")
    else:
        out.append('<p class="muted">No spot tests yet.</p>')

    out.append("<h2>Time to slides</h2>")
    if timed:
        out.append(
            "<div class='scroll'><table><tr><th>Test</th><th>Person</th><th>Rate</th><th>To detected (s)</th>"
            "<th>To display (s)</th><th>Slides</th><th>Kinds</th></tr>"
        )
        for r in timed:
            for who, p in r["details"]["per_person"].items():
                out.append(
                    f"<tr><td>{_e(r['test'])} {_e(r['label'])}</td><td>{_e(who)}</td>"
                    f"<td>{_e(p.get('rate'))}</td>"
                    f"<td>{_e(p.get('latency_to_detected_s'))}</td><td>{_e(p.get('latency_to_display_s'))}</td>"
                    f"<td>{_e(p.get('slides_loaded'))}</td><td>{_e(p.get('slide_kinds'))}</td></tr>"
                )
        out.append("</table></div>")
    else:
        out.append('<p class="muted">No recognition scenarios yet.</p>')

    errs = [(r["seq"], r["test"], m) for r in results for m in (r.get("details") or {}).get("errors", [])]
    if errs:
        out.append(
            "<h2>Errors seen during tests</h2><div class='scroll'><table>"
            "<tr><th>#</th><th>Test</th><th>Line</th></tr>"
        )
        out += [f"<tr><td>{s}</td><td>{_e(t)}</td><td>{_e(m)}</td></tr>" for s, t, m in errs[:200]]
        out.append("</table></div>")

    page = (
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>Site Test Report</title><style>{CSS}</style>"
        f"</head><body><main class='viz-root'>{''.join(out)}</main></body></html>"
    )
    path = session.file("report.html")
    path.write_text(page, encoding="utf-8")
    session.file("summary.json").write_text(
        json.dumps(
            {
                "session": session.name,
                "verdicts": dict(counts),
                "telemetry": tsum,
                "telemetry_verdict": tverdict,
                "threshold": advice,
                "spots": [{"label": r["label"], "verdict": r["verdict"]} for r in spots],
            },
            indent=2,
            default=str,
        )
    )
    return path


def median(vals):
    return statistics.median(vals) if vals else None
