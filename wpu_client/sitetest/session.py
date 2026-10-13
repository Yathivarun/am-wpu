"""Test sessions and result files.

A session is one site visit: data/sitetest/<YYYYMMDD-HHMM>_<name>/. The
folder name of the active one is kept in data/sitetest/CURRENT, so every
command lands in the same place without repeating --session.

Every test writes two files into <session>/results/:

    <NNN>_<test>[_<label>]_<VERDICT>.json   everything measured, machine-readable
    <NNN>_<test>[_<label>]_<VERDICT>.md     the same for a human: verdict, why, numbers

and appends one line to <session>/results.jsonl. Verdicts: PASS, FAIL,
WARN (works, but not as it should) and INFO (measured, nothing to judge).
"""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from wpu_client.paths import DATA_DIR

ROOT = DATA_DIR / "sitetest"
CURRENT = ROOT / "CURRENT"
VERDICTS = ("PASS", "FAIL", "WARN", "INFO")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:40] or "x"


class Session:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.results = self.path / "results"

    @property
    def name(self) -> str:
        return self.path.name

    # ── where things live ────────────────────────────────────────────────
    @property
    def telemetry_csv(self) -> Path:
        return self.path / "telemetry.csv"

    @property
    def app_state(self) -> Path:
        """What `app start` last configured (scene source, threshold, ...)."""
        return self.path / "app_state.json"

    def file(self, name: str) -> Path:
        return self.path / name

    def meta(self) -> dict:
        try:
            return json.loads((self.path / "session.json").read_text())
        except (OSError, ValueError):
            return {}

    def app_settings(self) -> dict:
        try:
            return json.loads(self.app_state.read_text())
        except (OSError, ValueError):
            return {}

    # ── results ──────────────────────────────────────────────────────────
    def record(
        self,
        test: str,
        verdict: str,
        summary: str,
        details: Optional[dict] = None,
        label: str = "",
        reasons: Optional[list] = None,
        started: Optional[float] = None,
        ended: Optional[float] = None,
        artifacts: Optional[list] = None,
    ) -> Path:
        """Write one result (json + md) and index it. Returns the json path.

        `artifacts`: files (sample frames, crops) copied into results/<stem>/
        so a result stays readable after the capture store evicts them.
        """
        verdict = verdict.upper()
        assert verdict in VERDICTS, verdict
        self.results.mkdir(parents=True, exist_ok=True)
        seq = len(list(self.results.glob("*.json"))) + 1
        stem = f"{seq:03d}_{_slug(test)}" + (f"_{_slug(label)}" if label else "") + f"_{verdict}"
        ended = ended or time.time()
        copied = []
        for i, src in enumerate(artifacts or []):
            src = Path(src)
            if src.is_file():
                dest_dir = self.results / stem
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / f"{i:02d}_{src.parent.name}_{src.name}"
                shutil.copy2(src, dest)
                copied.append(dest.relative_to(self.path).as_posix())
        doc = {
            "seq": seq,
            "test": test,
            "label": label,
            "verdict": verdict,
            "summary": summary,
            "reasons": reasons or [],
            "started": started,
            "ended": ended,
            "started_local": _local(started) if started else None,
            "ended_local": _local(ended),
            "app": self.app_settings(),
            "artifacts": copied,
            "details": details or {},
        }
        json_path = self.results / f"{stem}.json"
        json_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        (self.results / f"{stem}.md").write_text(render_md(doc), encoding="utf-8")
        with open(self.path / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {k: doc[k] for k in ("seq", "test", "label", "verdict", "summary", "started", "ended")},
                    default=str,
                )
                + "\n"
            )
        return json_path

    def all_results(self) -> list[dict]:
        out = []
        for p in sorted(self.results.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return out


def _local(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def render_md(doc: dict) -> str:
    lines = [
        f"# {doc['test']}" + (f" — {doc['label']}" if doc["label"] else ""),
        "",
        f"**{doc['verdict']}** — {doc['summary']}",
        "",
    ]
    if doc["started_local"]:
        lines.append(f"- Window: {doc['started_local']} → {doc['ended_local']}")
    else:
        lines.append(f"- At: {doc['ended_local']}")
    app = doc.get("app") or {}
    if app:
        lines.append(
            f"- App: scene_source={app.get('scene_source')}, "
            f"threshold={app.get('threshold')}, captures={app.get('capture')}"
        )
    if doc["reasons"]:
        lines += ["", "## Why", ""] + [f"- {r}" for r in doc["reasons"]]
    if doc.get("artifacts"):
        lines += ["", "## Sample frames", ""] + [f"- {a.removeprefix('results/')}" for a in doc["artifacts"]]
    if doc["details"]:
        lines += [
            "",
            "## Details",
            "",
            "```json",
            json.dumps(doc["details"], indent=2, default=str)[:20000],
            "```",
        ]
    return "\n".join(lines) + "\n"


# ── current-session pointer ─────────────────────────────────────────────


def start(name: str, note: str = "") -> Session:
    ROOT.mkdir(parents=True, exist_ok=True)
    folder = time.strftime("%Y%m%d-%H%M") + "_" + _slug(name)
    session = Session(ROOT / folder)
    session.results.mkdir(parents=True, exist_ok=True)
    (session.path / "session.json").write_text(
        json.dumps(
            {"name": name, "note": note, "started": time.time(), "started_local": _local(time.time())},
            indent=2,
        )
    )
    CURRENT.write_text(folder)
    return session


def current(required: bool = True) -> Optional[Session]:
    try:
        folder = CURRENT.read_text().strip()
        if folder and (ROOT / folder).is_dir():
            return Session(ROOT / folder)
    except OSError:
        pass
    if required:
        raise SystemExit("No active session — start one: python sitetest.py session start --name <site>")
    return None


def by_name(folder: str) -> Session:
    path = Path(folder)
    if not path.is_dir():
        path = ROOT / folder
    if not path.is_dir():
        raise SystemExit(f"No such session: {folder}")
    return Session(path)
