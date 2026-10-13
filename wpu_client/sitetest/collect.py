"""Bundle a session for the laptop: results, telemetry, logs, config, gallery
metadata, visit log, and the capture-store days the session touched.

The capture images are the bulk; --no-captures leaves out the JPEGs but keeps
every index.jsonl, which is what the report's numbers come from.
"""

import json
import tarfile
import time
from pathlib import Path

from wpu_client.paths import CONFIG_DIR, DATA_DIR
from wpu_client.sitetest import system
from wpu_client.sitetest.session import Session


def bundle(session: Session, out_dir: Path, with_captures: bool = True, with_dataset: bool = False) -> Path:
    meta = session.meta()
    started = meta.get("started") or session.path.stat().st_mtime
    now = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{session.name}.tar.gz"

    # Logs for the whole session, all three units (the journal is the full story).
    logs = session.file("logs")
    logs.mkdir(exist_ok=True)
    for unit in system.UNITS.values():
        res = system.run(
            ["journalctl", "-u", unit, "--since", f"@{int(started)}", "--no-pager", "-o", "short-iso"],
            timeout=120,
        )
        if res.stdout.strip():
            (logs / f"{unit}.log").write_text(res.stdout)
    kern = system.run(
        ["journalctl", "-k", "--since", f"@{int(started)}", "--no-pager", "-o", "short-iso"], timeout=120
    )
    (logs / "kernel.log").write_text(kern.stdout)
    (logs / "check.json").write_text(
        system.run([system.python_exe(), "main.py", "--check", "--diagnostic", "--json"], timeout=120).stdout
    )
    (logs / "git.txt").write_text(
        system.run(["git", "-C", str(system.PROJECT_ROOT), "log", "-1", "--format=%H %d %s"]).stdout
    )

    days = {time.strftime("%Y%m%d", time.localtime(t)) for t in (started, now)}
    day = started
    while day < now:
        days.add(time.strftime("%Y%m%d", time.localtime(day)))
        day += 86400

    with tarfile.open(out, "w:gz") as tar:
        tar.add(session.path, arcname=f"{session.name}/session")
        for path in (CONFIG_DIR / "config.yaml", system.APP_LOG):
            if path.exists():
                tar.add(path, arcname=f"{session.name}/{path.name}")
        visit_log = DATA_DIR / "dataset" / "visit_log.csv"
        if visit_log.exists():
            tar.add(visit_log, arcname=f"{session.name}/visit_log.csv")
        gdir = system.gallery_dir()
        if gdir.is_dir():
            for person in gdir.iterdir():
                for item in person.glob("*.json"):
                    tar.add(item, arcname=f"{session.name}/gallery/{person.name}/{item.name}")
                for item in person.glob("seed_*.jpg"):
                    tar.add(item, arcname=f"{session.name}/gallery/{person.name}/{item.name}")
        croot = system.captures_dir()
        for d in sorted(days):
            day_dir = croot / d
            if not day_dir.is_dir():
                continue
            if with_captures:
                tar.add(day_dir, arcname=f"{session.name}/captures/{d}")
            elif (day_dir / "index.jsonl").exists():
                tar.add(day_dir / "index.jsonl", arcname=f"{session.name}/captures/{d}/index.jsonl")
        if with_dataset and (DATA_DIR / "dataset").is_dir():
            tar.add(DATA_DIR / "dataset", arcname=f"{session.name}/dataset")
    session.file("collected.json").write_text(
        json.dumps(
            {
                "at": now,
                "bundle": str(out),
                "captures": with_captures,
                "size_mb": round(out.stat().st_size / 1e6, 1),
            },
            indent=2,
        )
    )
    return out
