#!/usr/bin/env python3
"""Keep the `name` field in every data/base_scenes/*/scenes_config.json in sync
with the background image each entry points at.

`name` is documentation, not behaviour: composition keys off the scene id, and
`name` exists so a config can be read at a glance and so log lines say which
picture a slide came from. Because it is derived rather than authored, it must
never be hand-edited — regenerate it instead.

Run this after adding, removing or renaming a background:

    python scripts/sync_scene_names.py            # rewrite the configs
    python scripts/sync_scene_names.py --check    # report drift, change nothing

--check is what the test suite uses, so a config that drifts fails CI rather
than going unnoticed.

An entry whose background is missing keeps its scene id as the name. That is a
normal state — a key may be added before its art arrives — so it is reported
but not treated as an error.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wpu_client.paths import DATA_DIR  # noqa: E402
from wpu_client.services.face_recognition.base_composer import (  # noqa: E402
    derive_scene_name,
    find_scene_file,
)

BASE_SCENES_DIR = DATA_DIR / "base_scenes"
CONFIG_NAME = "scenes_config.json"

# Lists holding nothing but scalars — coordinate pairs and gender lists. Left
# on one line so the configs stay scannable; json.dump would otherwise put
# every number of every [x, y] on a line of its own, which is exactly the
# hand-reading these files need to support.
_SCALAR_LIST = re.compile(r"\[\s*([^\[\]{}]*?)\s*\]", re.DOTALL)


def dumps(config: dict) -> str:
    """Pretty JSON, with scalar lists kept inline."""
    text = json.dumps(config, indent=2)
    # json.dumps already emitted the separating commas; collapsing the
    # whitespace around them is all that is needed, and it cannot corrupt a
    # value the way re-joining on "," would.
    text = _SCALAR_LIST.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    return text + "\n"


def scene_configs() -> list[Path]:
    """Every scenes_config.json under data/base_scenes/, in a stable order."""
    return sorted(BASE_SCENES_DIR.glob(f"*/{CONFIG_NAME}"))


def expected_names(config_path: Path, config: dict) -> dict[str, str]:
    """The `name` each entry in this config should carry."""
    scenes_dir = config_path.parent
    names = {}
    for scene_id in config:
        image = find_scene_file(scenes_dir, scene_id)
        names[scene_id] = derive_scene_name(image) if image else scene_id
    return names


def with_names(config: dict, names: dict[str, str]) -> dict:
    """The config with `name` set, first, in every entry.

    Entry order and every other field are preserved — this file is hand-tuned
    and a reordering diff would bury the real change.
    """
    updated = {}
    for scene_id, meta in config.items():
        if not isinstance(meta, dict):
            updated[scene_id] = meta
            continue
        rest = {k: v for k, v in meta.items() if k != "name"}
        updated[scene_id] = {"name": names[scene_id], **rest}
    return updated


def sync(config_path: Path, check_only: bool) -> tuple[bool, list[str]]:
    """Sync one config. Returns (changed, notes)."""
    with open(config_path) as f:
        config = json.load(f)

    names = expected_names(config_path, config)
    updated = with_names(config, names)
    changed = updated != config

    notes = []
    label = f"{config_path.parent.name}/{CONFIG_NAME}"
    for scene_id, meta in config.items():
        if not isinstance(meta, dict):
            continue
        if find_scene_file(config_path.parent, scene_id) is None:
            notes.append(f"  {label}[{scene_id}]: no background image — named after its id")
        elif meta.get("name") != names[scene_id]:
            notes.append(
                f"  {label}[{scene_id}]: {meta.get('name')!r} -> {names[scene_id]!r}"
            )

    if changed and not check_only:
        config_path.write_text(dumps(updated))

    return changed, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if any config would change; write nothing.",
    )
    args = ap.parse_args()

    configs = scene_configs()
    if not configs:
        print(f"No {CONFIG_NAME} found under {BASE_SCENES_DIR}")
        return 1

    drifted = []
    for config_path in configs:
        changed, notes = sync(config_path, args.check)
        for note in notes:
            print(note)
        if changed:
            drifted.append(config_path)

    if not drifted:
        print(f"{len(configs)} config(s) already in sync.")
        return 0

    if args.check:
        print(
            f"\n{len(drifted)} config(s) out of sync. "
            f"Run: python scripts/sync_scene_names.py"
        )
        return 1

    print(f"\nUpdated {len(drifted)} config(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
