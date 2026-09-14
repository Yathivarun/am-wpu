#!/usr/bin/env python3
"""Give every entry in data/base_scenes/*/scenes_config.json a `name`.

`name` is documentation, not behaviour: composition keys off the scene id, and
`name` exists so a config can be read at a glance and so log lines say which
picture a slide came from — "Horse and Rider" rather than "1".

A good name describes what the scene SHOWS, and only a person can write that.
So this script never invents one over yours: it fills in entries that have no
name yet, seeding them from the background's filename, and leaves every
existing name alone. Rename a scene by editing it here.

Run it after adding a background, or after adding a key to a config:

    python scripts/sync_scene_names.py            # fill in what is missing
    python scripts/sync_scene_names.py --check    # report gaps, change nothing

--check is what the test suite uses, so an entry that reaches CI with no name
fails rather than going unnoticed.

An entry whose background is missing is seeded with its scene id. That is a
normal state — a key may be added before its art arrives — and it is a
placeholder worth replacing, not an error.
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


def seed_name(scenes_dir: Path, scene_id: str) -> str:
    """A starting name for an entry that has none: its background's filename,
    or the scene id when there is no background to read one from."""
    image = find_scene_file(scenes_dir, scene_id)
    return derive_scene_name(image) if image else scene_id


def with_names(config: dict, scenes_dir: Path) -> dict:
    """The config with `name` present, first, in every entry.

    Existing names are carried through untouched. Entry order and every other
    field are preserved too — these files are hand-tuned and a reordering diff
    would bury the real change.
    """
    updated = {}
    for scene_id, meta in config.items():
        if not isinstance(meta, dict):
            updated[scene_id] = meta
            continue
        name = meta.get("name")
        if not (isinstance(name, str) and name.strip()):
            name = seed_name(scenes_dir, scene_id)
        rest = {k: v for k, v in meta.items() if k != "name"}
        updated[scene_id] = {"name": name, **rest}
    return updated


def sync(config_path: Path, check_only: bool) -> tuple[bool, list[str]]:
    """Fill in missing names in one config. Returns (changed, notes)."""
    with open(config_path) as f:
        config = json.load(f)

    updated = with_names(config, config_path.parent)
    changed = updated != config

    notes = []
    label = f"{config_path.parent.name}/{CONFIG_NAME}"
    for scene_id, meta in config.items():
        if not isinstance(meta, dict):
            continue
        name = meta.get("name")
        if not (isinstance(name, str) and name.strip()):
            notes.append(
                f"  {label}[{scene_id}]: no name — seeded {updated[scene_id]['name']!r}, "
                f"replace it with what the scene shows"
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
        print(f"{len(configs)} config(s): every entry has a name.")
        return 0

    if args.check:
        print(
            f"\n{len(drifted)} config(s) have entries with no name. "
            f"Run: python scripts/sync_scene_names.py"
        )
        return 1

    print(f"\nUpdated {len(drifted)} config(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
