"""Schema checks for the shipped data/base_scenes/*/scenes_config.json files.

These are hand-edited as scenes get tuned, and a malformed entry fails
silently at runtime — the composer skips it and the visitor just gets fewer
slides, with only a log line to show for it. Validating the shape here turns
that into a test failure instead.

Deliberately NOT asserted: that a background image exists for each key. Keys
without art are expected (art is dropped in later, and the server's own face
config omits scenes it has images for).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sync_scene_names import sync  # noqa: E402
from wpu_client.services.face_recognition.base_composer import (  # noqa: E402
    derive_scene_name,
    find_scene_file,
)

BASE_SCENES = Path("data/base_scenes")

BODY_SINGLE = BASE_SCENES / "sau_single" / "scenes_config.json"
BODY_DUO = BASE_SCENES / "sau_duo" / "scenes_config.json"
FACE_SINGLE = BASE_SCENES / "fru_single" / "scenes_config.json"
FACE_DUO = BASE_SCENES / "fru_duo" / "scenes_config.json"

ALL_CONFIGS = [BODY_SINGLE, BODY_DUO, FACE_SINGLE, FACE_DUO]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _assert_anchor(value, label):
    assert isinstance(value, list), f"{label} must be a list"
    assert len(value) == 2, f"{label} must be [x, y]"
    assert all(isinstance(c, (int, float)) for c in value), f"{label} must be numeric"


def _assert_scale(value, label):
    assert isinstance(value, (int, float)), f"{label} must be numeric"
    assert value > 0, f"{label} must be positive"


def _assert_face_anchor(anchor, label):
    assert isinstance(anchor, dict), f"{label} must be an object"
    _assert_anchor(anchor["target_eye_midpoint"], f"{label}.target_eye_midpoint")
    dist = anchor["target_eye_distance"]
    assert isinstance(dist, (int, float)) and dist > 0, (
        f"{label}.target_eye_distance must be a positive number"
    )
    assert isinstance(anchor["target_tilt_angle"], (int, float)), (
        f"{label}.target_tilt_angle must be numeric"
    )


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.parent.name)
def test_config_exists_and_is_a_json_object(path):
    assert path.exists(), f"{path} is missing"
    config = _load(path)
    assert isinstance(config, dict)
    assert config, f"{path} is empty"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.parent.name)
def test_scene_ids_are_filename_stems(path):
    """Keys are matched against '<key>.png' on disk, so they must be plain
    stems — no extension, no path separator."""
    for scene_id in _load(path):
        assert isinstance(scene_id, str)
        assert "/" not in scene_id and "\\" not in scene_id
        assert not Path(scene_id).suffix, f"{scene_id!r} looks like a filename"


def test_body_single_shape():
    for scene_id, meta in _load(BODY_SINGLE).items():
        _assert_scale(meta["scale"], f"sau_single[{scene_id}].scale")
        _assert_anchor(meta["anchor"], f"sau_single[{scene_id}].anchor")


def test_body_duo_shape():
    for scene_id, meta in _load(BODY_DUO).items():
        for n in (1, 2):
            _assert_scale(meta[f"scale_{n}"], f"sau_duo[{scene_id}].scale_{n}")
            _assert_anchor(meta[f"anchor_{n}"], f"sau_duo[{scene_id}].anchor_{n}")


def test_body_configs_carry_no_gender_key():
    """Body composition has no gender filter — matching the server's
    body/crops.json. A gender key here would be silently ignored."""
    for path in (BODY_SINGLE, BODY_DUO):
        for scene_id, meta in _load(path).items():
            assert "gender" not in meta, f"{path.parent.name}[{scene_id}]"


def test_face_single_shape():
    for scene_id, meta in _load(FACE_SINGLE).items():
        _assert_face_anchor(meta["face_anchor"], f"fru_single[{scene_id}].face_anchor")


def test_face_duo_shape():
    for scene_id, meta in _load(FACE_DUO).items():
        for n in (1, 2):
            _assert_face_anchor(
                meta[f"face_anchor_{n}"], f"fru_duo[{scene_id}].face_anchor_{n}"
            )


@pytest.mark.parametrize("path", [FACE_SINGLE, FACE_DUO], ids=lambda p: p.parent.name)
def test_face_gender_lists_are_lowercase_and_known(path):
    """Gender matching is exact against the lowercased value from the server,
    whose enum is {male, female}. A stray 'Male' or 'M' here would silently
    exclude that scene for everyone."""
    for scene_id, meta in _load(path).items():
        if "gender" not in meta:
            continue
        genders = meta["gender"]
        assert isinstance(genders, list) and genders, f"{scene_id}: empty gender list"
        for g in genders:
            assert g == g.lower(), f"{scene_id}: {g!r} must be lowercase"
            assert g in ("male", "female", "unknown", "mixed"), (
                f"{scene_id}: unknown gender {g!r}"
            )


def test_mixed_is_only_used_in_duo_configs():
    """'mixed' describes a PAIR — one man and one woman. In a single-person
    config it can never match, so a scene carrying it would silently never be
    shown."""
    for scene_id, meta in _load(FACE_SINGLE).items():
        assert "mixed" not in meta.get("gender", []), (
            f"fru_single[{scene_id}]: 'mixed' has no meaning for one person"
        )


def test_duo_anchors_are_distinct():
    """Two people stacked on the same anchor would render one on top of the
    other — always a config mistake."""
    for scene_id, meta in _load(BODY_DUO).items():
        assert meta["anchor_1"] != meta["anchor_2"], f"sau_duo[{scene_id}]"
    for scene_id, meta in _load(FACE_DUO).items():
        assert (
            meta["face_anchor_1"]["target_eye_midpoint"]
            != meta["face_anchor_2"]["target_eye_midpoint"]
        ), f"fru_duo[{scene_id}]"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.parent.name)
def test_every_entry_carries_a_name(path):
    """`name` is what makes these files readable without cross-referencing the
    directory listing, and what log lines quote when a slide is composed."""
    for scene_id, meta in _load(path).items():
        assert "name" in meta, f"{path.parent.name}[{scene_id}] has no name"
        assert isinstance(meta["name"], str) and meta["name"].strip(), (
            f"{path.parent.name}[{scene_id}]: name must be a non-empty string"
        )


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.parent.name)
def test_names_match_their_background_image(path):
    """`name` is derived, never authored — hand-editing it would drift the
    moment a background is renamed."""
    for scene_id, meta in _load(path).items():
        image = find_scene_file(path.parent, scene_id)
        expected = derive_scene_name(image) if image else scene_id
        assert meta["name"] == expected, (
            f"{path.parent.name}[{scene_id}]: name is {meta['name']!r}, "
            f"expected {expected!r} — run scripts/sync_scene_names.py"
        )


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.parent.name)
def test_configs_are_in_sync_with_disk(path):
    """The whole-file check the sync script performs, so adding a background
    without regenerating fails here rather than going unnoticed."""
    changed, notes = sync(path, check_only=True)
    assert not changed, (
        f"{path} is out of sync — run scripts/sync_scene_names.py\n" + "\n".join(notes)
    )
