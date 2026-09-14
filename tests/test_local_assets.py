"""Diagnostic-mode cutout resolution.

The gallery folder carries no labels, so which of its images is the body and
which is the face is worked out from metadata, then filenames, then shape.
Getting that wrong is silent — the face gets pasted where the body should go,
or nothing composes at all — so each pass is pinned here.
"""

import cv2
import numpy as np
import pytest

from wpu_client.services.face_recognition.local_assets import resolve_local_cutouts


def _write(path, width, height):
    """An RGBA image of the given shape, which is all the resolver measures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 4), np.uint8)
    img[:, :, 3] = 255
    assert cv2.imwrite(str(path), img)
    return path


# Real proportions from a seeded gallery: a full standing figure vs a face
# crop. Divided down to keep the test fast.
BODY = (152, 394)
FACE = (128, 128)


def test_missing_directory_is_not_an_error(tmp_path):
    assert resolve_local_cutouts(tmp_path / "absent") == {"sau": None, "fru": None}


def test_empty_directory_is_not_an_error(tmp_path):
    (tmp_path / "sketches").mkdir()
    assert resolve_local_cutouts(tmp_path / "sketches") == {"sau": None, "fru": None}


def test_meta_keys_win_over_everything(tmp_path):
    """The escape hatch: explicit filenames override both later passes, even
    when the names and shapes say the opposite."""
    body = _write(tmp_path / "looks_like_a_face.png", *FACE)
    face = _write(tmp_path / "person.png", *BODY)

    got = resolve_local_cutouts(tmp_path, {"sau": body.name, "fru": face.name})

    assert got["sau"] == body
    assert got["fru"] == face


@pytest.mark.parametrize("key", ["sau", "sau_cutout", "body"])
def test_body_meta_key_aliases(tmp_path, key):
    body = _write(tmp_path / "anything.png", *BODY)
    assert resolve_local_cutouts(tmp_path, {key: body.name})["sau"] == body


def test_meta_value_may_not_escape_the_gallery_folder(tmp_path):
    """A hand-edited meta.json is not a path — a traversal is ignored, not
    followed, and resolution falls through to the later passes."""
    outside = _write(tmp_path / "outside.png", *BODY)
    sketches = tmp_path / "sketches"
    _write(sketches / "person.png", *BODY)

    got = resolve_local_cutouts(sketches, {"sau": f"../{outside.name}"})

    assert got["sau"] == sketches / "person.png"


def test_meta_naming_a_missing_file_falls_through(tmp_path):
    _write(tmp_path / "person.png", *BODY)
    assert resolve_local_cutouts(tmp_path, {"sau": "gone.png"})["sau"] == tmp_path / "person.png"


@pytest.mark.parametrize("stem", ["sau_cutout", "body_v2", "person"])
def test_body_filename_hints(tmp_path, stem):
    body = _write(tmp_path / f"{stem}.png", *BODY)
    assert resolve_local_cutouts(tmp_path)["sau"] == body


@pytest.mark.parametrize("stem", ["fru_cutout", "face-01"])
def test_face_filename_hints(tmp_path, stem):
    face = _write(tmp_path / f"{stem}.png", *FACE)
    assert resolve_local_cutouts(tmp_path)["fru"] == face


def test_shape_resolves_unhinted_filenames(tmp_path):
    """The case the seeded galleries in this repo actually hit: files named
    after their source, not their role."""
    body = _write(tmp_path / "person.png", *BODY)
    face = _write(tmp_path / "d6.png", *FACE)

    got = resolve_local_cutouts(tmp_path)

    assert got["sau"] == body
    assert got["fru"] == face


def test_shape_resolves_when_neither_name_hints(tmp_path):
    body = _write(tmp_path / "a.png", *BODY)
    face = _write(tmp_path / "b.png", *FACE)

    got = resolve_local_cutouts(tmp_path)

    assert got["sau"] == body
    assert got["fru"] == face


def test_a_lone_square_image_is_a_face_not_a_body(tmp_path):
    """Half a resolution is fine — the caller composes whichever it got. What
    must not happen is a face being pasted as a body."""
    face = _write(tmp_path / "d4.png", *FACE)

    got = resolve_local_cutouts(tmp_path)

    assert got["fru"] == face
    assert got["sau"] is None


def test_a_lone_tall_image_is_a_body_not_a_face(tmp_path):
    body = _write(tmp_path / "whoever.png", *BODY)

    got = resolve_local_cutouts(tmp_path)

    assert got["sau"] == body
    assert got["fru"] is None


def test_hinted_file_is_not_reused_for_the_other_role(tmp_path):
    """One image named as the body must not also be handed back as the face."""
    body = _write(tmp_path / "sau.png", *BODY)

    got = resolve_local_cutouts(tmp_path)

    assert got["sau"] == body
    assert got["fru"] is None


def test_non_image_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "images").mkdir()  # a legacy sub-folder, not a candidate
    body = _write(tmp_path / "person.png", *BODY)

    got = resolve_local_cutouts(tmp_path)

    assert got["sau"] == body
    assert got["fru"] is None


def test_unreadable_image_does_not_raise(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"not a png")
    face = _write(tmp_path / "d2.png", *FACE)

    got = resolve_local_cutouts(tmp_path)

    assert got["fru"] == face
