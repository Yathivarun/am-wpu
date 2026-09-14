"""Pre-flight checks.

What matters here is the contract a fleet depends on: the exit code, the JSON
shape, and that a check reports rather than raises. The checks that need a
camera or a server are exercised through their failure paths only — the
success paths are on-device territory.
"""

import json

import pytest

from wpu_client import health
from wpu_client.config.settings import Settings
from wpu_client.health import FAIL, OK, WARN, Result


def _results(*pairs):
    return [Result(name, status, "detail") for name, status in pairs]


# ── exit code: the fleet contract ───────────────────────────────────────


def test_exit_code_is_zero_when_everything_passes():
    assert health.exit_code(_results(("deps", OK), ("models", OK))) == 0


def test_warnings_never_fail_a_run():
    """A Pi on built-in defaults still works. A fleet that wants to treat
    warnings as failures reads them out of the JSON."""
    assert health.exit_code(_results(("config", WARN), ("scenes", WARN))) == 0


def test_a_single_failure_fails_the_run():
    assert health.exit_code(_results(("deps", OK), ("camera", FAIL))) == 1


# ── JSON: what Ansible parses ───────────────────────────────────────────


def test_json_is_parseable_and_names_what_failed():
    results = _results(("deps", OK), ("camera", FAIL), ("config", WARN))

    payload = json.loads(health.render_json(results, diagnostic=False))

    assert payload["status"] == FAIL
    assert payload["failed"] == ["camera"]
    assert payload["warned"] == ["config"]
    assert payload["mode"] == "base"
    assert len(payload["checks"]) == 3
    assert payload["checks"][0] == {"name": "deps", "status": OK, "detail": "detail"}


def test_json_reports_ok_when_only_warnings():
    payload = json.loads(health.render_json(_results(("config", WARN)), diagnostic=True))

    assert payload["status"] == OK
    assert payload["warned"] == ["config"]
    assert payload["mode"] == "diagnostic"


# ── text rendering ──────────────────────────────────────────────────────


def test_text_names_the_failed_checks():
    text = health.render_text(_results(("deps", FAIL), ("camera", FAIL)), diagnostic=False)

    assert "FAIL — 2 check(s) failed: deps, camera" in text


def test_text_calls_out_warnings_without_declaring_failure():
    text = health.render_text(_results(("config", WARN)), diagnostic=False)

    assert "OK with warnings" in text
    assert "FAIL" not in text


def test_text_says_ready_when_clean():
    assert "ready to run" in health.render_text(_results(("deps", OK)), diagnostic=False)


# ── models ──────────────────────────────────────────────────────────────


def test_models_check_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "MODELS_DIR", tmp_path)
    (tmp_path / "mobilefacenet.onnx").write_bytes(b"x")

    result = health.check_models()

    assert result.status == FAIL
    assert "face_detection_yunet_2023mar.onnx" in result.detail
    assert "mobilefacenet.onnx" not in result.detail


def test_models_check_passes_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "MODELS_DIR", tmp_path)
    for name in health.REQUIRED_MODELS:
        (tmp_path / name).write_bytes(b"x")

    assert health.check_models().status == OK


# ── scenes ──────────────────────────────────────────────────────────────


def _scene_tree(root, counts, config=None):
    for name in health.SCENE_DIRS:
        scene_dir = root / "base_scenes" / name
        scene_dir.mkdir(parents=True)
        for i in range(counts.get(name, 1)):
            (scene_dir / f"{i + 1}.png").write_bytes(b"x")
        (scene_dir / "scenes_config.json").write_text(
            json.dumps(config if config is not None else {"1": {}})
        )


def test_scenes_check_fails_when_the_tree_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)

    assert health.check_scenes().status == FAIL


def test_scenes_check_fails_when_no_art_at_all(tmp_path, monkeypatch):
    """The state a fresh clone lands in once the images are untracked — the
    kiosk runs but composes nothing, which is exactly what must not pass."""
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    _scene_tree(tmp_path, dict.fromkeys(health.SCENE_DIRS, 0))

    result = health.check_scenes()

    assert result.status == FAIL
    assert "no scene art at all" in result.detail


def test_scenes_check_fails_when_one_directory_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    _scene_tree(tmp_path, {"fru_duo": 0})

    result = health.check_scenes()

    assert result.status == FAIL
    assert "fru_duo" in result.detail


def test_scenes_check_warns_when_a_config_key_has_no_image(tmp_path, monkeypatch):
    """A normal state — a key added before its art arrives — worth surfacing
    but not worth failing a deploy over."""
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    _scene_tree(tmp_path, {}, config={"1": {}, "99": {}})

    result = health.check_scenes()

    assert result.status == WARN
    assert "4 config key(s) have no image" in result.detail


def test_scenes_check_passes_on_a_complete_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    _scene_tree(tmp_path, {})

    assert health.check_scenes().status == OK


def test_scenes_check_fails_loudly_on_unreadable_config(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    _scene_tree(tmp_path, {})
    (tmp_path / "base_scenes" / "fru_duo" / "scenes_config.json").write_text("{not json")

    assert health.check_scenes().status == FAIL


# ── config ──────────────────────────────────────────────────────────────


def test_config_check_warns_when_running_on_built_in_defaults(tmp_path):
    """The fleet hazard: a unit whose config never deployed comes up pointing
    at the hardcoded default host and otherwise looks healthy."""
    result = health.check_config(Settings(), tmp_path / "absent.yaml")

    assert result.status == WARN
    assert "built-in defaults" in result.detail


def test_config_check_warns_on_the_empty_gallery_model(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    settings = Settings()
    settings.services.face_recognition.model = "sface"

    result = health.check_config(settings, config)

    assert result.status == WARN
    assert "matches nobody" in result.detail


def test_config_check_reports_the_host_it_will_talk_to(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    settings = Settings()
    settings.services.face_recognition.api_endpoint = "http://10.0.0.5:8000/api/v1/identify/"

    result = health.check_config(settings, config)

    assert result.status == OK
    assert "10.0.0.5:8000" in result.detail


# ── log dir and disk ────────────────────────────────────────────────────


def test_log_dir_check_fails_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "LOG_DIR", tmp_path / "absent")

    result = health.check_log_dir()

    assert result.status == FAIL
    assert "mkdir" in result.detail


def test_log_dir_check_passes_when_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "LOG_DIR", tmp_path)

    assert health.check_log_dir().status == OK


def test_disk_check_fails_below_the_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    monkeypatch.setattr(health, "MIN_FREE_BYTES", 1 << 62)

    assert health.check_disk().status == FAIL


# ── run_checks: mode decides which checks apply ─────────────────────────


@pytest.mark.parametrize(
    "diagnostic,present,absent",
    [(True, "gallery", "server"), (False, "server", "gallery")],
)
def test_run_checks_picks_the_checks_that_apply_to_this_mode(
    diagnostic, present, absent, tmp_path, monkeypatch
):
    """Probing a server on an offline unit, or a gallery on a base unit,
    reports a failure that does not matter there."""
    monkeypatch.setattr(health, "check_server", lambda s: Result("server", OK, ""))
    monkeypatch.setattr(health, "check_gallery", lambda s: Result("gallery", OK, ""))
    monkeypatch.setattr(health, "check_camera", lambda: Result("camera", OK, ""))
    monkeypatch.setattr(health, "check_picamera", lambda: Result("system", OK, ""))

    names = [
        r.name for r in health.run_checks(Settings(), diagnostic, tmp_path / "config.yaml")
    ]

    assert present in names
    assert absent not in names
