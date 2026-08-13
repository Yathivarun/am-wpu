import yaml

from wpu_client.config.settings import Settings

# NOTE: every test below passes an explicit config_path (always under
# tmp_path). Settings.load(None) resolves to the real CONFIG_DIR/config.yaml
# and writes a default file there if it's missing — never call it bare from a
# test, or it pollutes the actual repo checkout.


def test_load_missing_file_falls_back_to_defaults_and_writes_one(tmp_path):
    config_path = tmp_path / "config.yaml"
    assert not config_path.exists()

    settings = Settings.load(config_path)

    assert settings.log_level == "INFO"
    assert settings.services.slideshow.enabled is True
    assert settings.services.face_recognition.model == "sface"
    # create_default_config runs as a side effect of a missing file.
    assert config_path.exists()


def test_load_reads_values_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "log_level": "DEBUG",
                "services": {
                    "slideshow": {"advance_time": 7, "sort_mode": "numeric"},
                    "face_recognition": {"model": "mobilenet", "camera_id": 2},
                },
            }
        )
    )

    settings = Settings.load(config_path)

    assert settings.log_level == "DEBUG"
    assert settings.services.slideshow.advance_time == 7
    assert settings.services.slideshow.sort_mode == "numeric"
    assert settings.services.face_recognition.model == "mobilenet"
    assert settings.services.face_recognition.camera_id == 2
    # Fields not present in the YAML still fall back to their defaults.
    assert settings.services.slideshow.full_screen is True


def test_load_with_invalid_yaml_falls_back_to_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("services: [unterminated")

    settings = Settings.load(config_path)

    assert settings.log_level == "INFO"


def test_create_default_config_is_valid_yaml(tmp_path):
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()

    Settings.create_default_config(config_path)

    data = yaml.safe_load(config_path.read_text())
    assert data["services"]["face_recognition"]["model"] == "sface"


def test_get_service_config_returns_matching_section(tmp_path):
    settings = Settings.load(tmp_path / "config.yaml")

    assert settings.get_service_config("slideshow") is settings.services.slideshow
    assert settings.get_service_config("face_recognition") is settings.services.face_recognition
    assert settings.get_service_config("nonexistent") is None


def test_slideshow_default_sort_mode_is_alphabetical():
    # Regression guard: this was "numeric" until it crashed on mixed-name
    # directories (README changelog, 2026-07-11) — the safe default matters.
    settings = Settings()
    assert settings.services.slideshow.sort_mode == "alphabetical"


def test_base_mode_composition_defaults():
    """Base mode must be on by default, with the cutout names the server
    actually uploads today (<registration_id>/wpu/<name>)."""
    face = Settings().services.face_recognition

    assert face.sau_cutout_filename == "sau_cutout.png"
    assert face.fru_cutout_filename == "fru_cutout.png"
    assert face.video_filenames == ["video_1.mov", "video_2.mov", "video_3.mov"]
    # The rollback valve defaults off, i.e. local composition is the live path.
    assert face.use_legacy_final_images is False
    assert face.base_assets_max_bytes == 5 * 1024 * 1024 * 1024


def test_base_mode_fields_are_configurable(tmp_path):
    """Filenames and endpoint are placeholders pending backend confirmation,
    so they must be overridable from config.yaml."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "services": {
                    "face_recognition": {
                        "sau_cutout_filename": "body.png",
                        "fru_cutout_filename": "face.png",
                        "video_filenames": ["a.mp4"],
                        "wpu_videos_endpoint": "http://server/v2/videos",
                        "use_legacy_final_images": True,
                        "base_assets_max_bytes": 1024,
                    }
                }
            }
        )
    )

    face = Settings.load(config_path).services.face_recognition
    assert face.sau_cutout_filename == "body.png"
    assert face.fru_cutout_filename == "face.png"
    assert face.video_filenames == ["a.mp4"]
    assert face.wpu_videos_endpoint == "http://server/v2/videos"
    assert face.use_legacy_final_images is True
    assert face.base_assets_max_bytes == 1024


def test_example_config_matches_the_settings_schema():
    """config/config.yaml.example is the documented template — it must stay
    loadable as-is, so a documented key can't drift from the model."""
    with open("config/config.yaml.example") as f:
        settings = Settings(**yaml.safe_load(f))

    face = settings.services.face_recognition
    assert face.sau_cutout_filename == "sau_cutout.png"
    assert face.use_legacy_final_images is False
