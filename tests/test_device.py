"""Device identity.

The fleet is ~50 Pis imaged from one card: same hostname, same default
address. Identity therefore cannot come from either, and everything that
aggregates logs or check reports depends on this module getting it right.
"""

from wpu_client import device


def _isolate(monkeypatch, tmp_path, env=None, file_text=None, cpuinfo=None):
    """A device.py with all three sources under test control."""
    monkeypatch.delenv(device.DEVICE_ID_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(device.DEVICE_ID_ENV, env)

    id_file = tmp_path / "device-id"
    if file_text is not None:
        id_file.write_text(file_text)
    monkeypatch.setattr(device, "DEVICE_ID_FILE", id_file)

    cpu_file = tmp_path / "cpuinfo"
    if cpuinfo is not None:
        cpu_file.write_text(cpuinfo)
    monkeypatch.setattr(device, "CPUINFO", cpu_file)


PI_CPUINFO = """\
processor\t: 0
model name\t: ARMv8 Processor rev 3 (v8l)
Hardware\t: BCM2835
Revision\t: c03114
Serial\t\t: 100000003d1f9d2a
Model\t\t: Raspberry Pi 4 Model B Rev 1.4
"""


# ── precedence ──────────────────────────────────────────────────────────


def test_env_wins_over_everything(monkeypatch, tmp_path):
    """What a provisioning tool sets when it wants to name devices by where
    they stand rather than by serial."""
    _isolate(monkeypatch, tmp_path, env="mantapa-hall-03",
             file_text="from-file", cpuinfo=PI_CPUINFO)

    assert device.device_id() == "mantapa-hall-03"


def test_file_wins_over_the_serial(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, file_text="mantapa-hall-04\n", cpuinfo=PI_CPUINFO)

    assert device.device_id() == "mantapa-hall-04"


def test_serial_is_used_when_nothing_is_provisioned(monkeypatch, tmp_path):
    """The case that matters: 50 identical cards, no provisioning step, and
    each unit still labels itself distinctly."""
    _isolate(monkeypatch, tmp_path, cpuinfo=PI_CPUINFO)

    assert device.device_id() == "pi-100000003d1f9d2a"


def test_hostname_is_the_last_resort(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(device.socket, "gethostname", lambda: "a-laptop")

    assert device.device_id() == "a-laptop"


# ── each source's failure modes ─────────────────────────────────────────


def test_blank_env_is_not_an_identity(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, env="   ", cpuinfo=PI_CPUINFO)

    assert device.device_id() == "pi-100000003d1f9d2a"


def test_empty_file_is_not_an_identity(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, file_text="\n", cpuinfo=PI_CPUINFO)

    assert device.device_id() == "pi-100000003d1f9d2a"


def test_all_zero_serial_is_rejected(monkeypatch, tmp_path):
    """Some non-Pi ARM boards report a zeroed serial, which is no more of an
    identity than the shared hostname is."""
    _isolate(monkeypatch, tmp_path, cpuinfo="Serial\t\t: 0000000000000000\n")
    monkeypatch.setattr(device.socket, "gethostname", lambda: "a-board")

    assert device.device_id() == "a-board"


def test_cpuinfo_without_a_serial_falls_through(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, cpuinfo="processor\t: 0\nmodel name\t: x86\n")
    monkeypatch.setattr(device.socket, "gethostname", lambda: "a-laptop")

    assert device.device_id() == "a-laptop"


def test_missing_sources_are_not_errors(monkeypatch, tmp_path):
    """No /proc/cpuinfo, no /etc/wpu-client — the dev-machine case."""
    _isolate(monkeypatch, tmp_path)

    assert device.device_id()  # whatever it is, it did not raise


# ── the source is reported, so a fallback is visible ────────────────────


def test_source_names_the_env_var(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, env="x", cpuinfo=PI_CPUINFO)

    assert device.DEVICE_ID_ENV in device.device_id_source()


def test_source_names_the_file(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, file_text="x", cpuinfo=PI_CPUINFO)

    assert "device-id" in device.device_id_source()


def test_source_says_serial(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path, cpuinfo=PI_CPUINFO)

    assert "serial" in device.device_id_source()


def test_source_flags_the_hostname_fallback(monkeypatch, tmp_path):
    """check_identity keys its warning off this prefix."""
    _isolate(monkeypatch, tmp_path)

    assert device.device_id_source().startswith("hostname")
