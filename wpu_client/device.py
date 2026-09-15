"""Which Pi this is.

Every unit in the fleet is imaged from the same card: same hostname, same
default address, same everything. That is fine until logs from 50 of them
arrive in one place, at which point `hostname` is not an identity — it is the
same string 50 times, and the stream it labels is unreadable.

So identity is taken from the first of these that yields something:

1. `WPU_DEVICE_ID` in the environment, or a bare id in /etc/wpu-client/device-id.
   What a provisioning tool sets when it wants to name devices itself
   ("mantapa-hall-03") rather than accept a hardware serial.
2. The SoC serial from /proc/cpuinfo. Unique per board, stable across
   reimaging, and needs no provisioning step at all — which is what makes a
   pile of identical cards work out of the box.
3. The hostname. Only reached off-Pi (a dev laptop, CI), where it is unique
   enough and nobody is aggregating anything.

The result is what goes in every log line and in `--check --json`, so it is
also the join key between a Grafana panel and a `switch-mode.sh` session.
"""

import os
import socket
from pathlib import Path

DEVICE_ID_ENV = "WPU_DEVICE_ID"
DEVICE_ID_FILE = Path("/etc/wpu-client/device-id")
CPUINFO = Path("/proc/cpuinfo")

# A Pi's serial is 16 hex digits, historically zero-padded to the point that
# only the tail carries information. The full value is kept anyway: it is what
# `vcgencmd otp_dump` and the sticker on the board agree on.
_SERIAL_KEY = "serial"


def _from_env() -> str | None:
    value = (os.environ.get(DEVICE_ID_ENV) or "").strip()
    return value or None


def _from_file() -> str | None:
    try:
        value = DEVICE_ID_FILE.read_text().strip()
    except OSError:
        return None
    return value or None


def _from_cpuinfo() -> str | None:
    """The SoC serial, on a Pi. None anywhere else."""
    try:
        text = CPUINFO.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == _SERIAL_KEY:
            serial = value.strip()
            # Non-Pi ARM boards report all zeroes here, which is no more of an
            # identity than the shared hostname is.
            if serial and set(serial) != {"0"}:
                return f"pi-{serial}"
    return None


def device_id() -> str:
    """A label for this unit that is unique within the fleet."""
    for source in (_from_env, _from_file, _from_cpuinfo):
        value = source()
        if value:
            return value
    return socket.gethostname()


def device_id_source() -> str:
    """Where device_id() got its answer — reported by `--check` so a unit
    running on the hostname fallback is visible before its logs are."""
    if _from_env():
        return f"${DEVICE_ID_ENV}"
    if _from_file():
        return str(DEVICE_ID_FILE)
    if _from_cpuinfo():
        return f"{CPUINFO} serial"
    return "hostname (no serial, no override)"
