"""Read the composition geometry the server embeds in cutout PNGs.

The server's sketch service writes one text chunk, keyword
``am-sketch-geometry``, holding JSON: the canvas the cutout was cut from and,
for FRU, the face landmarks and face box — so the Pi can place cutouts exactly
as the server's own scene composition does without detecting anything itself.
See sketch-service/sketch_service/cutout_meta.py for the writer and the field
reference.

cv2 cannot read PNG text chunks, so this walks the chunk list directly (stdlib
only). Chunks are skipped by seeking, never loaded, so the pixel data is not
read twice. Anything unexpected — not a PNG, truncated, malformed JSON, no
chunk — returns None, and callers fall back to their pre-geometry behaviour.
"""

import json
import logging
import struct
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

GEOMETRY_KEYWORD = "am-sketch-geometry"
SUPPORTED_VERSION = 1

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_TEXT_CHUNK = 1024 * 1024  # geometry is ~300 bytes; refuse anything absurd


def _decode_text_chunk(ctype: bytes, data: bytes) -> tuple[str, str] | None:
    keyword, sep, rest = data.partition(b"\x00")
    if not sep:
        return None
    key = keyword.decode("latin-1")
    if ctype == b"tEXt":
        return key, rest.decode("latin-1")
    if ctype == b"zTXt":
        # rest = compression method (1 byte, 0 = zlib) + compressed text
        return key, zlib.decompress(rest[1:]).decode("latin-1")
    if ctype == b"iTXt":
        # rest = compression flag, method, language\0, translated keyword\0, text
        flag = rest[0]
        _lang, _, rest = rest[2:].partition(b"\x00")
        _tkey, _, text = rest.partition(b"\x00")
        if flag:
            text = zlib.decompress(text)
        return key, text.decode("utf-8")
    return None


def read_png_text(path: Path | str, keyword: str) -> str | None:
    """Value of the first tEXt/zTXt/iTXt chunk named ``keyword``, or None."""
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != _PNG_SIGNATURE:
                return None
            while True:
                header = fh.read(8)
                if len(header) < 8:
                    return None
                length, ctype = struct.unpack(">I4s", header)
                if ctype == b"IEND":
                    return None
                if ctype in (b"tEXt", b"zTXt", b"iTXt") and length <= _MAX_TEXT_CHUNK:
                    decoded = _decode_text_chunk(ctype, fh.read(length))
                    fh.seek(4, 1)  # CRC
                    if decoded and decoded[0] == keyword:
                        return decoded[1]
                else:
                    fh.seek(length + 4, 1)  # data + CRC
    except (OSError, ValueError, IndexError, zlib.error, struct.error) as e:
        logger.debug(f"Could not read PNG text from {path}: {e}")
        return None


def read_geometry(path: Path | str) -> dict | None:
    """The cutout's embedded composition geometry, or None if absent/unusable."""
    raw = read_png_text(path, GEOMETRY_KEYWORD)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning(f"Malformed {GEOMETRY_KEYWORD} chunk in {path} — ignoring it")
        return None
    if not isinstance(data, dict) or data.get("v") != SUPPORTED_VERSION:
        logger.warning(f"Unsupported {GEOMETRY_KEYWORD} version in {path} — ignoring it")
        return None
    return data
