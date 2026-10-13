"""Display test patterns, shown through the real slideshow (slideshow-only unit).

Authored at the production panel's physical 6:7 aspect, like the scene art:
with scale_mode `fill` a correctly set-up panel shows the circle round and the
squares square. A 16:9 reference circle is included that *should* look
squashed — if it looks round instead, the panel/scale_mode pairing is wrong.
"""

from pathlib import Path

import cv2
import numpy as np

W, H = 1200, 1400  # 6:7
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (128, 128, 128)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, text, y, scale=1.6, color=WHITE, thick=3):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


def geometry() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    for x in range(0, W + 1, 100):
        cv2.line(img, (x, 0), (x, H), (60, 60, 60), 1)
    for y in range(0, H + 1, 100):
        cv2.line(img, (0, y), (W, y), (60, 60, 60), 1)
    cv2.circle(img, (W // 2, H // 2), 500, WHITE, 6, cv2.LINE_AA)
    cv2.rectangle(img, (W // 2 - 200, H // 2 - 200), (W // 2 + 200, H // 2 + 200), (0, 200, 255), 5)
    _text(img, "1 / 6  GEOMETRY", 90)
    _text(img, "circle must look ROUND, box SQUARE", H - 80, 1.2)
    return img


def reference_16x9() -> np.ndarray:
    img = np.full((1080, 1920, 3), 30, np.uint8)
    cv2.circle(img, (960, 540), 420, (0, 200, 255), 6, cv2.LINE_AA)
    _text(img, "2 / 6  16:9 REFERENCE", 90)
    _text(img, "this circle SHOULD look squashed on the 6:7 panel", 1020, 1.2)
    return img


def ramps() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    steps = 16
    bw = W // steps
    for i in range(steps):
        v = int(i * 255 / (steps - 1))
        img[150:550, i * bw : (i + 1) * bw] = (v, v, v)
        cv2.putText(img, str(v), (i * bw + 8, 590), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    bars = [
        (255, 255, 255),
        (0, 255, 255),
        (255, 255, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 0, 255),
        (255, 0, 0),
        (0, 0, 0),
    ]
    cw = W // len(bars)
    for i, c in enumerate(bars):
        img[700:1150, i * cw : (i + 1) * cw] = c
    _text(img, "3 / 6  GREY + COLOUR", 90)
    _text(img, "all 16 grey steps distinct? dark steps visible?", H - 80, 1.1)
    return img


def text_sizes() -> np.ndarray:
    img = np.full((H, W, 3), 245, np.uint8)
    y = 160
    for px in (16, 24, 32, 48, 64, 96, 128):
        scale = px / 22.0
        cv2.putText(img, f"{px}px Welcome Varun", (40, y), FONT, scale, BLACK, max(1, px // 16), cv2.LINE_AA)
        y += px + 70
    _text(img, "4 / 6  READABILITY", 80, 1.6, BLACK)
    _text(img, "note smallest size readable from the visitor spot", H - 60, 1.0, BLACK, 2)
    return img


def edges() -> np.ndarray:
    img = np.full((H, W, 3), 40, np.uint8)
    for inset, color in ((0, (0, 0, 255)), (10, (0, 255, 0)), (25, (255, 0, 0)), (50, WHITE)):
        cv2.rectangle(img, (inset, inset), (W - 1 - inset, H - 1 - inset), color, 3)
    _text(img, "5 / 6  EDGES / OVERSCAN", 200)
    _text(img, "red=0 green=10 blue=25 white=50 px", 280, 1.1)
    _text(img, "note the outermost line fully visible", 350, 1.1)
    return img


def uniformity() -> np.ndarray:
    img = np.full((H, W, 3), 255, np.uint8)
    _text(img, "6 / 6  FULL WHITE — even brightness? dead pixels?", H // 2, 1.0, BLACK, 2)
    return img


PATTERNS = [
    ("01_geometry", geometry),
    ("02_reference_16x9", reference_16x9),
    ("03_ramps", ramps),
    ("04_text", text_sizes),
    ("05_edges", edges),
    ("06_white", uniformity),
]


def write_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in PATTERNS:
        path = out / f"{name}.png"
        cv2.imwrite(str(path), fn())
        paths.append(path)
    return paths
