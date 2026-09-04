# WPU Client

A Raspberry Pi kiosk that recognises visitors and shows them personalised
slides. It runs a fullscreen GTK4 slideshow and, alongside it, an on-device
face recognition loop. When someone is recognised, the slideshow switches from
the stock images to that person's own slides; when they leave, it switches back.

## How it works

```
camera ──► YuNet detect ──► MobileFaceNet embed (512-D)
                                     │
                                     ▼
                        server /identify  (or local gallery
                                     │     in diagnostic mode)
                                     ▼
                        fetch SAU + FRU cutouts
                                     │
                                     ▼
                    compose onto data/base_scenes/ backgrounds
                                     │
                                     ▼
                              slideshow displays
```

Every visitor is stored as two transparent PNG cutouts — **SAU** (body) and
**FRU** (face) — not as finished pictures. The Pi composes the finished slides
itself, pasting those cutouts onto local scene backgrounds. One person in
frame uses the `*_single` scenes; two recognised people together use the
`*_duo` scenes.

The two cutouts are placed by different mechanics and are never mixed into the
same slide:

- **SAU** — scaled and dropped onto a bottom-centre anchor (the anchor is where
  the feet land).
- **FRU** — warped onto an eye anchor (scale + rotate + translate).

## Requirements

- Raspberry Pi with a camera (Pi OS Bookworm or later)
- Python 3.11+
- System packages: `python3-picamera2`, `python3-gi`, GTK 4, GStreamer
- A reachable registration server for base mode

## Installation

```bash
git clone <repo> && cd am-wpu-client
./scripts/setup.sh
```

`setup.sh` creates a `--system-site-packages` venv (so `picamera2` and `gi`
resolve from apt), installs the dependencies, verifies the three ONNX models
are present, and prints the config keys worth checking.

Then create a config and point it at your server:

```bash
cp config/config.yaml.example config/config.yaml
```

## Configuration

`config/config.yaml`. The full annotated template is
`config/config.yaml.example`. Three keys matter per device:

| Key | What to set it to |
|---|---|
| `api_endpoint`, `wpu_endpoint`, `sau_media_endpoint` | Your server's host — all three point at the same machine |
| `model` | `mobilenet` (MobileFaceNet, 512-D). This is the only gallery the server enrols into |
| `scale_mode` | `fill` for the 6:7 production panel, `fit` for an ordinary monitor |

`model: sface` selects the 128-D embedder and queries the server's `sface`
gallery, which is empty in production — it will recognise nobody. It exists
for local diagnostic use.

## Running

```bash
python main.py                      # both services
python main.py --diagnostic         # offline mode, no server
python main.py --log-level DEBUG    # verbose
python main.py --service slideshow  # slideshow only
```

Installed as systemd units, use the mode switcher:

```bash
scripts/switch-mode.sh server       # base mode
scripts/switch-mode.sh diagnostic   # offline mode
scripts/switch-mode.sh stop         # stop both, release the camera
scripts/switch-mode.sh status
```

Only one unit may run at a time — whichever is active holds the camera
exclusively. Stop both before running `main.py` or `seed_face.py` by hand.

## Modes

**Base mode** (default) matches faces against the server and fetches each
person's cutouts from it.

**Diagnostic mode** (`--diagnostic`) is fully offline. Faces are matched
against a local gallery under `data/embeddings/`, seeded with:

```bash
python scripts/seed_face.py --name "Varun" --face varun.jpg
```

Each person's two cutouts go in `data/embeddings/<slug>/sketches/`. Which file
is the body and which is the face is worked out from `meta.json`, then the
filename, then the image's proportions — so `person.png` + `d6.png` resolve
correctly as they are. To pin it explicitly, add to `meta.json`:

```json
{ "name": "Varun", "gender": "male", "sau": "person.png", "fru": "d2.png" }
```

Everything after the match — composition, caching, display — is the same code
as base mode, onto the same `data/base_scenes/` backgrounds. Only the source
of the identity and the cutouts differs.

## Scenes

Backgrounds and their placement configs live in `data/base_scenes/`:

| Directory | Composes |
|---|---|
| `sau_single/` | one body |
| `sau_duo/` | two bodies |
| `fru_single/` | one face |
| `fru_duo/` | two faces |

Each directory has numbered image files and a `scenes_config.json` whose keys
are the image filename stems. See `data/base_scenes/README.md` for the config
shapes and how to tune anchors.

## Troubleshooting

| Symptom | Check |
|---|---|
| Stock images only, never a visitor | `onnxruntime` imported cleanly at startup; a failed import aborts the recognition thread |
| `NumPy 1.x cannot be run in NumPy 2.x` | `pip install --force-reinstall "numpy==1.26.4"` — picamera2 needs the 1.x ABI |
| Camera busy / cannot be opened | Another unit holds it: `scripts/switch-mode.sh stop` |
| Recognised, but no slides | No cutouts came back, or no scene matched their gender — run with `--log-level DEBUG` and look for `compose` lines |
| Nobody is ever recognised | `model` must be `mobilenet`; `sface` queries an empty gallery |
| Image is stretched or cropped wrong | `scale_mode` doesn't match the panel |

Logs go to stdout and `/var/log/wpu-client/app.log`.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## Layout

```
main.py                       entry point, starts and supervises services
config/                       config.yaml + annotated example
models/                       YuNet detector, MobileFaceNet + SFace embedders
data/base_scenes/             scene backgrounds + placement configs
data/stock_images/            idle slideshow content
data/embeddings/              diagnostic gallery (seeded people)
wpu_client/services/
  face_recognition/           detect, embed, identify, fetch, compose
  slideshow/                  GTK4 display
scripts/                      setup, mode switch, seeding, sandbox test
```
