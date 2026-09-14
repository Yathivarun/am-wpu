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

`setup.sh` installs the system packages, creates a `--system-site-packages`
venv (so `picamera2` and `gi` resolve from apt), installs the dependencies,
verifies the three ONNX models are present, installs the three systemd units,
enables server mode on boot, and finishes with a pre-flight check.

It installs but does not *start* anything, so the camera stays free and you
can fix the config before a kiosk goes live.

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

## Health check

Before starting anything — and any time a unit misbehaves — ask it what is
wrong:

```bash
python main.py --check
```

```
wpu-client pre-flight — kiosk-07 — base mode

  deps     OK    numpy 1.26.4, opencv 4.11.0, onnxruntime 1.29.0, ...
  system   OK    picamera2 + gtk4 import
  models   OK    3/3 present (45 MB)
  config   OK    host=10.0.0.5:8000 model=mobilenet scale_mode=fill
  scenes   OK    sau_single 6, sau_duo 4, fru_single 9, fru_duo 4
  server   FAIL  10.0.0.5:8000 unreachable (ConnectError: ...)
  camera   OK    imx708 claimable
  logs     OK    /var/log/wpu-client
  disk     OK    41203 MB free of 61055 MB

FAIL — 1 check(s) failed: server
```

It starts no services, opens the camera only when no unit is holding it, and
**exits 1 if any check fails** — so a fleet tool can assert on it directly.
`--json` emits the same result as a machine-readable object on stdout:

```bash
python main.py --check --json
python main.py --check --diagnostic   # checks the gallery instead of the server
```

Warnings never fail the run. They flag a unit that works but probably not as
you intended — most often a missing `config/config.yaml`, which leaves the
client on built-in defaults pointing at a hardcoded host.

## Services

Three systemd units are installed by `setup.sh`. They are mutually exclusive:
the two recognition modes both hold the camera, and all three drive the
fullscreen display, so starting one stops the others.

| Unit | What it runs | On boot |
|---|---|---|
| `slideshow-server.service` | Recognition + slideshow, identities from the server | **enabled** |
| `slideshow-diagnostic.service` | Recognition + slideshow, identities from the local gallery — fully offline | disabled |
| `slideshow-only.service` | Slideshow alone — no camera, no models, no server | disabled |

Only `slideshow-server` starts on boot. The other two are installed ready to
go and turned on deliberately.

`scripts/switch-mode.sh` wraps the whole lifecycle:

```bash
scripts/switch-mode.sh server        # start server mode (stops the others)
scripts/switch-mode.sh diagnostic    # start offline mode
scripts/switch-mode.sh only          # start the display alone
scripts/switch-mode.sh stop          # stop all three, release camera + display
scripts/switch-mode.sh status        # what is running, and what starts on boot
scripts/switch-mode.sh logs          # follow the running unit's log
scripts/switch-mode.sh logs diagnostic
scripts/switch-mode.sh enable only   # change which mode starts on boot
scripts/switch-mode.sh disable only
scripts/switch-mode.sh check         # pre-flight, same as main.py --check
```

`status` prints both axes, which are independent — a unit can be running now
without starting on boot, and vice versa:

```
UNIT                           ACTIVE     ON-BOOT
slideshow-server.service       active     enabled
slideshow-diagnostic.service   inactive   disabled
slideshow-only.service         inactive   disabled
```

`enable` switches which mode owns boot: it disables the other two first, so
they can never race for the camera at startup.

Underneath it is ordinary systemd, if you prefer it directly:

```bash
sudo systemctl start|stop|restart slideshow-server.service
sudo systemctl status slideshow-server.service
sudo systemctl enable|disable slideshow-server.service
journalctl -u slideshow-server.service -f
journalctl -u slideshow-server.service --since "1 hour ago"
```

### Logs

Every mode logs to stdout (captured by journald) and to
`/var/log/wpu-client/app.log`. Lines carry the hostname, so a fleet's logs
stay legible once aggregated:

```
2026-09-04 16:11:35 - kiosk-07 - wpu_client.services... - INFO - Local match: Varun
```

If the log directory is missing or not writable the client still starts and
logs to stdout only — `main.py --check` reports it under `logs`.

## Running by hand

Stop the services first; whichever is active holds the camera.

```bash
scripts/switch-mode.sh stop

python main.py                      # both services, server mode
python main.py --diagnostic         # offline mode, no server
python main.py --service slideshow  # slideshow only
python main.py --log-level DEBUG    # verbose
python main.py --config path.yaml   # alternate config
```

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
are the image filename stems. After adding, removing or renaming a background,
regenerate the derived `name` fields:

```bash
python scripts/sync_scene_names.py
```

See `data/base_scenes/README.md` for the config shapes, the gender rules and
how to tune anchors.

## Troubleshooting

| Symptom | Check |
|---|---|
| Stock images only, never a visitor | `onnxruntime` imported cleanly at startup; a failed import aborts the recognition thread |
| `NumPy 1.x cannot be run in NumPy 2.x` | `pip install --force-reinstall "numpy==1.26.4"` — picamera2 needs the 1.x ABI |
| Camera busy / cannot be opened | Another unit holds it: `scripts/switch-mode.sh stop` |
| Recognised, but no slides | No cutouts came back, or no scene matched their gender — run with `--log-level DEBUG` and look for `compose` lines |
| Nobody is ever recognised | `model` must be `mobilenet`; `sface` queries an empty gallery |
| Image is stretched or cropped wrong | `scale_mode` doesn't match the panel |
| Unit won't start, no useful error | `journalctl -u slideshow-server.service -n 50` |
| Wrong mode came up after a reboot | `scripts/switch-mode.sh status` — check the ON-BOOT column |

Start with `python main.py --check`; it covers most of the table above.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## Layout

```
main.py                       entry point, --check, starts/supervises services
wpu_client/health.py          pre-flight checks behind main.py --check
config/                       config.yaml + annotated example
systemd/                      unit templates for the three service modes
models/                       YuNet detector, MobileFaceNet + SFace embedders
data/base_scenes/             scene backgrounds + placement configs
data/stock_images/            idle slideshow content
data/embeddings/              diagnostic gallery (seeded people)
wpu_client/services/
  face_recognition/           detect, embed, identify, fetch, compose
  slideshow/                  GTK4 display
scripts/                      setup, mode switch, seeding, sandbox test
```
