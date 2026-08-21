# WPU Client

Face-recognition kiosk for a Raspberry Pi. A full-screen slideshow runs continuously;
when a registered visitor steps in front of the camera, the display switches to slides
personalised for that visitor, then returns to the stock loop once they leave.

All face detection and embedding happens **on-device**. The registration server is
only ever asked "whose vector is this?" — it never receives an image.

---

## Contents

- [How it works](#how-it-works)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the client](#running-the-client)
- [Operating modes](#operating-modes)
- [What gets displayed](#what-gets-displayed)
- [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting)
- [Server API contract](#server-api-contract)
- [Data, caches and disk usage](#data-caches-and-disk-usage)
- [Development](#development)
- [Project structure](#project-structure)
- [Release history](#release-history)

---

## How it works

Two services run side by side in one process, communicating over an in-process event
bus. Neither imports the other.

```
  Pi camera (imx219)
        │  1640x1232 full-FOV frames, every detection_interval seconds
        ▼
  ┌─────────────────────────────────────────────┐
  │ FaceRecognitionService  (worker thread)     │
  │                                             │
  │  YuNet            detect + 5 landmarks      │
  │  MobileFaceNet    align → 512-D embedding   │
  │  local cache      already-tracked person?   │──── yes ──┐
  │  POST /identify   server names the vector   │           │
  │  asset fetch      cutouts + video for them  │           │
  │  compose          slides, on-device         │           │
  └─────────────────────────────────────────────┘           │
        │  person.detected / person.left / faces.recognized  │
        ▼                                                    │
  ┌─────────────────────────────────────────────┐            │
  │ SlideshowService  (GTK4, main thread)       │◀───────────┘
  │  stock loop  ⇄  visitor slides + overlay    │
  └─────────────────────────────────────────────┘
        │
        ▼
     Display
```

**The recognition cycle, step by step:**

1. A frame is captured every `detection_interval` seconds.
2. **YuNet** locates every face and returns a bounding box plus five landmarks.
   The largest `MAX_FACES_PER_FRAME` (2) faces are processed.
3. Each face is aligned to the ArcFace template and embedded by **MobileFaceNet**
   into a 512-D L2-normalised vector.
4. The vector is compared against the vectors of everyone already being tracked
   (cosine distance < 0.4). A hit means "same person, still here" — no network call.
5. A miss goes to the **server** (`POST /identify`) — or, in diagnostic mode, to the
   local seeded gallery. The server replies with a name and `registration_id`.
6. That visitor's **SAU (body) and FRU (face) alpha cutouts** and their **SAU video**
   are fetched and cached, then composed on-device onto the scene backgrounds in
   `data/base_scenes/`.
7. `person.detected` fires and the slideshow switches to those slides. When the face
   has been gone for `person_timeout` seconds, `person.left` fires and the stock loop
   resumes.

Because detection and embedding are local, a server outage degrades the kiosk to a
plain stock slideshow rather than breaking it.

---

## Features

**Display**

- Full-screen GTK4 slideshow, configurable advance interval and sort order.
- Mixed image and video slides. Videos play through a manual GStreamer pipeline,
  always muted, and advance on end-of-stream (with a 10-second fallback if the
  duration can't be read).
- **Switchable display scaling** for the production 6:7 panel or an ordinary test
  monitor — one config key, see [Display scaling](#display-scaling).
- On-screen recognition overlay (one line per visible face) and a live camera
  preview thumbnail, both toggleable.

**Recognition**

- On-device YuNet detection and MobileFaceNet (512-D) embedding; no image ever
  leaves the Pi.
- **Multi-face tracking** — every visible face gets its own tracking entry, overlay
  line and dataset counter.
- Local cosine cache so a visitor standing still doesn't re-query the server on
  every frame.
- **Two-person (duo) display** — when exactly two known people are tracked together,
  they are composed into shared scenes.
- **Offline diagnostic mode** — match against a locally seeded gallery with no
  server at all.

**Assets**

- **On-device composition**: the server stores one cutout pair per person and the Pi
  composes person × scene on demand, rather than the server pre-rendering every
  combination.
- **ETag/Last-Modified revalidation** — re-photographing a visitor overwrites the
  same object key server-side, and the client notices and rebuilds their slides.
- **LRU disk budget** with a size cap; duo pairs are evicted first because they grow
  quadratically and rebuild cheaply from the retained cutouts.
- Best-effort by design: a missing cutout, an absent video or an unreachable server
  is logged and skipped, never raised.

**Operations**

- Two systemd units (server / diagnostic) that `Conflicts=` each other, so they can
  never fight over the camera.
- Idempotent `setup.sh` — safe to re-run for updates.
- Per-visit CSV log and optional dataset frame capture.
- Rollback valve to the previous server-composed-image behaviour, one config flip.

---

## Requirements

| | |
|---|---|
| Hardware | Raspberry Pi 4 (2 GB+), imx219 camera module, display |
| OS | Raspberry Pi OS / Debian, 64-bit |
| Python | 3.11+ |
| Display stack | GTK4, GStreamer |
| Disk | ~500 MB for the app and models, plus the asset cache (default cap 5 GB) |

Pinned and **not** to be bumped casually: `numpy==1.26.4` and
`opencv-python==4.11.0.86`. The Pi's `picamera2` is built against the NumPy 1.x C
ABI, and OpenCV 4.12+ requires NumPy ≥ 2 — the two move together or not at all.
See the comment in `pyproject.toml` before changing either.

---

## Installation

### Option A — fresh Raspberry Pi, from a release zip

```bash
unzip wpu-client-vX.Y.Z.zip -d am-wpu-client
cd am-wpu-client
chmod +rx scripts/setup.sh      # see note below
./scripts/setup.sh
```

`setup.sh` is idempotent and safe to re-run on an already-provisioned device. It runs
eight steps:

| Step | What it does |
|---|---|
| `[0/8]` | Normalises file permissions across the unpacked tree |
| `[1/8]` | Installs apt system packages (GTK4, Picamera2, GStreamer, `libcap-dev`) |
| `[2/8]` | Creates a `--system-site-packages` venv and installs Python deps |
| `[3/8]` | Creates `data/`, `models/`, `config/` and `/var/log/wpu-client` |
| `[4/8]` | Verifies all three `.onnx` models are present — **fails hard if not** |
| `[5/8]` | Seeds `config/config.yaml` from the template, echoes key settings |
| `[6/8]` | Installs both systemd units with this device's paths substituted in |
| `[7/8]` | Leaves both units **disabled and stopped** |
| `[8/8]` | Prints how to start a mode |

**Nothing starts automatically.** Whichever service runs holds the camera
exclusively, and an auto-started one would block `seed_face.py`, the benchmarks,
`rpicam-*` and any foreground `main.py` run — surfacing as a confusing "camera
unavailable". Start the mode you want, when you want it.

> **Note on the `chmod`:** `unzip` applies the current umask, and an unusual umask
> can land scripts as mode `111` — executable but not *readable*, which makes
> `./scripts/setup.sh` fail with "Permission denied" because bash must *read* a
> script. The `chmod` unsticks `setup.sh` itself; step `[0/8]` fixes the rest.

**After setup, before the first run**, open `config/config.yaml` and check the three
things setup echoes back: the endpoint host, `model`, and `scale_mode`. See
[Configuration](#configuration).

### Option B — from a git checkout

```bash
git clone <repo> && cd am-wpu-client
UV_HTTP_TIMEOUT=900 uv sync --frozen
cp config/config.yaml.example config/config.yaml
```

- `--frozen` installs exactly what `uv.lock` pins. Without it uv is free to
  re-resolve and has pulled NumPy 2.x back in, breaking `picamera2` on import.
- `UV_HTTP_TIMEOUT=900` matters on a slow link: the OpenCV wheel is ~42 MB and uv's
  30-second default expires mid-download, failing the whole sync after three retries.

If a `git pull` aborts because `uv.lock` is locally modified, stash it rather than
merging — the tracked lock is the source of truth:

```bash
git stash push -m "pi-local uv.lock" uv.lock && git pull && uv sync --frozen
```

### System dependencies

Installed by `setup.sh` step `[1/8]`; listed here for manual installs. There is **no
`dlib` dependency**, so no `cmake`/`g++`/`libboost` toolchain is needed.

```bash
sudo apt install -y \
    python3 python3-venv python3-dev \
    libgtk-4-1 gir1.2-gtk-4.0 python3-gi python3-gi-cairo \
    python3-picamera2 \
    libgl1 libglib2.0-0 libcap-dev \
    gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

`libcap-dev` is required to build `python-prctl` (a `picamera2` dependency); without
it the Python install fails with "You need to install libcap development headers".

Video slides are decoded with a manual GStreamer pipeline
(`filesrc ! decodebin ! videoconvert ! appsink`) rather than `Gtk.MediaFile`, because
the `gtk4paintablesink` that backend needs isn't packaged on all Raspberry Pi OS
releases — and without it video plays silently but never paints a frame.

### Models

Three ONNX files ship in `models/`, all tracked in git and bundled in release zips:

| File | Size | Role |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 0.2 MB | Face detection + 5 landmarks. Always used. |
| `mobilefacenet.onnx` | 8.4 MB | **Production** recogniser, 512-D → server `auraface` gallery |
| `face_recognition_sface_2021dec.onnx` | 39 MB | Alternative recogniser, 128-D → server `sface` gallery |

---

## Configuration

`config/config.yaml` is **per-device**: it is gitignored, is not shipped in release
zips, and is seeded from the tracked `config/config.yaml.example`. That separation
exists because tracking it once shipped one developer's cluster IP and model choice
to every Pi in the fleet. Edit `config.yaml` freely; edit `config.yaml.example` only
when changing what a *new* device should get.

If no config file exists at all, the client writes one from its built-in defaults and
carries on.

### The three settings to check on every device

**1. Endpoint host** — all three endpoints must point at the same server.

```yaml
api_endpoint:       "http://192.168.1.19:8000/api/v1/identify/"
wpu_endpoint:       "http://192.168.1.19:8000/api/v1/wpu/images"
sau_media_endpoint: "http://192.168.1.19:8000/api/v1/sau/media"
```

Change the host in all three together. A half-updated set is how the video route once
ended up on `localhost` while everything else talked to the cluster. Note the port:
`:8000` is the API; port 80 does not answer.

**2. Recognition model** — this picks *both* the on-device embedder and the server
gallery queried. The two galleries are disjoint vector spaces, so getting this wrong
does not lower accuracy, it **matches nobody**.

```yaml
model: "mobilenet"    # correct for the live cluster
```

| Value | Embedder | Server gallery | Status |
|---|---|---|---|
| `mobilenet` | MobileFaceNet 512-D | `auraface` | **Use this.** Where registrations are enrolled. |
| `sface` | OpenCV SFace 128-D | `sface` | Lighter on CPU, but that gallery is empty upstream. Diagnostic use only. |

**3. Display scaling** — see below.

### Display scaling

`slideshow.scale_mode` is the only setting that decides on-screen shape, and it
applies identically to images and videos.

| Value | Behaviour | Use for |
|---|---|---|
| `fill` | Stretch to the window, aspect ignored | **Production 6:7 panel** |
| `fit` | Letterbox, aspect preserved | **Ordinary test monitor** |
| `crop` | Fill the window and crop the overflow, aspect preserved | Alternative for either |

The production display advertises a 1920×1080 framebuffer but is physically a 6:7
panel, so the hardware squeezes everything horizontally. `fill` stretches each slide
non-uniformly, and that stretch is exactly what the panel's squeeze cancels out. On a
16:9 monitor there is no squeeze to cancel, which is why `fill` looks stretched
there — switch to `fit` for testing and back to `fill` before shipping. No other
change is needed in either direction.

### Full reference

**`services.slideshow`**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Run the slideshow service |
| `full_screen` | `true` | Start fullscreen |
| `advance_time` | `3` | Seconds per image slide (videos advance on end) |
| `image_directory` | `data/stock_images` | Stock loop source |
| `image_extensions` | png/jpg/jpeg/gif/bmp/webp | Which files count as image slides |
| `video_extensions` | mov/mp4 | Which files count as video slides |
| `background_color` | `black` | Letterbox/pillarbox colour |
| `scale_mode` | `fill` | See [Display scaling](#display-scaling) |
| `sort_mode` | `alphabetical` | `alphabetical` or `numeric` |

**`services.face_recognition`**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Run the recognition service |
| `camera_id` | `0` | Picamera2 device index |
| `n` | `3` | Number of matches requested from the server |
| `model` | `mobilenet` | Recogniser + server gallery. See above. |
| `api_endpoint` | cluster `/identify/` | Identity lookup |
| `wpu_endpoint` | cluster `/wpu/images` | SAU/FRU cutouts (id as query param) |
| `sau_media_endpoint` | cluster `/sau/media` | Videos (id appended as a **path** segment) |
| `video_count` | `1` | Videos to take, by position. `0` disables video entirely. |
| `sau_cutout_filename` | `sau_cutout.png` | Fallback name for the body cutout |
| `fru_cutout_filename` | `fru_cutout.png` | Fallback name for the face cutout |
| `use_legacy_final_images` | `false` | Rollback valve — show server-composed images instead |
| `base_assets_max_bytes` | 5 GB | Asset cache cap before LRU eviction |
| `detection_interval` | `5` | Seconds between captures |
| `min_face_size` | `100` | Ignore faces smaller than this, in pixels |
| `display_result` | `true` | Show the recognition overlay |
| `overlay_hide_delay` | `3` | Seconds before the overlay hides |
| `person_timeout` | `10` | Seconds without a face before the visit ends |
| `diagnostic_mode` | `false` | Offline local matching (or use `--diagnostic`) |
| `diagnostic_gallery_dir` | `data/embeddings` | Seeded gallery location |
| `diagnostic_match_threshold` | `0.5` | Cosine gate for a local match |

Two notes on `video_count`: videos are identified **by position, not by name**,
because the server stores every upload under a generated `{uuid}{ext}` key and the
capture station's filename never reaches the client. The station uploads the
composited clip first, so `1` takes exactly that one. Raise it to also show the raw
captures behind it.

---

## Running the client

```bash
.venv/bin/python main.py                          # both services (default)
.venv/bin/python main.py --service slideshow      # display only, no camera
.venv/bin/python main.py --service face-recognition   # recognition only, no window
.venv/bin/python main.py --diagnostic             # offline mode
.venv/bin/python main.py --config /path/to.yaml   # alternate config
.venv/bin/python main.py --log-level DEBUG        # DEBUG/INFO/WARNING/ERROR
```

Logs go to stdout **and** `/var/log/wpu-client/app.log`.

### As a service

```bash
scripts/switch-mode.sh server       # online recognition via the registration server
scripts/switch-mode.sh diagnostic   # offline recognition, local gallery
scripts/switch-mode.sh stop         # stop both — releases the camera
scripts/switch-mode.sh status       # report which, if either, is active
```

The two units declare `Conflicts=` on each other, so starting one always stops the
other first. Neither is enabled at boot, so nothing runs after a reboot until you
start it. Use `stop` before anything else that needs the camera.

To start on boot anyway: `sudo systemctl enable --now slideshow-server.service`.

The units in `systemd/` are **templates** containing `@APP_DIR@`-style placeholders —
never install them by hand. `setup.sh` reads the real directory, user, uid and
session type off the device and substitutes them in.

### Keyboard controls

| Key | Action |
|---|---|
| `ESC` | Exit |
| `SPACE` / `→` | Next slide |
| `←` | Previous slide |
| `F` | Toggle fullscreen |

---

## Operating modes

### Server mode (production)

Recognition goes to the registration server; assets are fetched from it and composed
on-device. This is the default — `diagnostic_mode: false`.

### Diagnostic mode (offline)

Faces are matched against a gallery seeded on the Pi. No server is contacted for
anything, ever — not even as a fallback. Enable with `--diagnostic` or
`diagnostic_mode: true`.

Seed a person:

```bash
# Seed both models now, add the sketches later:
python scripts/seed_face.py --name "Ekan" --face /path/to/ekan.jpg

# Seed with sketches in one go:
python scripts/seed_face.py --name "Ekan" --face ekan.jpg --sketches /path/to/sketches/

# Seed a single model:
python scripts/seed_face.py --name "Ekan" --face ekan.jpg --models mobilenet
```

This writes `data/embeddings/<slug>/`:

```
embedding_mobilenet.npy    512-D L2-normalised vector
embedding_sface.npy        128-D L2-normalised vector
meta.json                  {"name": ..., "models": [...]}
sketches/                  slides shown when this person is matched
```

Both models are seeded by default so the gallery matches whichever `model:` is
configured at runtime. Sketches can be dropped into `sketches/` later without
re-seeding. Three people (`varun`, `samvaran`, `kevin`) ship pre-seeded.

Diagnostic mode has its own two-person composition using `data/duo_scenes/`, cached
to `data/duo_output/`. This shares no assets or code path with base-mode composition
and cannot affect it.

---

## What gets displayed

Slide source is decided **per visitor** by asset availability, not by mode. In
precedence order:

1. **Diagnostic mode** → that person's local sketches, or nothing. Never the server.
2. **Unrecognized face** → local slides if any exist, otherwise nothing. Never the
   server: an unrecognized face is tracked under a generated `__unrecognized__<hex>`
   pseudo-id that no endpoint can resolve, so asking is a guaranteed-failing round
   trip on the recognition thread.
3. **`use_legacy_final_images: true`** → the server's pre-composed images, even if
   local slides exist. This is the rollback valve, so it must win.
4. **Locally-composed slides exist** → use them.
5. **Otherwise** → the server's images.

So if the server sends composed scenes rather than raw cutouts, they are still
displayed, via rule 5. If it sends **both**, cutouts win at rule 4 and the composed
scenes are never fetched.

> **Edge case worth knowing:** a visitor with a SAU cutout but no FRU cutout still
> gets `sketch_dir` set, which satisfies rule 4 — so the server's face scenes are
> dropped for that visitor even though only body scenes could be composed.

Video slides are mixed in alongside images in whichever set is chosen.

---

## Diagnostics and troubleshooting

### Health check

```bash
scripts/switch-mode.sh stop                 # free the camera first
tail -f /var/log/wpu-client/app.log &
.venv/bin/python main.py --log-level DEBUG
```

A healthy startup logs, in order: the recogniser initialising with its dimension and
gallery name, the camera configuring, and the slideshow loading stock images.

Verify the environment independently:

```bash
.venv/bin/python -c "import numpy, cv2, onnxruntime; print(numpy.__version__, cv2.__version__, onnxruntime.__version__)"
.venv/bin/python -c "from picamera2 import Picamera2; print('picamera2 ok')"
```

Expect `1.26.4 4.11.0 <1.26+>`. A NumPy `2.x` here is the problem, not a detail.

### Testing the server without a Pi

```bash
python scripts/test_identify_sandbox.py --image face.jpg --base-url http://192.168.1.19:8000
```

Runs the same YuNet + MobileFaceNet pipeline the kiosk uses, posts the vector, and
prints what comes back. Defaults to `--model mobilenet` so a bare run reproduces
exactly what the kiosk sends.

### Symptom table

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to initialise recogniser 'mobilenet': No module named 'onnxruntime'` | Dependencies installed without onnxruntime. **This is fatal** — the recognition thread stops and the kiosk shows stock images forever. | `uv sync --frozen` |
| `MISSING models/mobilefacenet.onnx` from setup | Incomplete release zip | Re-cut the zip; the model is tracked in git |
| Recognition runs, nobody is ever matched | `model: sface` — querying the empty 128-D gallery | Set `model: "mobilenet"` |
| `_ARRAY_API not found` / "compiled using NumPy 1.x cannot be run in NumPy 2.x" | NumPy 2.x got installed | `uv sync --frozen`, or `pip install --force-reinstall numpy==1.26.4` |
| `uv sync` times out downloading opencv | 42 MB wheel vs uv's 30 s default | `UV_HTTP_TIMEOUT=900 uv sync --frozen` |
| "Camera unavailable" | Another service holds it exclusively | `scripts/switch-mode.sh stop` |
| Everything looks stretched | `scale_mode: fill` on a 16:9 monitor | `scale_mode: "fit"` |
| Everything looks squashed on the real panel | `scale_mode: fit` on the 6:7 panel | `scale_mode: "fill"` |
| Video plays with no picture | Missing GStreamer plugins | Install the `gstreamer1.0-plugins-*` packages above |
| No slides for a recognized visitor | No cutouts server-side, or no scene backgrounds locally | Check `data/base_scenes/` has backgrounds, not just `scenes_config.json` |
| First recognition stalls for ~30 s | An asset endpoint is unreachable and retrying | Check the endpoint host and `:8000` |

### Visit log

Every completed visit appends a row to `data/dataset/visit_log.csv`:

```
person_name, registration_id, timestamp_in, timestamp_out, confidence, frames_seen, duration_seconds
```

### Debug frames

```bash
SAVE_DEBUG_FRAMES=1 .venv/bin/python main.py
```

Writes annotated detection frames to `/tmp/face_detection_debug` (every 2nd frame,
50 max) for checking framing, lighting and detection boxes.

### Benchmarks

Standalone, not part of the app. Run with the camera free.

```bash
python scripts/benchmarks/YuNet_CPU_test.py       # detection latency and CPU
python scripts/benchmarks/capture_benchmark.py    # capture + payload sizing
python scripts/benchmarks/benchmark.py            # embedder throughput
```

---

## Server API contract

### Identify — `POST {api_endpoint}`

```json
{
  "type": "face",
  "n": 4,
  "model": "auraface",
  "face_vector": "0.0123,-0.0456,..."
}
```

`face_vector` is a **comma-separated string**, 512 floats for MobileFaceNet or 128
for SFace. `model` names the gallery: `auraface` for MobileFaceNet, `sface` for
SFace. It is required — more than one embedding space exists server-side, and the
server cannot infer which one a probe belongs to.

Response fields consumed by the client:

```json
{
  "registration_id": "…",
  "name": "John Doe",
  "preferred_name": "John",
  "confidence": 0.95,
  "distance": 0.31,
  "match_type": "…",
  "gender": "male",
  "message": "…"
}
```

A response is treated as a match when `name` is non-null. `gender` is optional and
selects which local scenes the person may be composed onto; servers that don't return
it leave it null, and a null gender never excludes a scene.

### Cutouts — `GET {wpu_endpoint}?registration_id=<id>`

Returns `sau_signed_url` and `fru_signed_url`. Labels win. The older flat
`signed_urls` list is still read as a fallback and matched by filename suffix, which
works because the server writes cutouts to fixed keys
(`<registration_id>/wpu/sau_cutout.png`).

### Videos — `GET {sau_media_endpoint}/<registration_id>`

The id is a **path segment**, not a query parameter — this endpoint differs from the
others. Returns `video_urls` (or a legacy scalar `video_url`). Selected **by
position**: the server stores each upload under a generated `{uuid}{ext}` key, so
there is no filename to match on.

Asset listings use a single-attempt request rather than the retrying client. A
registration with no media legitimately 404s, and retrying that three times stalled
the recognition thread for 30 seconds per visitor.

---

## Data, caches and disk usage

```
data/
├── stock_images/     Default slideshow loop (tracked)
├── embeddings/       Diagnostic gallery — 3 seeded people (tracked)
├── people/           Reference photos for those people (tracked)
├── base_scenes/      Backgrounds + placement configs for on-device composition
│   ├── sau_single/   {"<id>": {"scale": f, "anchor": [x, y]}}
│   ├── sau_duo/      {"<id>": {"scale_1": …, "anchor_1": …, "scale_2": …, "anchor_2": …}}
│   ├── fru_single/   {"<id>": {"gender": [...], "face_anchor": {...}}}
│   └── fru_duo/      same shape as duo_scenes
├── duo_scenes/       Diagnostic-only duo backgrounds + config (tracked)
├── duo_output/       Composed diagnostic duo scenes (gitignored)
├── base_assets/      Fetched cutouts + composed slides (gitignored)
│   ├── <registration_id>/
│   │   ├── raw/          sau.png, fru.png, video_1.mov
│   │   ├── manifest.json per-asset fetch state + ETag validators
│   │   └── display/      composed slides + hardlinked videos
│   └── duo/<id_a>__<id_b>/
└── dataset/          Captured frames + visit_log.csv (gitignored)
```

Scene backgrounds are `<id>.png`, where the filename stem matches the JSON key in
that directory's `scenes_config.json`.

**Caps and eviction.** `base_assets/` is capped by `base_assets_max_bytes` (5 GB
default); over the cap, least-recently-used entries are evicted, duo pairs first.
Composed output is JPEG q90 — roughly 6× smaller than PNG at no visible cost on a
projector. `dataset/` is capped at 30 GB with at most 50 frames saved per visit.
Videos are hardlinked into `display/`, and the size accounting is inode-aware so a
hardlink isn't counted twice.

**Cache freshness.** The server writes each cutout to a fixed object key, so
re-photographing a person overwrites the same path — a URL comparison alone could
never notice. Each cached asset records the ETag/Last-Modified it was fetched with,
and a cheap `HEAD` revalidates before reuse. A moved validator re-downloads the
asset and invalidates every slide and pair composed from the old copy.

---

## Development

```bash
uv sync --extra dev
ruff check .            # line length 110; rules E, F, I, B
pytest -q
```

CI runs both on pull requests to `main` and on pushes to `main`. It installs an
explicit portable subset of the runtime dependencies rather than the project itself,
because `pygobject` and `picamera2` need a GTK4/libcamera stack that only exists on a
Pi. NumPy and OpenCV are pinned there to the same versions `pyproject.toml` ships, so
CI tests the stack that actually runs. Tests therefore cover only modules that don't
import the camera or display; `tests/conftest.py` stubs `gi` for the slideshow tests.

**A branch push does not run CI** — open a pull request against `main` to get the
lint and test jobs on record.

### Cutting a release

```bash
git tag vX.Y.Z
scripts/make_release.sh vX.Y.Z     # -> wpu-client-vX.Y.Z.zip
```

The zip bundles code, all three models, the config template, stock images and the
three seeded people — everything needed to `unzip` on a Pi and run `setup.sh`.
`config.yaml`, `dataset/`, `benchmark_output/` and `archive/` are never included.

---

## Project structure

```
.
├── main.py                       Entry point, CLI, service orchestration
├── wpu_client/
│   ├── paths.py                  Project-root-relative path resolution
│   ├── config/settings.py        Pydantic config schema + YAML loading
│   ├── core/
│   │   ├── events.py             Pub/sub event bus
│   │   └── service_base.py       Service lifecycle base class
│   ├── models/api.py             Request/response models
│   ├── services/
│   │   ├── face_recognition/
│   │   │   ├── face_service.py         Capture, detect, embed, track, orchestrate
│   │   │   ├── sface_embedder.py       Camera-free embedders (shared with scripts)
│   │   │   ├── asset_fetcher.py        Download + cache cutouts/videos, disk budget
│   │   │   ├── base_composer.py        Compose SAU/FRU onto base scenes
│   │   │   ├── duo_composer.py         Diagnostic two-person composition
│   │   │   └── diagnostic_gallery.py   Local seeded gallery
│   │   └── slideshow/slideshow_service.py   GTK4 UI, video playback, mode switching
│   └── utils/http.py             HTTP client (retrying + single-attempt variants)
├── models/                       Runtime ONNX weights (tracked, bundled)
├── config/
│   ├── config.yaml.example       Tracked template
│   └── config.yaml               Per-device, gitignored
├── data/                         See above
├── scripts/
│   ├── setup.sh                  Pi bootstrap, idempotent
│   ├── switch-mode.sh            server ⇄ diagnostic ⇄ stop
│   ├── make_release.sh           Versioned release zip
│   ├── seed_face.py              Seed a person into the diagnostic gallery
│   ├── test_identify_sandbox.py  Exercise the identify API from a laptop
│   └── benchmarks/               Standalone perf scripts
├── systemd/                      Unit templates (@APP_DIR@ placeholders)
└── tests/                        pytest suite
```

### Adding a service

1. Create a directory under `wpu_client/services/`.
2. Inherit from `ServiceBase` and implement `start()` / `stop()`.
3. Add a config model in `wpu_client/config/settings.py` and a field on
   `ServicesConfig`.
4. Register it in `main.py`.
5. Communicate through the event bus — services should not import each other.

Published events:

| Event | Payload | Meaning |
|---|---|---|
| `faces.recognized` | `lines` — one overlay string per visible face | Overlay refresh |
| `person.detected` | `registration_id`, `person_name`, `confidence`, `sketch_dir`, `unrecognized` | Switch to this visitor's slides |
| `person.left` | `registration_id`, `person_name` | Visit ended; return to the stock loop |

---

## Release history

Newest first. Full detail is in `git log`.

**2026-08 — Production model, packaging and display scaling.**
MobileFaceNet is now the production recogniser everywhere: `onnxruntime` moved from
an optional extra into base dependencies (a missing import is fatal, not a
degradation), `models/mobilefacenet.onnx` is tracked again, `setup.sh` requires it,
and all defaults name it. `scale_mode` now actually drives display scaling, making
the production 6:7 panel and an ordinary test monitor a one-key switch. SAU videos
were wired to the real `/api/v1/sau/media/<id>` route and are selected by position;
cutouts read the labelled response fields, with filename matching kept as a fallback.
Unrecognized faces no longer trigger doomed server round trips.

**2026-08-13 — Local SAU/FRU composition in base mode.**
The Pi composes each visitor's slides on-device from raw alpha cutouts instead of
downloading pre-composed images, moving the "pre-render every person × scene" cost
off the server. Added duo display in base mode, ETag-based cache invalidation, the
LRU disk budget, and the `use_legacy_final_images` rollback valve.

**2026-07-11 — Repo reorganisation and two systemd modes.**
Models moved to a top-level `models/`, images to `data/`, config to `config/`.
Systemd units install disabled so they never steal the camera. `seed_face.py` seeds
both models per person. Fixed a slideshow crash on mixed numeric/alphabetic filenames
by defaulting `sort_mode` to `alphabetical`.

**2026-06-17 — Offline diagnostic mode.**
Local seeded gallery matching with no server dependency, plus two-person scene
composition for the diagnostic path.

**Earlier — migration from GlintR100-INT8 to on-device embedding.**
Replaced a ~250% CPU AuraFace/GlintR100 session with lighter on-device pipelines,
established that the client's embedding must mirror the server backend exactly
(alignment included) for probe vectors to be comparable to gallery vectors, and made
`model` a required field on the identify request.
