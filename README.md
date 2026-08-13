# WPU Client

Face recognition slideshow application with modular service architecture, for a
Raspberry Pi kiosk display.

## Changelog

### 2026-08-13 — Local SAU/FRU composition in base (server-recognition) mode

Base mode no longer downloads pre-composed final images. The server serves two raw
alpha cutouts per person — a body cutout (**SAU**) and a face cutout (**FRU**) — over
the existing `wpu_endpoint`, and the Pi composes them onto local scene backgrounds
live, for both single-person and 2-person display. This moves the "pre-render every
person × scene combination" cost off the server: it stores one cutout pair per
person, and the Pi composes on demand.

- **New scene assets** under `data/base_scenes/{sau_single,sau_duo,fru_single,fru_duo}/`,
  each a `scenes_config.json` plus background images. **These do not exist yet** — the
  code degrades cleanly until they're supplied, composing nothing and falling back.
- **Duo display now works in base mode too**, not just diagnostic. If one of a pair
  has no usable assets, the other is shown alone rather than nothing.
- **Caching:** fetched cutouts and composed slides live under
  `data/base_assets/<registration_id>/`, pairs under `data/base_assets/duo/<a>__<b>/`.
  Cutouts are revalidated by ETag, so re-photographing a person (which overwrites the
  same object key server-side) correctly invalidates their slides and every pair they
  appear in. Composed output is JPEG q90 — ~6× smaller than PNG at no visible cost on
  a projector.
- **Disk budget:** `base_assets_max_bytes` (default 5 GB) caps the cache; over it,
  least-recently-used entries are evicted, duo pairs first since pairs grow
  quadratically with visitor count and rebuild cheaply from the retained cutouts.
- **Rollback valve:** `use_legacy_final_images: true` reverts to the old
  pre-composed-image path with one config flip, no redeploy.
- **Diagnostic mode is unchanged** — its duo feature (`data/duo_scenes/`,
  `data/duo_output/`, `compose_duo`) shares no assets or code path with the above.

Pending backend work, tracked as placeholders in config: the `gender` field on the
`/identify` response, and the videos endpoint (`wpu_videos_endpoint` — no such route
exists server-side yet). Video fetching is best-effort and single-person only; a
missing video never fails anything.

### 2026-07-11 — Repo reorganisation, dual-model cleanup, two systemd modes

Restructured the repo for release packaging and fixed several latent bugs, without
changing the recognition behaviour (both `sface` and `mobilenet` models are still
supported — this is v1, accuracy is still being validated between them).

- **Layout:** `.onnx` model weights moved out of `wpu_client/services/face_recognition/`
  into a top-level `models/`; `stock_images/` → `data/stock_images/`; the diagnostic
  gallery `diagnostic_gallery/` → `data/embeddings/`; `config.yaml` → `config/config.yaml`
  (template at `config/config.yaml.example`); `dataset/` (runtime capture output) →
  `data/dataset/`. See **Project Structure** below.
- **Diagnostic gallery pruned to 3 people** (`varun`, `samvaran`, `kevin`) for the
  release; the other 10 previously-seeded people were moved (not deleted) to a
  git-ignored `archive/` for reversibility.
- **`scripts/seed_face.py`** (moved from `tools/`) now seeds **both** models per person
  (`embedding_sface.npy` + `embedding_mobilenet.npy`) in one pass, matching what
  `diagnostic_gallery.py` actually expects — the previous copy only wrote a legacy
  single `embedding.npy`. **`scripts/test_identify_sandbox.py`** (also moved from
  `tools/`) now reuses the shared embedder module instead of a hand-duplicated
  detect/align/embed pipeline, and supports `--model {sface,mobilenet}`.
- **Fixed a slideshow crash:** `sort_mode: "numeric"` (the default) raised
  `TypeError: '<' not supported between instances of 'int' and 'str'` the moment a
  non-numerically-named image sat in the same directory as numerically-named ones. The
  sort key is now type-safe regardless of filename mix; default `sort_mode` is now
  `"alphabetical"`.
- **Added `psutil`** to `pyproject.toml` — it's imported unconditionally by
  `face_service.py` but was missing from declared dependencies.
- **Two systemd services** (`systemd/slideshow-server.service`,
  `systemd/slideshow-diagnostic.service`), mutually exclusive via `Conflicts=`, plus
  `scripts/switch-mode.sh {server|diagnostic}` to flip between them. See **Deployment**.
- **`scripts/setup.sh`** — idempotent Pi bootstrap (apt deps, venv, models check,
  systemd install). **`scripts/make_release.sh vX.Y.Z`** — packages a versioned release
  zip (code + both models + config + stock images + the 3 seeded people).
- Removed genuinely dead/duplicate files: a stray duplicate `diagnostic_gallery.py`
  copy, an old backup of `slideshow_service.py`, two orphaned/unreferenced `.onnx`
  experiments (`exp3_freeze90_arcface.onnx`, `glintr100_int8_static_150.onnx`), the
  never-read `face_stock_images/` (obsolete gender-based sketch selection, superseded
  by the per-person diagnostic gallery below), and some stray scratch files.

### 2026-06-17 Ekanth — Diagnostic (offline) mode

Added a fully **offline diagnostic mode** so the WPU can recognise people and show
sketches **without the server / Triton** (useful when the Pi can't reach the
sandbox, or for a self-contained demo). Toggle with `--diagnostic` (or
`diagnostic_mode: true` under `face_recognition` in `config.yaml`).

**Flow:**
1. **Seed** a known person with `scripts/seed_face.py --name <N> --face <img> [--sketches <dir|img>]`.
   The same YuNet → SFace/MobileFaceNet pipeline generates an embedding per model, saved
   to `data/embeddings/<slug>/embedding_<model>.npy` + `meta.json`, alongside a per-person
   `sketches/` folder.
2. At runtime the face service **matches live faces against the local gallery**
   by cosine distance (`diagnostic_match_threshold`, default `0.5`) — no `/identify`
   call, no network.
3. On a match it emits `person.detected` carrying that person's `sketch_dir`, and the
   slideshow shows **that individual's own** sketches from
   `data/embeddings/<slug>/sketches/`. No face → stock images, as usual.

**Sketches are per-person**, not by gender: each seeded person's face-swap sketch(es)
live in their own `sketches/` folder and are shown only when *that* person is
recognised. The render (the actual face-swap) is produced externally and is decoupled
from the WPU — sketches can be dropped into `sketches/` at seed time (`--sketches`) or
later when the render is delivered, with no re-seed. If the folder is empty (sketch not
ready yet), recognition still fires but the slideshow stays on stock images.

**Key files:**
- `wpu_client/services/face_recognition/sface_embedder.py` — camera-free YuNet+SFace /
  YuNet+MobileFaceNet embedders (no `picamera2`), the single source of truth shared by
  the seed tool and verified byte-identical to the live service path.
- `wpu_client/services/face_recognition/diagnostic_gallery.py` — loads seeded entries
  (embeddings + per-person `sketches/`) and matches by cosine distance.
- `scripts/seed_face.py` — seed a face (and optionally sketches) into the local gallery.

### 2026-06-17 Ekanth

Migrated the on-device face-recognition model from **GlintR100-INT8 (512D)** to
**SFace-128D (OpenCV)** to cut CPU usage on the Raspberry Pi 4 and to make the WPU
probe embeddings comparable to the registration gallery. A **MobileFaceNet-512D**
path (→ server's `auraface` gallery) was added alongside it later; both are
currently kept and selected with `model:` in `config.yaml` (see reorg entry above).

- **Recognition model:** `cv2.FaceRecognizerSF` with
  `face_recognition_sface_2021dec.onnx` replaces the GlintR100 `onnxruntime`
  session. SFace is far lighter than AuraFace/GlintR100 — AuraFace was spiking to
  ~250% CPU (~160% even after INT8 quantization); SFace runs comfortably under one
  core. SFace is also commercially usable.
- **Embedding pipeline mirrors the registration server (`SFaceBackend`)
  exactly**, so the 128D probe vectors are directly comparable to the gallery
  vectors in the `face_vectors_sface` Qdrant collection:
  - Keep the **raw YuNet detection row (bbox + 5 landmarks)** — the previous code
    discarded the landmarks and did a naive bbox crop, which is *not* valid for
    SFace.
  - `alignCrop(BGR frame, raw YuNet row)` → 112×112 ArcFace template →
    `feature()` → **L2-normalise**.
  - Largest face selected by bbox area (same rule as registration).
- **API request:** `IdentifyRequest` sends `model="sface"` (or `"auraface"` for the
  MobileFaceNet path). This is required — the server needs it to disambiguate which
  gallery a probe vector belongs to, since more than one 128D embedding space exists
  server-side. The server queries the matching collection and applies its calibrated
  threshold.
- Model files (`face_recognition_sface_2021dec.onnx`, `mobilefacenet.onnx`,
  `face_detection_yunet_2023mar.onnx`) live under `models/`, are bundled in release
  zips, and are gitignored-by-default with explicit tracking exceptions (see
  `.gitignore`).

**Operational notes:**
- The local-cache similarity threshold (`_similarity_threshold = 0.4`, cosine)
  governs only the "same person still here" de-dup and may need re-tuning per model's
  distance distribution. Server-side matching uses its own calibrated threshold and is
  unaffected.
- WPU can only recognise people who have a vector in the matching server-side
  collection for the configured model. New registrations get one automatically;
  older registrations may need backfilling for a newly-added model.

## Features

- **Slideshow Service**: Full-screen image display with configurable timing
- **Face Recognition Service**: Continuous face detection and identification, either
  via the registration server or fully offline (diagnostic mode)
- **Local scene composition**: Composes each visitor's slides on-device from raw
  SAU/FRU cutouts onto local backgrounds, for one or two people, instead of
  downloading pre-rendered images
- **Event Bus**: Inter-service communication for displaying recognition results
- **Configurable**: YAML-based configuration for all settings
- **Extensible**: Easy to add new services (voice, gesture, etc.)

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip, in a venv
pip install -e .
```

On a fresh Raspberry Pi, prefer `scripts/setup.sh` (see **Deployment** below) — it
installs system packages, creates the venv, verifies the bundled models are present,
and installs the systemd units in one idempotent pass.

### System Dependencies

GTK4 (slideshow UI) and Picamera2 (camera capture) need system packages; there is
**no `dlib` dependency** in this project, so no `cmake`/`g++`/`libboost` toolchain is
required.

Video slides (mixed in alongside image slides — see `video_extensions` in
`config/config.yaml.example`) are decoded with a manual GStreamer pipeline
(`filesrc ! decodebin ! videoconvert ! appsink`), not `Gtk.MediaFile` — the
`gtk4paintablesink` package that backend needs isn't available on all
Debian/Raspberry Pi OS releases, and without it video plays silently but
never paints a frame. The manual pipeline only needs the plugins below.

**Debian/Ubuntu (incl. Raspberry Pi OS):**
```bash
sudo apt install -y \
    python3 python3-venv python3-dev \
    libgtk-4-1 gir1.2-gtk-4.0 python3-gi python3-gi-cairo \
    python3-picamera2 \
    libgl1 libglib2.0-0 \
    gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

## Configuration

Copy the template and edit for your deployment:

```bash
cp config/config.yaml.example config/config.yaml
```

```yaml
services:
  slideshow:
    enabled: true
    full_screen: true
    advance_time: 3
    image_directory: "data/stock_images"

  face_recognition:
    enabled: true
    camera_id: 0
    n: 4
    model: "sface"  # or "mobilenet" — see Changelog
    api_endpoint: "http://localhost:8000/api/v1/identify/"
    detection_interval: 5
```

`config/config.yaml` itself is gitignored (only `config/config.yaml.example` is
tracked) so local endpoint/host edits never get committed by accident.

## Usage

```bash
# Run all services
python main.py

# Run only slideshow
python main.py --service slideshow

# Run only face recognition
python main.py --service face-recognition

# Use custom config
python main.py --config /path/to/config.yaml

# Set log level
python main.py --log-level DEBUG

# Offline diagnostic mode (local gallery, no server)
python main.py --diagnostic
```

## Deployment

### Fresh Raspberry Pi setup

```bash
unzip wpu-client-vX.Y.Z.zip -d am-wpu-client
cd am-wpu-client
chmod +rx scripts/setup.sh     # see note below
./scripts/setup.sh
```

Idempotent — installs apt system packages, creates a `--system-site-packages` venv
(needed so `picamera2`/`gi` resolve from the system packages above), verifies the
bundled models are present under `models/`, and installs + enables the
`slideshow-server` systemd unit (leaving `slideshow-diagnostic` installed but
disabled).

The app can be unpacked anywhere and run as any user — `setup.sh` reads the actual
directory, user, uid and session type off the device and substitutes them into the
systemd unit templates in `systemd/` before installing them to
`/etc/systemd/system/`. Do not install those templates by hand; they contain
`@APP_DIR@`-style placeholders. Fleet convention is `/home/dreamvu/wpu_client`.

`config/config.yaml` is **per-device**: untracked, not shipped in the release zip, and
seeded from `config/config.yaml.example` by step `[5/8]`. Point `api_endpoint` /
`wpu_endpoint` at the master server before first run — setup echoes their current
values so a default `localhost` is obvious in the log.

Production installs `sface` only. The research `mobilenet` model needs `onnxruntime`,
which is an optional extra rather than a base dependency:

```bash
./.venv/bin/pip install -e '.[research]'   # only if running model: "mobilenet"
```

**Note on the `chmod`:** `unzip` applies the current umask, and an unusual umask can
land the scripts as mode `111` (`---x--x--x`) — executable but not *readable*, which
makes `./scripts/setup.sh` fail with "Permission denied" because bash has to read a
script, not just have `+x` on it. The `chmod` above unsticks `setup.sh` itself;
step `[0/8]` then re-asserts sane modes across the rest of the tree.

### Switching between server and diagnostic mode

```bash
scripts/switch-mode.sh server      # online recognition via the registration server
scripts/switch-mode.sh diagnostic  # offline recognition, local 3-person gallery
```

The two systemd units declare `Conflicts=` on each other, so starting one always
stops the other first — they never fight over the camera or display.

### Cutting a release

```bash
git tag v1.0.0
scripts/make_release.sh v1.0.0     # -> wpu-client-v1.0.0.zip
```

The zip bundles code, both runtime models (~45 MB total), config template, stock
images, and the 3 seeded diagnostic people — everything needed to `unzip` on a Pi and
run `scripts/setup.sh`. `dataset/`, `benchmark_output/`, and `archive/` are never
included.

## Keyboard Controls

- **ESC** - Exit application
- **SPACE / RIGHT** - Next image
- **LEFT** - Previous image
- **F** - Toggle fullscreen

## API Format

The face recognition service sends POST requests to `/api/v1/identify/`:

```json
{
  "type": "face",
  "n": 4,
  "model": "sface",
  "face_vector": [0.1, 0.2, ..., 0.9]
}
```

Expected response:
```json
{
  "success": true,
  "person_name": "John Doe",
  "confidence": 0.95,
  "message": "Face identified successfully"
}
```

## Project Structure

```
.
├── wpu_client/                 # main package (unchanged location)
│   ├── paths.py                 # project-root-relative path resolution
│   ├── config/                  # configuration management
│   ├── core/                    # base classes and event bus
│   ├── services/
│   │   ├── face_recognition/    # detection + recognition + diagnostic gallery
│   │   └── slideshow/           # GTK4 slideshow UI
│   ├── models/                  # API request/response models
│   └── utils/                   # HTTP client, etc.
├── models/                     # runtime .onnx weights (bundled + tracked)
├── data/
│   ├── stock_images/            # default slideshow images
│   ├── embeddings/              # diagnostic gallery (3 seeded people, tracked)
│   ├── people/                  # reference photos for the seeded people
│   ├── duo_scenes/              # diagnostic duo backgrounds + placement config
│   ├── duo_output/              # gitignored: composed diagnostic duo scenes
│   ├── base_scenes/             # base-mode backgrounds + placement configs
│   │                            #   sau_single/ sau_duo/ fru_single/ fru_duo/
│   └── base_assets/             # gitignored: fetched cutouts + composed slides,
│                                #   size-capped, LRU-evicted
├── archive/                    # git-ignored: pruned diagnostic people, not deleted
├── config/
│   ├── config.yaml.example      # tracked template
│   └── config.yaml              # your local copy (gitignored)
├── scripts/
│   ├── setup.sh                 # Pi bootstrap
│   ├── make_release.sh          # versioned release zip
│   ├── switch-mode.sh           # server <-> diagnostic
│   ├── seed_face.py             # seed a person into the diagnostic gallery
│   ├── test_identify_sandbox.py # test the identify API from a laptop, no Pi needed
│   └── benchmarks/              # standalone perf scripts, not part of the app
├── systemd/
│   ├── slideshow-server.service
│   └── slideshow-diagnostic.service
└── main.py                     # entry point
```

## Adding New Services

1. Create a new directory under `wpu_client/services/`
2. Inherit from `ServiceBase`
3. Add configuration to `config/settings.py`
4. Register in `main.py`
