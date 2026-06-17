# WPU Client

Face recognition slideshow application with modular service architecture.

## Changelog

### 2026-06-17 Ekanth — Diagnostic (offline) mode

Added a fully **offline diagnostic mode** so the WPU can recognise people and show
sketches **without the server / Triton** (useful when the Pi can't reach the
sandbox, or for a self-contained demo). Toggle with `--diagnostic` (or
`diagnostic_mode: true` under `face_recognition` in `config.yaml`).

**Flow:**
1. **Seed** a known person with `tools/seed_face.py --name <N> --gender <male|female> --face <img>`.
   The same YuNet → SFace pipeline generates a 128-D embedding, saved to
   `diagnostic_gallery/<slug>/embedding.npy` + `meta.json` (`{name, gender}`).
2. At runtime the face service **matches live faces against the local gallery**
   by cosine distance (`diagnostic_match_threshold`, default `0.5`) — no `/identify`
   call, no network.
3. On a match it emits `person.detected` carrying the person's **gender**, and the
   slideshow shows that gender's face-swap sketches from
   `face_stock_images/<gender>/`. No face → stock images, as usual.

**Sketch selection is by gender, not per person** (Option A): the seed stores each
person's gender, which picks `face_stock_images/male/` or `…/female/`. Everyone of
the same gender sees the same sketch set — matching the supplied assets (10 sketches
split 5 male / 5 female). Chosen because diagnostic mode seeds *known* people, so the
gender is known at seed time — zero extra models, zero runtime misclassification.

**New files:**
- `wpu_client/services/face_recognition/sface_embedder.py` — camera-free YuNet+SFace
  embedder (no `picamera2`), the single source of truth shared by the seed tool and
  verified byte-identical to the live service path.
- `wpu_client/services/face_recognition/diagnostic_gallery.py` — loads seeded entries
  and matches by cosine distance.
- `tools/seed_face.py` — seed a face + gender into the local gallery.

**Other changes:**
- `face_service.py` — diagnostic branch (`_match_local_gallery`) replaces the server
  call; `gender` plumbed through `person.detected`. Server mode is unchanged.
- `slideshow_service.py` — in diagnostic mode, visitor mode loads local
  `face_stock_images/<gender>/` sketches instead of fetching signed URLs.
- Config: `diagnostic_mode`, `diagnostic_gallery_dir`, `diagnostic_match_threshold`,
  `face_sketch_directory`.
- `.gitignore` — `face_stock_images/` and `diagnostic_gallery/` are gitignored
  (heavy assets / local data; deploy to the Pi separately like the model files).

**Validated off-Pi:** seed → gallery → match works end-to-end, and the seed embedding
is byte-identical to the live SFace probe (same first-3 values as the Stage-A harness),
confirming gallery and live vectors share one embedding space.

### 2026-06-17 Ekanth

Migrated the on-device face-recognition model from **GlintR100-INT8 (512D)** to
**SFace-128D (OpenCV)** to cut CPU usage on the Raspberry Pi 4 and to make the WPU
probe embeddings comparable to the registration gallery.

- **Recognition model:** `cv2.FaceRecognizerSF` with
  `face_recognition_sface_2021dec.onnx` replaces the GlintR100 `onnxruntime`
  session. SFace is far lighter than AuraFace/GlintR100 — AuraFace was spiking to
  ~250% CPU (~160% even after INT8 quantization); SFace runs comfortably under one
  core. SFace is also commercially usable.
- **Embedding pipeline now mirrors the registration server (`SFaceBackend`)
  exactly**, so the 128D probe vectors are directly comparable to the gallery
  vectors in the `face_vectors_sface` Qdrant collection:
  - Keep the **raw YuNet detection row (bbox + 5 landmarks)** — the previous code
    discarded the landmarks and did a naive bbox crop, which is *not* valid for
    SFace.
  - `alignCrop(BGR frame, raw YuNet row)` → 112×112 ArcFace template →
    `feature()` → **L2-normalise**.
  - Largest face selected by bbox area (same rule as registration).
- **API request:** `IdentifyRequest` now sends `model="sface"`. This is required —
  the server cannot disambiguate a 128D vector otherwise (dlib is also 128D). The
  server queries the `face_vectors_sface` collection and applies its calibrated
  `sface_threshold`.
- Model files (`face_recognition_sface_2021dec.onnx`,
  `face_detection_yunet_2023mar.onnx`) are gitignored and deployed to the Pi
  separately.

**Operational notes:**
- The local-cache similarity threshold (`_similarity_threshold = 0.4`, cosine)
  governs only the "same person still here" de-dup and may need re-tuning for
  SFace's 128D distribution. Server-side matching uses the calibrated
  `sface_threshold` and is unaffected.
- WPU can only recognize people who have a 128D vector in `face_vectors_sface`.
  New registrations get one automatically; registrations made before `sface` was
  added to the server's `registration_models` need backfilling.

## Features

- **Slideshow Service**: Full-screen image display with configurable timing
- **Face Recognition Service**: Continuous face detection and identification via API
- **Event Bus**: Inter-service communication for displaying recognition results
- **Configurable**: YAML-based configuration for all settings
- **Extensible**: Easy to add new services (voice, gesture, etc.)

## Installation

```bash
# Install dependencies (requires dlib)
pip install -r requirements.txt

# Or using uv (if available)
uv sync
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y \
    python3-dev \
    cmake \
    g++ \
    libgtk-3-dev \
    libboost-all-dev \
    libglib2.0-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install -y \
    python3-devel \
    cmake \
    gcc-c++ \
    gtk3-devel \
    boost-devel \
    glib2-devel
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
services:
  slideshow:
    enabled: true
    full_screen: true
    advance_time: 3
    image_directory: "stock_images"

  face_recognition:
    enabled: true
    camera_id: 0
    n: 4
    api_endpoint: "http://localhost:8000/api/v1/identify/"
    detection_interval: 5
```

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
```

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
  "face_vector": [0.1, 0.2, ..., 0.9]  // 128 floats
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
wpu_client/
├── config.yaml           # Global configuration
├── main.py               # Application entry point
├── wpu_client/
│   ├── config/           # Configuration management
│   ├── core/             # Base classes and event bus
│   ├── services/         # Service implementations
│   ├── models/           # Data models
│   └── utils/            # Utilities
└── stock_images/         # Images for slideshow
```

## Adding New Services

1. Create a new directory under `wpu_client/services/`
2. Inherit from `ServiceBase`
3. Add configuration to `config/settings.py`
4. Register in `main.py`
