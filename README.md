# WPU Client

Face recognition slideshow application with modular service architecture.

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
