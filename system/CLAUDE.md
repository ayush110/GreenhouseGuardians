# Greenhouse Guardians - System

## Project Overview

Multi-camera synchronized data acquisition system for greenhouse monitoring. Captures simultaneous RGB + depth images from 4 cameras across multiple Raspberry Pis, orchestrated via a web dashboard.

## Architecture

```
172.20.10.2 : Pi 4 + Intel RealSense D435 → port 8000  (d435_server.cpp)
172.20.10.4 : Pi Zero RIGHT               → port 8001  (pi_zero_right.py)
172.20.10.5 : Pi Zero BOTTOM              → port 8001  (pi_zero_bottom.py)
172.20.10.7 : Pi Zero LEFT                → port 8001  (pi_zero_left.py)
127.0.0.1   : Dashboard (local)           → port 5050  (dashboard.py)
```

## Key Files

| File | Role |
|------|------|
| `launch.sh` | SSH into all Pis, start services in tmux, launch dashboard |
| `dashboard.py` | Flask web UI — triggers synchronized captures, displays images |
| `d435_server.cpp` | C++ HTTP server for RealSense D435 (RGB + aligned depth) |
| `pi_zero_left/right/bottom.py` | Flask servers on each Pi Zero (picamera2) |
| `pixel_depth.py` | Utility to extract 3D point from depth + intrinsics |

The root-level `dash.py` and `app.py` are more advanced versions of the dashboard and Pi camera server respectively (with `trigger_at_ms` synchronization support).

## Synchronized Capture Flow

1. Dashboard computes `trigger_at_ms` = now + 1500ms
2. Sends parallel POST `/capture?trigger_at_ms=<ts>` to all 4 cameras
3. Each camera waits until trigger time, then captures
4. Dashboard receives responses, writes organized `captures/<timestamp>/` directory

### Capture Output Structure
```
captures/20260318_143728_778/
├── d435/
│   ├── rgb.jpg           # JPEG from RealSense color sensor
│   ├── depth_raw.png     # 16-bit depth PNG (1 unit = 1mm)
│   ├── depth_raw.npy     # NumPy uint16 array of same
│   └── meta.json         # Camera intrinsics (fx, fy, cx, cy) + depth_scale
├── left/
│   ├── rgb.jpg
│   └── meta.json
├── right/  ...
└── bottom/ ...
```

## Camera Settings

**Pi Zero (picamera2):**
- Preview/stream: 960×540
- Still capture: 4608×2592 @ JPEG quality 95

**D435 (librealsense2):**
- RGB + depth: 640×480 @ 15 FPS (configurable via CLI args)
- Depth format: RS2_FORMAT_Z16, depth scale 0.001 m/unit

## Building the D435 Server

```bash
g++ -std=c++17 d435_server.cpp -o d435_server \
  -lrealsense2 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
```

Dependencies: `librealsense2`, `opencv`, `httplib` (header-only), `nlohmann/json` (header-only)

## Running Locally (Dashboard Only)

```bash
cd /Users/ayushshah/Projects/GreenhouseGuardians
source myenv/bin/activate
python3 system/dashboard.py   # or python3 dash.py for the advanced version
# Open http://127.0.0.1:5050
```

## Running the Full System

```bash
cd system
bash launch.sh
```

Requires SSH access to all four Pis and tmux installed locally.

## API Endpoints

**Pi Zero servers (port 8001):**
- `GET /health` — status check
- `GET /stream/rgb.mjpg` — MJPEG live preview
- `POST /capture` — high-res still; optional `trigger_at_ms` query param

**D435 server (port 8000):**
- `GET /health`
- `GET /stream/rgb.mjpg` — color MJPEG stream
- `GET /capture` — returns binary: [RGB JPEG] + [Depth PNG], with intrinsics in headers

## Development Notes

- `capture_scripts/` contains older/experimental capture tools — not part of the active system
- `pixel_depth.py` is a standalone utility; pass a capture directory to extract 3D points
- The `myenv/` virtualenv is at the repo root, not inside `system/`
- Depth `.npy` files store raw uint16 values; multiply by `depth_scale_m_per_unit` (0.001) for meters
