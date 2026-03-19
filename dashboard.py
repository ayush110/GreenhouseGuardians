from flask import Flask, render_template_string, jsonify, request
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

app = Flask(__name__)

D435_BASE = "http://172.20.10.2:8000"

PI_ZERO_CAMS = {
    "left":   "http://172.20.10.7:8001",
    "right":  "http://172.20.10.4:8001",
    "bottom": "http://172.20.10.5:8001",
}

API_BASE = "https://kayenm-greenhouseguardians.hf.space/api/upload"

SAVE_DIR = Path.home() / "multi_camera_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HTTP = requests.Session()
HTTP.mount("http://",  requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))
HTTP.mount("https://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))

HTML = """
<!doctype html>
<html>
<head>
    <title>Multi-Camera Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 24px;
            background: #f5f7fb;
            color: #111827;
        }
        .topbar {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 15px;
        }
        .btn:hover { background: #1d4ed8; }
        .status { font-weight: 600; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            padding: 16px;
        }
        img {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            background: #e5e7eb;
        }
        code {
            background: #eef2ff;
            padding: 2px 6px;
            border-radius: 6px;
        }
        .small {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 4px;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #e5e7eb;
        }
        .field-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .field-row label {
            font-size: 14px;
            font-weight: 600;
            white-space: nowrap;
        }
        .field-row input {
            padding: 8px 10px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 14px;
            width: 110px;
        }
    </style>
</head>
<body>
    <h1>Multi-Camera Dashboard</h1>

    <div class="small">D435: <code>{{ d435_base }}</code></div>
    <div class="small">Pi Zero Left: <code>{{ pi_zero_cams["left"] }}</code></div>
    <div class="small">Pi Zero Right: <code>{{ pi_zero_cams["right"] }}</code></div>
    <div class="small">Pi Zero Bottom: <code>{{ pi_zero_cams["bottom"] }}</code></div>

    <div class="card" style="margin: 16px 0;">
        <h3 style="margin-top:0;">Capture Settings</h3>
        <div class="field-row">
            <label for="greenhouse_row">Greenhouse Row</label>
            <input type="number" id="greenhouse_row" value="1" min="1" step="1" />
            <label for="distance_from_start">Distance from Start (m)</label>
            <input type="number" id="distance_from_start" value="0.0" min="0" step="0.1" />
        </div>
    </div>

    <div class="topbar">
        <button class="btn" onclick="captureAll()">Capture All</button>
        <button class="btn" onclick="checkHealth()">Health Check</button>
        <div id="status" class="status">Ready.</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>D435 RGB</h3>
            <img id="d435rgb" src="{{ d435_base }}/stream/rgb.mjpg?ts={{ now_ts }}" />
        </div>
        <div class="card">
            <h3>D435 Depth</h3>
            <img id="d435depth" src="{{ d435_base }}/stream/depth.mjpg?ts={{ now_ts }}" />
        </div>
        <div class="card">
            <h3>Pi Zero Left</h3>
            <img id="pi_left" src="{{ pi_zero_cams['left'] }}/stream/rgb.mjpg?ts={{ now_ts }}" />
        </div>
        <div class="card">
            <h3>Pi Zero Right</h3>
            <img id="pi_right" src="{{ pi_zero_cams['right'] }}/stream/rgb.mjpg?ts={{ now_ts }}" />
        </div>
        <div class="card">
            <h3>Pi Zero Bottom</h3>
            <img id="pi_bottom" src="{{ pi_zero_cams['bottom'] }}/stream/rgb.mjpg?ts={{ now_ts }}" />
        </div>
    </div>

    <div class="card" style="margin-top:20px;">
        <h3>Latest Health / Capture Result</h3>
        <pre id="infoBox">No data yet.</pre>
    </div>

    <script>
        async function captureAll() {
            const status = document.getElementById("status");
            const greenhouse_row = parseInt(document.getElementById("greenhouse_row").value, 10);
            const distanceFromRowStart = parseFloat(document.getElementById("distance_from_start").value);

            if (isNaN(greenhouse_row) || isNaN(distanceFromRowStart)) {
                status.textContent = "Please enter valid row and distance values.";
                return;
            }

            status.textContent = "Capturing all cameras...";
            try {
                const res = await fetch("/capture_all", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ greenhouse_row, distanceFromRowStart }),
                });
                const data = await res.json();
                if (data.ok) {
                    status.textContent = "Saved: " + data.timestamp + " → " + data.save_dir;
                } else {
                    status.textContent = "Capture finished with some errors. Round: " + data.timestamp;
                }
                reloadStreams();
                document.getElementById("infoBox").textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                status.textContent = "Capture error: " + err;
            }
        }

        async function checkHealth() {
            const status = document.getElementById("status");
            status.textContent = "Checking health...";
            try {
                const res = await fetch("/health_all");
                const data = await res.json();
                document.getElementById("infoBox").textContent = JSON.stringify(data, null, 2);
                status.textContent = data.ok ? "Health OK." : "Some devices unavailable.";
            } catch (err) {
                status.textContent = "Health error: " + err;
            }
        }

        function reloadStreams() {
            const ts = Date.now();
            document.getElementById("d435rgb").src = "{{ d435_base }}/stream/rgb.mjpg?ts=" + ts;
            document.getElementById("d435depth").src = "{{ d435_base }}/stream/depth.mjpg?ts=" + ts;
            document.getElementById("pi_left").src = "{{ pi_zero_cams['left'] }}/stream/rgb.mjpg?ts=" + ts;
            document.getElementById("pi_right").src = "{{ pi_zero_cams['right'] }}/stream/rgb.mjpg?ts=" + ts;
            document.getElementById("pi_bottom").src = "{{ pi_zero_cams['bottom'] }}/stream/rgb.mjpg?ts=" + ts;
        }

        window.addEventListener("load", () => {
            checkHealth();
            setInterval(checkHealth, 5000);
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        d435_base=D435_BASE,
        pi_zero_cams=PI_ZERO_CAMS,
        now_ts=int(datetime.now().timestamp()),
    )


@app.route("/health_all")
def health_all():
    out = {"ok": True, "d435": {}, "pi_zeros": {}}

    try:
        r = HTTP.get(f"{D435_BASE}/health", timeout=3)
        out["d435"] = r.json()
    except Exception as e:
        out["ok"] = False
        out["d435"] = {"ok": False, "error": str(e)}

    for cam_name, base_url in PI_ZERO_CAMS.items():
        try:
            r = HTTP.get(f"{base_url}/health", timeout=3)
            out["pi_zeros"][cam_name] = r.json()
        except Exception as e:
            out["ok"] = False
            out["pi_zeros"][cam_name] = {"ok": False, "error": str(e)}

    return jsonify(out)


def capture_d435(round_dir: Path, trigger_at_ms: int):
    r = HTTP.post(
        f"{D435_BASE}/capture",
        json={"trigger_at_ms": trigger_at_ms},
        timeout=25,
    )
    if r.status_code != 200:
        raise RuntimeError(f"D435 capture failed with status {r.status_code}: {r.text}")

    rgb_size = int(r.headers.get("X-RGB-Size", "0"))
    depth_scale = float(r.headers.get("X-Depth-Scale", "0"))
    fx = float(r.headers.get("X-FX", "0"))
    fy = float(r.headers.get("X-FY", "0"))
    cx = float(r.headers.get("X-CX", "0"))
    cy = float(r.headers.get("X-CY", "0"))
    width = int(r.headers.get("X-Width", "0"))
    height = int(r.headers.get("X-Height", "0"))
    d435_ts = r.headers.get("X-Timestamp", "")

    payload = r.content
    if rgb_size <= 0 or rgb_size >= len(payload):
        raise RuntimeError("Invalid D435 payload sizes")

    rgb_bytes = payload[:rgb_size]
    depth_png_bytes = payload[rgb_size:]

    d435_dir = round_dir / "d435"
    d435_dir.mkdir(exist_ok=True)

    rgb_path = d435_dir / "d435_rgb.jpg"
    depth_path = d435_dir / "d435_depth_raw.png"
    npy_path = d435_dir / "d435_depth_raw.npy"
    meta_path = d435_dir / "d435_meta.json"

    rgb_path.write_bytes(rgb_bytes)
    depth_path.write_bytes(depth_png_bytes)

    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    np.save(npy_path, depth_img)

    meta = {
        "camera_name": "d435",
        "timestamp": d435_ts,
        "depth_format": "RS2_FORMAT_Z16 stored as 16-bit PNG",
        "depth_dtype": str(depth_img.dtype) if depth_img is not None else None,
        "depth_shape": list(depth_img.shape) if depth_img is not None else None,
        "depth_scale_m_per_unit": depth_scale,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height},
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "npy_path": str(npy_path),
        "meta_path": str(meta_path),
        "timestamp": d435_ts,
    }


def capture_pi_zero(cam_name: str, base_url: str, round_dir: Path, trigger_at_ms: int):
    r = HTTP.post(
        f"{base_url}/capture",
        json={"trigger_at_ms": trigger_at_ms},
        timeout=25,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{cam_name} capture failed with status {r.status_code}: {r.text}")

    pi_ts = r.headers.get("X-Timestamp", "")
    returned_cam_name = r.headers.get("X-Camera-Name", cam_name)
    width = int(r.headers.get("X-Width", "0"))
    height = int(r.headers.get("X-Height", "0"))

    pi_dir = round_dir / cam_name
    pi_dir.mkdir(exist_ok=True)

    img_path = pi_dir / f"{returned_cam_name}_rgb.jpg"
    meta_path = pi_dir / f"{returned_cam_name}_meta.json"

    img_path.write_bytes(r.content)

    meta = {
        "camera_name": returned_cam_name,
        "timestamp": pi_ts,
        "width": width,
        "height": height,
        "base_url": base_url,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "rgb_path": str(img_path),
        "meta_path": str(meta_path),
        "timestamp": pi_ts,
        "camera_name": returned_cam_name,
    }


def _fmt_ts(dt: datetime) -> str:
    """Format datetime as 2026-03-19T06:14:02.973Z (UTC, 3 decimal places)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def upload_rgb_pair(saved: dict, greenhouse_row: int, distance: float, timestamp: str):
    """POST left + right Pi Zero images to /uploadData."""
    files = []
    for side in ("left", "right"):
        info = saved["pi_zeros"].get(side)
        if info and info.get("rgb_path"):
            p = Path(info["rgb_path"])
            files.append(("images", (p.name, p.read_bytes(), "image/jpeg")))

    if not files:
        raise RuntimeError("No left/right images available to upload")

    r = HTTP.post(
        f"{API_BASE}/uploadData",
        files=files,
        data={"timestamp": timestamp, "greenhouse_row": greenhouse_row, "distanceFromRowStart": distance},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def upload_d435_data(saved: dict, greenhouse_row: int, distance: float, timestamp: str):
    """POST D435 RGB + depth .npy with intrinsics to /uploadData."""
    d435_info = saved.get("d435")
    if not d435_info:
        raise RuntimeError("No D435 capture data available to upload")

    meta = json.loads(Path(d435_info["meta_path"]).read_text())
    rgb_path = Path(d435_info["rgb_path"])
    npy_path = Path(d435_info["npy_path"])

    r = HTTP.post(
        f"{API_BASE}/uploadData",
        files=[
            ("images",      (rgb_path.name, rgb_path.read_bytes(), "image/jpeg")),
            ("depth_image", (npy_path.name, npy_path.read_bytes(), "application/octet-stream")),
        ],
        data={
            "timestamp":           timestamp,
            "greenhouse_row":      greenhouse_row,
            "distanceFromRowStart": distance,
            "fx":          meta["intrinsics"]["fx"],
            "fy":          meta["intrinsics"]["fy"],
            "depth_scale": meta["depth_scale_m_per_unit"],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


@app.route("/capture_all", methods=["POST"])
def capture_all():
    body = request.get_json(silent=True) or {}
    greenhouse_row = int(body.get("greenhouse_row", 1))
    distance = float(body.get("distanceFromRowStart", 0.0))

    now = datetime.now(timezone.utc)
    round_timestamp   = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]      # safe for dir names
    api_ts_rgb  = _fmt_ts(now)                                      # e.g. 2026-03-19T06:14:02.973Z
    api_ts_d435 = _fmt_ts(now + timedelta(seconds=1))               # +1 s to avoid same-key overwrite

    round_dir = SAVE_DIR / round_timestamp
    round_dir.mkdir(parents=True, exist_ok=True)

    trigger_at_ms = (time.time_ns() // 1_000_000) + 1500

    result = {
        "ok": True,
        "timestamp": round_timestamp,
        "trigger_at_ms": trigger_at_ms,
        "save_dir": str(round_dir),
        "greenhouse_row": greenhouse_row,
        "distanceFromRowStart": distance,
        "saved": {"d435": None, "pi_zeros": {}},
        "uploads": {},
    }

    # --- Phase 1: capture all cameras in parallel ---
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures[executor.submit(capture_d435, round_dir, trigger_at_ms)] = ("d435", "d435")
        for cam_name, base_url in PI_ZERO_CAMS.items():
            futures[executor.submit(capture_pi_zero, cam_name, base_url, round_dir, trigger_at_ms)] = ("pi_zero", cam_name)

        for future in as_completed(futures):
            kind, name = futures[future]
            try:
                saved_info = future.result()
                if kind == "d435":
                    result["saved"]["d435"] = saved_info
                else:
                    result["saved"]["pi_zeros"][name] = saved_info
            except Exception as e:
                result["ok"] = False
                result[f"{name}_error"] = str(e)

    # --- Phase 2: two sequential uploads with offset timestamps ---
    try:
        result["uploads"]["rgb_pair"] = upload_rgb_pair(
            result["saved"], greenhouse_row, distance, api_ts_rgb
        )
    except Exception as e:
        result["ok"] = False
        result["uploads"]["rgb_pair_error"] = str(e)

    try:
        result["uploads"]["d435"] = upload_d435_data(
            result["saved"], greenhouse_row, distance, api_ts_d435
        )
    except Exception as e:
        result["ok"] = False
        result["uploads"]["d435_error"] = str(e)

    (round_dir / "capture_summary.json").write_text(json.dumps(result, indent=2))

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True)
