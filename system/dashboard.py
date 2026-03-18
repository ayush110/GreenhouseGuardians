from flask import Flask, render_template_string, jsonify, Response
import requests
from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np

app = Flask(__name__)

D435_BASE = "http://192.168.1.20:8000"     # change
PI_ZERO_BASE = "http://192.168.1.21:8001"  # change

SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

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
        .btn:hover {
            background: #1d4ed8;
        }
        .status {
            font-weight: 600;
        }
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
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #e5e7eb;
        }
    </style>
</head>
<body>
    <h1>Multi-Camera Dashboard</h1>
    <div class="small">D435: <code>{{ d435_base }}</code></div>
    <div class="small">Pi Zero: <code>{{ pi_zero_base }}</code></div>

    <div class="topbar">
        <button class="btn" onclick="captureAll()">Capture All</button>
        <button class="btn" onclick="checkHealth()">Health Check</button>
        <div id="status" class="status">Ready.</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>D435 RGB</h3>
            <img src="/proxy/d435/rgb.mjpg" />
        </div>
        <div class="card">
            <h3>D435 Depth</h3>
            <img src="/proxy/d435/depth.mjpg" />
        </div>
        <div class="card">
            <h3>Pi Zero RGB</h3>
            <img src="/proxy/pizero/rgb.mjpg" />
        </div>
    </div>

    <div class="card" style="margin-top:20px;">
        <h3>Latest Health</h3>
        <pre id="healthBox">No data yet.</pre>
    </div>

    <script>
        async function captureAll() {
            const status = document.getElementById("status");
            status.textContent = "Capturing all cameras...";
            try {
                const res = await fetch("/capture_all", { method: "POST" });
                const data = await res.json();
                if (data.ok) {
                    status.textContent = "Saved round: " + data.timestamp;
                } else {
                    status.textContent = "Capture failed: " + (data.error || "unknown");
                }
            } catch (err) {
                status.textContent = "Capture error: " + err;
            }
        }

        async function checkHealth() {
            const status = document.getElementById("status");
            const healthBox = document.getElementById("healthBox");
            status.textContent = "Checking health...";
            try {
                const res = await fetch("/health_all");
                const data = await res.json();
                healthBox.textContent = JSON.stringify(data, null, 2);
                if (data.ok) {
                    status.textContent = "Health OK.";
                } else {
                    status.textContent = "Some devices unavailable.";
                }
            } catch (err) {
                status.textContent = "Health error: " + err;
            }
        }
    </script>
</body>
</html>
"""

def stream_proxy(url: str, content_type: str):
    upstream = requests.get(url, stream=True, timeout=(5, 60))
    headers = {
        "Content-Type": content_type,
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return Response(upstream.iter_content(chunk_size=4096), headers=headers)

@app.route("/")
def index():
    return render_template_string(HTML, d435_base=D435_BASE, pi_zero_base=PI_ZERO_BASE)

@app.route("/proxy/d435/rgb.mjpg")
def proxy_d435_rgb():
    return stream_proxy(f"{D435_BASE}/stream/rgb.mjpg", "multipart/x-mixed-replace; boundary=frame")

@app.route("/proxy/d435/depth.mjpg")
def proxy_d435_depth():
    return stream_proxy(f"{D435_BASE}/stream/depth.mjpg", "multipart/x-mixed-replace; boundary=frame")

@app.route("/proxy/pizero/rgb.mjpg")
def proxy_pizero_rgb():
    return stream_proxy(f"{PI_ZERO_BASE}/stream/rgb.mjpg", "multipart/x-mixed-replace; boundary=frame")

@app.route("/health_all")
def health_all():
    out = {"ok": True}

    try:
        r = requests.get(f"{D435_BASE}/health", timeout=5)
        out["d435"] = r.json()
    except Exception as e:
        out["ok"] = False
        out["d435"] = {"ok": False, "error": str(e)}

    try:
        r = requests.get(f"{PI_ZERO_BASE}/health", timeout=5)
        out["pi_zero"] = r.json()
    except Exception as e:
        out["ok"] = False
        out["pi_zero"] = {"ok": False, "error": str(e)}

    return jsonify(out)

@app.route("/capture_all", methods=["POST"])
def capture_all():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    round_dir = SAVE_DIR / timestamp
    round_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ok": True,
        "timestamp": timestamp,
        "saved": {}
    }

    # Capture D435
    try:
        r = requests.post(f"{D435_BASE}/capture", timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"D435 capture failed with status {r.status_code}")

        rgb_size = int(r.headers.get("X-RGB-Size", "0"))
        frame_counter = int(r.headers.get("X-Frame-Counter", "0"))
        depth_scale = float(r.headers.get("X-Depth-Scale", "0"))
        fx = float(r.headers.get("X-FX", "0"))
        fy = float(r.headers.get("X-FY", "0"))
        cx = float(r.headers.get("X-CX", "0"))
        cy = float(r.headers.get("X-CY", "0"))
        width = int(r.headers.get("X-Width", "0"))
        height = int(r.headers.get("X-Height", "0"))
        d435_ts = r.headers.get("X-Timestamp", timestamp)

        payload = r.content
        if rgb_size <= 0 or rgb_size >= len(payload):
            raise RuntimeError("Invalid D435 payload sizes")

        rgb_bytes = payload[:rgb_size]
        depth_png_bytes = payload[rgb_size:]

        d435_dir = round_dir / "d435"
        d435_dir.mkdir(exist_ok=True)

        rgb_path = d435_dir / "rgb.jpg"
        depth_path = d435_dir / "depth_raw.png"
        npy_path = d435_dir / "depth_raw.npy"
        meta_path = d435_dir / "meta.json"

        rgb_path.write_bytes(rgb_bytes)
        depth_path.write_bytes(depth_png_bytes)

        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        np.save(npy_path, depth_img)

        meta = {
            "timestamp": d435_ts,
            "frame_counter": frame_counter,
            "depth_format": "RS2_FORMAT_Z16 stored as 16-bit PNG",
            "depth_dtype": str(depth_img.dtype),
            "depth_shape": list(depth_img.shape),
            "depth_scale_m_per_unit": depth_scale,
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height
            }
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        result["saved"]["d435"] = {
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "npy_path": str(npy_path),
            "meta_path": str(meta_path),
        }

    except Exception as e:
        result["ok"] = False
        result["d435_error"] = str(e)

    # Capture Pi Zero
    try:
        r = requests.post(f"{PI_ZERO_BASE}/capture", timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Pi Zero capture failed with status {r.status_code}")

        pi_ts = r.headers.get("X-Timestamp", timestamp)
        frame_counter = int(r.headers.get("X-Frame-Counter", "0"))
        cam_name = r.headers.get("X-Camera-Name", "pi_zero")
        width = int(r.headers.get("X-Width", "0"))
        height = int(r.headers.get("X-Height", "0"))

        pi_dir = round_dir / cam_name
        pi_dir.mkdir(exist_ok=True)

        img_path = pi_dir / "rgb.jpg"
        meta_path = pi_dir / "meta.json"

        img_path.write_bytes(r.content)

        meta = {
            "timestamp": pi_ts,
            "frame_counter": frame_counter,
            "camera_name": cam_name,
            "width": width,
            "height": height
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        result["saved"]["pi_zero"] = {
            "rgb_path": str(img_path),
            "meta_path": str(meta_path),
        }

    except Exception as e:
        result["ok"] = False
        result["pi_zero_error"] = str(e)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True)