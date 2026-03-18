from flask import Flask, render_template_string, jsonify, Response
import requests
from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np

app = Flask(__name__)

PI4_BASE = "http://172.20.10.2:8000"   # change this
SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

HTML = """
<!doctype html>
<html>
<head>
    <title>D435 Dashboard</title>
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
    <h1>D435 Local Dashboard</h1>
    <div class="small">Pi endpoint: <code>{{ pi4_base }}</code></div>

    <div class="topbar">
        <button class="btn" onclick="captureNow()">Capture</button>
        <button class="btn" onclick="checkHealth()">Health Check</button>
        <div id="status" class="status">Ready.</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>RGB Preview</h3>
            <img src="/proxy/stream/rgb.mjpg" />
        </div>
        <div class="card">
            <h3>Depth Preview</h3>
            <img src="/proxy/stream/depth.mjpg" />
        </div>
    </div>

    <div class="card" style="margin-top:20px;">
        <h3>Latest Health</h3>
        <pre id="healthBox">No data yet.</pre>
    </div>

    <script>
        async function captureNow() {
            const status = document.getElementById("status");
            status.textContent = "Capturing...";
            try {
                const res = await fetch("/capture", { method: "POST" });
                const data = await res.json();
                if (data.ok) {
                    status.textContent = "Saved: " + data.timestamp;
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
                const res = await fetch("/health");
                const data = await res.json();
                healthBox.textContent = JSON.stringify(data, null, 2);
                if (data.ok) {
                    status.textContent = "Pi reachable.";
                } else {
                    status.textContent = "Health failed.";
                }
            } catch (err) {
                status.textContent = "Health error: " + err;
            }
        }
    </script>
</body>
</html>
"""

def stream_proxy(path: str, content_type: str):
    upstream = requests.get(f"{PI4_BASE}{path}", stream=True, timeout=(5, 60))
    headers = {
        "Content-Type": content_type,
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return Response(upstream.iter_content(chunk_size=4096), headers=headers)

@app.route("/")
def index():
    return render_template_string(HTML, pi4_base=PI4_BASE)

@app.route("/health")
def health():
    try:
        r = requests.get(f"{PI4_BASE}/health", timeout=5)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/proxy/stream/rgb.mjpg")
def proxy_rgb_stream():
    return stream_proxy("/stream/rgb.mjpg", "multipart/x-mixed-replace; boundary=frame")

@app.route("/proxy/stream/depth.mjpg")
def proxy_depth_stream():
    return stream_proxy("/stream/depth.mjpg", "multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    try:
        r = requests.post(f"{PI4_BASE}/capture", timeout=15)

        if r.status_code != 200:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"ok": False, "error": f"Pi returned status {r.status_code}"}), 500

        ts = r.headers.get("X-Timestamp", datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
        frame_counter = int(r.headers.get("X-Frame-Counter", "0"))
        rgb_size = int(r.headers.get("X-RGB-Size", "0"))
        depth_scale = float(r.headers.get("X-Depth-Scale", "0"))
        fx = float(r.headers.get("X-FX", "0"))
        fy = float(r.headers.get("X-FY", "0"))
        cx = float(r.headers.get("X-CX", "0"))
        cy = float(r.headers.get("X-CY", "0"))
        width = int(r.headers.get("X-Width", "0"))
        height = int(r.headers.get("X-Height", "0"))

        payload = r.content
        if rgb_size <= 0 or rgb_size >= len(payload):
            return jsonify({"ok": False, "error": "Invalid payload sizes"}), 500

        rgb_bytes = payload[:rgb_size]
        depth_png_bytes = payload[rgb_size:]

        round_dir = SAVE_DIR / ts
        round_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = round_dir / "rgb.jpg"
        depth_path = round_dir / "depth_raw.png"
        npy_path = round_dir / "depth_raw.npy"
        meta_path = round_dir / "meta.json"

        rgb_path.write_bytes(rgb_bytes)
        depth_path.write_bytes(depth_png_bytes)

        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            return jsonify({"ok": False, "error": "Failed to reload saved depth PNG"}), 500

        np.save(npy_path, depth_img)

        meta = {
            "timestamp": ts,
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

        return jsonify({
            "ok": True,
            "timestamp": ts,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "npy_path": str(npy_path),
            "meta_path": str(meta_path),
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True)