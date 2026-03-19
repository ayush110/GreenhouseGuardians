from flask import Flask, render_template_string, jsonify
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np

app = Flask(__name__)

D435_BASE = "http://172.20.10.2:8000"

PI_ZERO_CAMS = {
    "right":  "http://172.20.10.4:8001",
    "bottom": "http://172.20.10.5:8001",
    "left":   "http://172.20.10.7:8001",
}

SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

# How far in the future (ms) to schedule the synchronized capture.
# Gives all cameras time to receive the request and wait.
TRIGGER_LEAD_MS = 800

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
        .btn:disabled { background: #93c5fd; cursor: not-allowed; }
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
        code { background: #eef2ff; padding: 2px 6px; border-radius: 6px; }
        .small { color: #6b7280; font-size: 14px; margin-bottom: 4px; }
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
    <div class="small">Pi Zero Left: <code>{{ pi_zero_cams["left"] }}</code></div>
    <div class="small">Pi Zero Right: <code>{{ pi_zero_cams["right"] }}</code></div>
    <div class="small">Pi Zero Bottom: <code>{{ pi_zero_cams["bottom"] }}</code></div>

    <div class="topbar">
        <button class="btn" id="captureBtn" onclick="captureAll()">Capture All</button>
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
        <h3>Latest Health</h3>
        <pre id="healthBox">No data yet.</pre>
    </div>

    <script>
        async function captureAll() {
            const btn = document.getElementById("captureBtn");
            const status = document.getElementById("status");
            btn.disabled = true;
            status.textContent = "Capturing all cameras...";
            try {
                const res = await fetch("/capture_all", { method: "POST" });
                const data = await res.json();
                if (data.ok) {
                    status.textContent = "Saved: " + data.timestamp;
                } else {
                    status.textContent = "Partial capture, saved: " + data.timestamp;
                }
                reloadStreams();
            } catch (err) {
                status.textContent = "Capture error: " + err;
            } finally {
                btn.disabled = false;
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
                status.textContent = data.ok ? "Health OK." : "Some devices unavailable.";
            } catch (err) {
                status.textContent = "Health error: " + err;
            }
        }

        function reloadStreams() {
            const ts = Date.now();
            document.getElementById("d435rgb").src    = "{{ d435_base }}/stream/rgb.mjpg?ts=" + ts;
            document.getElementById("d435depth").src  = "{{ d435_base }}/stream/depth.mjpg?ts=" + ts;
            document.getElementById("pi_left").src    = "{{ pi_zero_cams['left'] }}/stream/rgb.mjpg?ts=" + ts;
            document.getElementById("pi_right").src   = "{{ pi_zero_cams['right'] }}/stream/rgb.mjpg?ts=" + ts;
            document.getElementById("pi_bottom").src  = "{{ pi_zero_cams['bottom'] }}/stream/rgb.mjpg?ts=" + ts;
        }

        window.addEventListener("load", () => {
            checkHealth();
            setInterval(checkHealth, 10000);
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

    def fetch_health(url):
        return requests.get(f"{url}/health", timeout=3).json()

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            "d435": ex.submit(fetch_health, D435_BASE),
            **{name: ex.submit(fetch_health, url) for name, url in PI_ZERO_CAMS.items()},
        }
        for key, future in futures.items():
            try:
                result = future.result()
                if key == "d435":
                    out["d435"] = result
                else:
                    out["pi_zeros"][key] = result
            except Exception as e:
                out["ok"] = False
                if key == "d435":
                    out["d435"] = {"ok": False, "error": str(e)}
                else:
                    out["pi_zeros"][key] = {"ok": False, "error": str(e)}

    return jsonify(out)


def _capture_d435(round_dir, timestamp, trigger_at_ms):
    body = {"trigger_at_ms": trigger_at_ms}
    r = requests.post(f"{D435_BASE}/capture", json=body, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"D435 capture failed: HTTP {r.status_code}")

    rgb_size   = int(r.headers.get("X-RGB-Size", "0"))
    depth_scale = float(r.headers.get("X-Depth-Scale", "0"))
    fx         = float(r.headers.get("X-FX", "0"))
    fy         = float(r.headers.get("X-FY", "0"))
    cx         = float(r.headers.get("X-CX", "0"))
    cy         = float(r.headers.get("X-CY", "0"))
    width      = int(r.headers.get("X-Width", "0"))
    height     = int(r.headers.get("X-Height", "0"))
    frame_ctr  = int(r.headers.get("X-Frame-Counter", "0"))
    d435_ts    = r.headers.get("X-Timestamp", timestamp)

    payload = r.content
    if rgb_size <= 0 or rgb_size >= len(payload):
        raise RuntimeError("Invalid D435 payload")

    rgb_bytes       = payload[:rgb_size]
    depth_png_bytes = payload[rgb_size:]

    d435_dir = round_dir / "d435"
    d435_dir.mkdir(exist_ok=True)

    (d435_dir / "rgb.jpg").write_bytes(rgb_bytes)
    (d435_dir / "depth_raw.png").write_bytes(depth_png_bytes)

    depth_img = cv2.imread(str(d435_dir / "depth_raw.png"), cv2.IMREAD_UNCHANGED)
    np.save(str(d435_dir / "depth_raw.npy"), depth_img)

    meta = {
        "timestamp": d435_ts,
        "frame_counter": frame_ctr,
        "depth_format": "RS2_FORMAT_Z16 stored as 16-bit PNG",
        "depth_dtype": str(depth_img.dtype) if depth_img is not None else None,
        "depth_shape": list(depth_img.shape) if depth_img is not None else None,
        "depth_scale_m_per_unit": depth_scale,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height},
    }
    (d435_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "rgb_path":   str(d435_dir / "rgb.jpg"),
        "depth_path": str(d435_dir / "depth_raw.png"),
        "npy_path":   str(d435_dir / "depth_raw.npy"),
        "meta_path":  str(d435_dir / "meta.json"),
    }


def _capture_pi_zero(cam_name, base_url, round_dir, timestamp, trigger_at_ms):
    r = requests.post(
        f"{base_url}/capture",
        params={"trigger_at_ms": trigger_at_ms},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{cam_name} capture failed: HTTP {r.status_code}")

    pi_ts      = r.headers.get("X-Timestamp", timestamp)
    frame_ctr  = int(r.headers.get("X-Frame-Counter", "0"))
    cam_label  = r.headers.get("X-Camera-Name", cam_name)
    width      = int(r.headers.get("X-Width", "0"))
    height     = int(r.headers.get("X-Height", "0"))

    pi_dir = round_dir / cam_name
    pi_dir.mkdir(exist_ok=True)

    (pi_dir / "rgb.jpg").write_bytes(r.content)
    meta = {
        "timestamp":    pi_ts,
        "frame_counter": frame_ctr,
        "camera_name":  cam_label,
        "width":        width,
        "height":       height,
        "base_url":     base_url,
    }
    (pi_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "rgb_path":  str(pi_dir / "rgb.jpg"),
        "meta_path": str(pi_dir / "meta.json"),
    }


@app.route("/capture_all", methods=["POST"])
def capture_all():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    round_dir = SAVE_DIR / timestamp
    round_dir.mkdir(parents=True, exist_ok=True)

    trigger_at_ms = int(time.time() * 1000) + TRIGGER_LEAD_MS

    result = {"ok": True, "timestamp": timestamp, "saved": {}}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            "d435": ex.submit(_capture_d435, round_dir, timestamp, trigger_at_ms),
            **{
                name: ex.submit(_capture_pi_zero, name, url, round_dir, timestamp, trigger_at_ms)
                for name, url in PI_ZERO_CAMS.items()
            },
        }
        for key, future in futures.items():
            try:
                result["saved"][key] = future.result()
            except Exception as e:
                result["ok"] = False
                result[f"{key}_error"] = str(e)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
