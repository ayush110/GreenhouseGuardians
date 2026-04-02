from unittest import result

from flask import Flask, render_template_string, jsonify, request
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io

app = Flask(__name__)

D435_BASE = "http://172.20.10.3:8000"

PI_ZERO_CAMS = {
    "left":  "http://172.20.10.5:8001",
    "right": "http://172.20.10.4:8001",
}

API_BASE = "https://greenhouseguardiansorg-greenhouseguardians.hf.space/api/upload"

SAVE_DIR = Path.home() / "multi_camera_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HTTP = requests.Session()
HTTP.mount("http://",  requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))
HTTP.mount("https://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))

HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Greenhouse Guardians</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d0f14;
            color: #e2e8f0;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* ── Top bar ── */
        .topbar {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 16px;
            height: 52px;
            background: #13151e;
            border-bottom: 1px solid #1f2235;
            flex-shrink: 0;
        }

        .logo {
            font-size: 14px;
            font-weight: 700;
            color: #4ade80;
            letter-spacing: 0.02em;
            white-space: nowrap;
            margin-right: 4px;
        }

        .sep {
            width: 1px;
            height: 20px;
            background: #1f2235;
            flex-shrink: 0;
        }

        .field {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .field label {
            font-size: 11px;
            font-weight: 500;
            color: #64748b;
            white-space: nowrap;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .field input {
            background: #1a1d2b;
            border: 1px solid #2a2e45;
            color: #e2e8f0;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 13px;
            width: 68px;
            transition: border-color 0.15s;
        }
        .field input:focus {
            outline: none;
            border-color: #4ade80;
        }

        .btn {
            padding: 6px 16px;
            border-radius: 6px;
            border: none;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: 0.03em;
            transition: opacity 0.15s, transform 0.1s;
            white-space: nowrap;
        }
        .btn:active { transform: scale(0.97); }
        .btn-capture { background: #4ade80; color: #0d0f14; }
        .btn-capture:hover { background: #22c55e; }
        .btn-health {
            background: transparent;
            color: #64748b;
            border: 1px solid #2a2e45;
        }
        .btn-health:hover { border-color: #4ade80; color: #4ade80; }

        .status-wrap {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 7px;
            overflow: hidden;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #4ade80;
            flex-shrink: 0;
            transition: background 0.3s;
        }
        .status-dot.busy { background: #fbbf24; }
        .status-dot.error { background: #f87171; }
        #statusText {
            font-size: 12px;
            color: #475569;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 340px;
        }

        /* ── Stream grid ── */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 6px;
            padding: 6px;
            flex: 1;
            min-height: 0;
        }

        .card {
            background: #13151e;
            border-radius: 8px;
            border: 1px solid #1f2235;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 5px 10px;
            flex-shrink: 0;
            border-bottom: 1px solid #1f2235;
        }
        .card-title {
            font-size: 10px;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .pill {
            font-size: 9px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 20px;
            background: #1a2e1f;
            color: #4ade80;
            letter-spacing: 0.04em;
        }

        .card img {
            flex: 1;
            width: 100%;
            min-height: 0;
            object-fit: contain;
            display: block;
            background: #0d0f14;
        }

        /* ── Toast notification ── */
        #toast {
            position: fixed;
            bottom: 18px;
            left: 50%;
            transform: translateX(-50%) translateY(20px);
            background: #1a1d2b;
            border: 1px solid #2a2e45;
            color: #e2e8f0;
            font-size: 12px;
            padding: 8px 18px;
            border-radius: 20px;
            opacity: 0;
            transition: opacity 0.25s, transform 0.25s;
            pointer-events: none;
            white-space: nowrap;
            z-index: 100;
        }
        #toast.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
        #toast.success { border-color: #4ade80; color: #4ade80; }
        #toast.error   { border-color: #f87171; color: #f87171; }
    </style>
</head>
<body>

<div class="topbar">
    <span class="logo">Greenhouse Guardians</span>
    <div class="sep"></div>
    <div class="field">
        <label>Row</label>
        <input type="number" id="greenhouse_row" value="1" min="1" step="1" />
    </div>
    <div class="field">
        <label>Distance (m)</label>
        <input type="number" id="distance_from_start" value="0.0" min="0" step="0.1" />
    </div>
    <div class="sep"></div>
    <button class="btn btn-capture" onclick="captureAll()">Capture All</button>
    <button class="btn btn-health"  onclick="checkHealth()">Health</button>
    <div class="status-wrap">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">Ready</span>
    </div>
</div>

<div class="grid">
    <div class="card">
        <div class="card-header">
            <span class="card-title">D435 &mdash; RGB</span>
            <span class="pill" id="pill-d435rgb">LIVE</span>
        </div>
        <img id="d435rgb" src="{{ d435_base }}/stream/rgb.mjpg?ts={{ now_ts }}" />
    </div>
    <div class="card">
        <div class="card-header">
            <span class="card-title">D435 &mdash; Depth</span>
            <span class="pill" id="pill-d435depth">LIVE</span>
        </div>
        <img id="d435depth" src="{{ d435_base }}/stream/depth.mjpg?ts={{ now_ts }}" />
    </div>
    <div class="card">
        <div class="card-header">
            <span class="card-title">Pi Zero &mdash; Left</span>
            <span class="pill" id="pill-left">LIVE</span>
        </div>
        <img id="pi_left" src="{{ pi_zero_cams['left'] }}/stream/rgb.mjpg?ts={{ now_ts }}" />
    </div>
    <div class="card">
        <div class="card-header">
            <span class="card-title">Pi Zero &mdash; Right</span>
            <span class="pill" id="pill-right">LIVE</span>
        </div>
        <img id="pi_right" src="{{ pi_zero_cams['right'] }}/stream/rgb.mjpg?ts={{ now_ts }}" />
    </div>
</div>

<div id="toast"></div>

<script>
    function setStatus(text, state) {
        document.getElementById("statusText").textContent = text;
        const dot = document.getElementById("statusDot");
        dot.className = "status-dot" + (state ? " " + state : "");
    }

    function showToast(text, type) {
        const t = document.getElementById("toast");
        t.textContent = text;
        t.className = "show " + (type || "");
        clearTimeout(t._timer);
        t._timer = setTimeout(() => { t.className = ""; }, 3500);
    }

    function setPill(id, online) {
        const el = document.getElementById("pill-" + id);
        if (!el) return;
        el.textContent = online ? "LIVE" : "OFF";
        el.style.background = online ? "#1a2e1f" : "#2e1a1a";
        el.style.color       = online ? "#4ade80" : "#f87171";
    }

    async function captureAll() {
        const greenhouse_row = parseInt(document.getElementById("greenhouse_row").value, 10);
        const distanceFromRowStart = parseFloat(document.getElementById("distance_from_start").value);
        if (isNaN(greenhouse_row) || isNaN(distanceFromRowStart)) {
            showToast("Enter valid row and distance values.", "error");
            return;
        }
        setStatus("Capturing…", "busy");
        try {
            const res = await fetch("/capture_all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ greenhouse_row, distanceFromRowStart }),
            });
            const data = await res.json();
            if (data.ok) {
                setStatus("Captured · Row " + greenhouse_row + " · " + distanceFromRowStart + " m", "");
                showToast("Capture saved — " + data.timestamp, "success");
            } else {
                setStatus("Capture completed with errors", "error");
                showToast("Some captures failed — check logs", "error");
            }
            reloadStreams();
        } catch (err) {
            setStatus("Capture failed", "error");
            showToast("Error: " + err, "error");
        }
    }

    async function checkHealth() {
        setStatus("Checking health…", "busy");
        try {
            const res = await fetch("/health_all");
            const data = await res.json();
            setPill("d435rgb",  data.d435?.ok ?? false);
            setPill("d435depth", data.d435?.ok ?? false);
            setPill("left",  data.pi_zeros?.left?.ok  ?? false);
            setPill("right", data.pi_zeros?.right?.ok ?? false);
            if (data.ok) {
                setStatus("All devices online", "");
            } else {
                const offline = [];
                if (!data.d435?.ok) offline.push("D435");
                if (!data.pi_zeros?.left?.ok)  offline.push("Left");
                if (!data.pi_zeros?.right?.ok) offline.push("Right");
                setStatus("Offline: " + offline.join(", "), "error");
            }
        } catch (err) {
            setStatus("Health check failed", "error");
        }
    }

    function reloadStreams() {
        const ts = Date.now();
        document.getElementById("d435rgb").src   = "{{ d435_base }}/stream/rgb.mjpg?ts="   + ts;
        document.getElementById("d435depth").src = "{{ d435_base }}/stream/depth.mjpg?ts=" + ts;
        document.getElementById("pi_left").src   = "{{ pi_zero_cams['left'] }}/stream/rgb.mjpg?ts="  + ts;
        document.getElementById("pi_right").src  = "{{ pi_zero_cams['right'] }}/stream/rgb.mjpg?ts=" + ts;
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


def capture_d435(round_dir: Path):
    r = HTTP.post(
        f"{D435_BASE}/capture",
        timeout=25,
    )
    if r.status_code != 200:
        raise RuntimeError(f"D435 capture failed with status {r.status_code}: {r.text}")

    rgb_size = int(r.headers.get("X-RGB-Size", "0"))
    depth_raw_size = int(r.headers.get("X-Depth-Raw-Size", "0"))
    depth_scale = float(r.headers.get("X-Depth-Scale", "0"))
    fx = float(r.headers.get("X-FX", "0"))
    fy = float(r.headers.get("X-FY", "0"))
    cx = float(r.headers.get("X-CX", "0"))
    cy = float(r.headers.get("X-CY", "0"))
    width = int(r.headers.get("X-Width", "0"))
    height = int(r.headers.get("X-Height", "0"))
    d435_ts = r.headers.get("X-Timestamp", "")

    payload = r.content
    expected_size = rgb_size + depth_raw_size

    if rgb_size <= 0 or depth_raw_size <= 0:
        raise RuntimeError(
            f"Invalid D435 payload sizes: rgb_size={rgb_size}, depth_raw_size={depth_raw_size}"
        )

    if len(payload) != expected_size:
        raise RuntimeError(
            f"Unexpected D435 payload length: got {len(payload)}, expected {expected_size}"
        )

    rgb_bytes = payload[:rgb_size]
    depth_raw_bytes = payload[rgb_size:rgb_size + depth_raw_size]

    expected_depth_bytes = width * height * 2  # uint16 depth
    if depth_raw_size != expected_depth_bytes:
        raise RuntimeError(
            f"Depth byte size mismatch: got {depth_raw_size}, expected {expected_depth_bytes}"
        )

    depth_img = np.frombuffer(depth_raw_bytes, dtype=np.uint16).reshape((height, width))

    npy_buffer = io.BytesIO()
    np.save(npy_buffer, depth_img)
    npy_bytes = npy_buffer.getvalue()

    d435_dir = round_dir / "d435"
    d435_dir.mkdir(exist_ok=True)

    rgb_path = d435_dir / "d435_rgb.jpg"
    raw_depth_path = d435_dir / "d435_depth_raw.bin"
    npy_path = d435_dir / "d435_depth_raw.npy"
    meta_path = d435_dir / "d435_meta.json"

    # Optional local archive/debug copies
    rgb_path.write_bytes(rgb_bytes)
    raw_depth_path.write_bytes(depth_raw_bytes)
    npy_path.write_bytes(npy_bytes)

    meta = {
        "camera_name": "d435",
        "timestamp": d435_ts,
        "depth_format": "RS2_FORMAT_Z16 raw uint16 bytes",
        "depth_dtype": str(depth_img.dtype),
        "depth_shape": list(depth_img.shape),
        "depth_scale_m_per_unit": depth_scale,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(raw_depth_path),   # kept for compatibility with summary/debug
        "npy_path": str(npy_path),
        "meta_path": str(meta_path),
        "timestamp": d435_ts,
        # extra in-memory values if you want to upload without rereading files later
        "rgb_bytes": rgb_bytes,
        "npy_bytes": npy_bytes,
        "depth_raw_bytes": depth_raw_bytes,
        "intrinsics": meta["intrinsics"],
        "depth_scale_m_per_unit": depth_scale,
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

    if "intrinsics" in d435_info and "depth_scale_m_per_unit" in d435_info:
        intrinsics = d435_info["intrinsics"]
        depth_scale = d435_info["depth_scale_m_per_unit"]
    else:
        meta = json.loads(Path(d435_info["meta_path"]).read_text())
        intrinsics = meta["intrinsics"]
        depth_scale = meta["depth_scale_m_per_unit"]

    rgb_name = Path(d435_info["rgb_path"]).name
    npy_name = Path(d435_info["npy_path"]).name

    rgb_bytes = d435_info.get("rgb_bytes")
    if rgb_bytes is None:
        rgb_bytes = Path(d435_info["rgb_path"]).read_bytes()

    npy_bytes = d435_info.get("npy_bytes")
    if npy_bytes is None:
        npy_bytes = Path(d435_info["npy_path"]).read_bytes()

    r = HTTP.post(
        f"{API_BASE}/uploadData",
        files=[
            ("images", (rgb_name, rgb_bytes, "image/jpeg")),
            ("depth_image", (npy_name, npy_bytes, "application/octet-stream")),
        ],
        data={
            "timestamp": timestamp,
            "greenhouse_row": greenhouse_row,
            "distanceFromRowStart": distance,
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "depth_scale": depth_scale,
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
    round_timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    api_ts_d435 = _fmt_ts(now)

    round_dir = SAVE_DIR / round_timestamp
    round_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ok": True,
        "timestamp": round_timestamp,
        "save_dir": str(round_dir),
        "greenhouse_row": greenhouse_row,
        "distanceFromRowStart": distance,
        "saved": {"d435": None},
        "uploads": {},
    }

    try:
        result["saved"]["d435"] = capture_d435(round_dir)
    except Exception as e:
        result["ok"] = False
        result["d435_error"] = str(e)

    if result["saved"]["d435"] is not None:
        try:
            result["uploads"]["d435"] = upload_d435_data(
                result["saved"], greenhouse_row, distance, api_ts_d435
            )
        except Exception as e:
            result["ok"] = False
            result["uploads"]["d435_error"] = str(e)

    summary_result = strip_binary_fields(result)
    (round_dir / "capture_summary.json").write_text(json.dumps(summary_result, indent=2))
    return jsonify(summary_result)

def strip_binary_fields(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, bytes):
                out[k] = f"<bytes:{len(v)}>"
            else:
                out[k] = strip_binary_fields(v)
        return out
    if isinstance(obj, list):
        return [strip_binary_fields(x) for x in obj]
    return obj

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True)
