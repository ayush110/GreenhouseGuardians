#!/usr/bin/env python3
"""
viewer.py  —  Multi-camera WebRTC dashboard
────────────────────────────────────────────
Drop-in replacement for dashboard.py.

WHAT CHANGED: live-preview transport only.
  - Pi Zero left / right  →  WebRTC <video> elements (H.264)
    Browser connects directly to each Pi's POST /offer endpoint.
  - D435 RGB / Depth      →  MJPEG <img> tags (C++ server unchanged)

WHAT IS THE SAME:
  GET  /health_all    — identical
  POST /capture_all   — identical (capture + upload logic)
  All upload helpers  — identical

Run on dashboard laptop:
  python3 viewer.py
  open http://localhost:5050
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template_string, request

# ── Hardware addresses (unchanged from dashboard.py) ─────────────────────────
D435_BASE    = "http://172.20.10.2:8000"
PI_ZERO_CAMS = {
    "left":  "http://172.20.10.7:8001",
    "right": "http://172.20.10.4:8001",
}
API_BASE     = "https://kayenm-greenhouseguardians.hf.space/api/upload"
SAVE_DIR     = Path.home() / "multi_camera_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app  = Flask(__name__)
HTTP = requests.Session()
HTTP.mount("http://",  requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))
HTTP.mount("https://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))

# ── Dashboard HTML ────────────────────────────────────────────────────────────

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Greenhouse Guardians — WebRTC</title>
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
.sep { width: 1px; height: 20px; background: #1f2235; flex-shrink: 0; }
.field { display: flex; align-items: center; gap: 6px; }
.field label {
    font-size: 11px; font-weight: 500; color: #64748b;
    white-space: nowrap; text-transform: uppercase; letter-spacing: 0.06em;
}
.field input {
    background: #1a1d2b; border: 1px solid #2a2e45; color: #e2e8f0;
    border-radius: 6px; padding: 4px 8px; font-size: 13px; width: 68px;
    transition: border-color 0.15s;
}
.field input:focus { outline: none; border-color: #4ade80; }

.btn {
    padding: 6px 16px; border-radius: 6px; border: none;
    font-size: 12px; font-weight: 600; cursor: pointer;
    letter-spacing: 0.03em; transition: opacity 0.15s, transform 0.1s;
    white-space: nowrap;
}
.btn:active { transform: scale(0.97); }
.btn-capture { background: #4ade80; color: #0d0f14; }
.btn-capture:hover { background: #22c55e; }
.btn-health  { background: transparent; color: #64748b; border: 1px solid #2a2e45; }
.btn-health:hover { border-color: #4ade80; color: #4ade80; }

.status-wrap {
    margin-left: auto; display: flex; align-items: center;
    gap: 7px; overflow: hidden;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #4ade80; flex-shrink: 0; transition: background 0.3s;
}
.status-dot.busy  { background: #fbbf24; }
.status-dot.error { background: #f87171; }
#statusText {
    font-size: 12px; color: #475569; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 340px;
}

/* ── Stream grid ── */
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px; padding: 6px; flex: 1; min-height: 0;
}
.card {
    background: #13151e; border-radius: 8px; border: 1px solid #1f2235;
    overflow: hidden; display: flex; flex-direction: column; min-height: 0;
}
.card-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 5px 10px; flex-shrink: 0; border-bottom: 1px solid #1f2235;
}
.card-title {
    font-size: 10px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.pill {
    font-size: 9px; font-weight: 600; padding: 2px 7px; border-radius: 20px;
    background: #1a2e1f; color: #4ade80; letter-spacing: 0.04em;
}

/* Shared styles for both MJPEG <img> and WebRTC <video> */
.card img, .card video {
    flex: 1; width: 100%; min-height: 0;
    object-fit: cover; display: block; background: #0d0f14;
}

/* Connecting overlay */
.card-media-wrap {
    flex: 1; position: relative; min-height: 0; display: flex;
}
.card-media-wrap video, .card-media-wrap img {
    flex: 1; width: 100%; min-height: 0; object-fit: cover;
}
.overlay {
    position: absolute; inset: 0; display: flex;
    align-items: center; justify-content: center;
    background: rgba(13,15,20,0.7); color: #475569;
    font-size: 12px; font-weight: 500; letter-spacing: 0.06em;
    pointer-events: none; transition: opacity 0.3s;
}
.overlay.hidden { opacity: 0; }

/* ── Toast ── */
#toast {
    position: fixed; bottom: 18px; left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: #1a1d2b; border: 1px solid #2a2e45;
    color: #e2e8f0; font-size: 12px; padding: 8px 18px;
    border-radius: 20px; opacity: 0;
    transition: opacity 0.25s, transform 0.25s;
    pointer-events: none; white-space: nowrap; z-index: 100;
}
#toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
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

    <!-- D435 RGB — MJPEG (C++ server unchanged) -->
    <div class="card">
        <div class="card-header">
            <span class="card-title">D435 &mdash; RGB</span>
            <span class="pill" id="pill-d435rgb">LIVE</span>
        </div>
        <img id="d435rgb" src="{{ d435_base }}/stream/rgb.mjpg?t={{ now_ts }}"
             onerror="setPill('d435rgb',false)" />
    </div>

    <!-- D435 Depth — MJPEG (C++ server unchanged) -->
    <div class="card">
        <div class="card-header">
            <span class="card-title">D435 &mdash; Depth</span>
            <span class="pill" id="pill-d435depth">LIVE</span>
        </div>
        <img id="d435depth" src="{{ d435_base }}/stream/depth.mjpg?t={{ now_ts }}"
             onerror="setPill('d435depth',false)" />
    </div>

    <!-- Pi Zero Left — WebRTC -->
    <div class="card">
        <div class="card-header">
            <span class="card-title">Pi Zero &mdash; Left</span>
            <span class="pill" id="pill-left">CONNECTING</span>
        </div>
        <div class="card-media-wrap">
            <video id="pi_left" autoplay muted playsinline></video>
            <div class="overlay" id="overlay-left">CONNECTING…</div>
        </div>
    </div>

    <!-- Pi Zero Right — WebRTC -->
    <div class="card">
        <div class="card-header">
            <span class="card-title">Pi Zero &mdash; Right</span>
            <span class="pill" id="pill-right">CONNECTING</span>
        </div>
        <div class="card-media-wrap">
            <video id="pi_right" autoplay muted playsinline></video>
            <div class="overlay" id="overlay-right">CONNECTING…</div>
        </div>
    </div>

</div>

<div id="toast"></div>

<script>
// ── Utility ──────────────────────────────────────────────────────────────────

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

function setPill(id, state) {
    // state: true = LIVE, false = OFF, 'connecting' = CONNECTING
    const el = document.getElementById("pill-" + id);
    if (!el) return;
    if (state === true) {
        el.textContent = "LIVE";
        el.style.background = "#1a2e1f";
        el.style.color       = "#4ade80";
    } else if (state === "connecting") {
        el.textContent = "CONNECTING";
        el.style.background = "#2a2a1a";
        el.style.color       = "#fbbf24";
    } else {
        el.textContent = "OFF";
        el.style.background = "#2e1a1a";
        el.style.color       = "#f87171";
    }
}

function setOverlay(id, visible, text) {
    const el = document.getElementById("overlay-" + id);
    if (!el) return;
    el.textContent = text || "";
    el.className = visible ? "overlay" : "overlay hidden";
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── WebRTC ───────────────────────────────────────────────────────────────────

/**
 * Perform a single WebRTC offer/answer exchange with the Pi camera.
 * Returns the RTCPeerConnection on success, throws on failure.
 *
 * Flow:
 *  1. Browser creates RTCPeerConnection with no ICE servers (LAN only).
 *  2. addTransceiver('video', recvonly) so the browser knows it wants video.
 *  3. createOffer → setLocalDescription → wait for ICE to complete.
 *  4. POST offer SDP to Pi's /offer endpoint.
 *  5. setRemoteDescription with Pi's answer.
 *  6. ontrack → attach stream to <video> element.
 */
async function connectOnce(videoEl, offerUrl) {
    const pc = new RTCPeerConnection({
        // No STUN/TURN needed on a local network.
        // If cameras are accessed over a different subnet, add:
        //   iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
        iceServers: []
    });

    // Attach incoming video track to the <video> element
    pc.ontrack = (evt) => {
        if (evt.streams && evt.streams[0]) {
            videoEl.srcObject = evt.streams[0];
        }
    };

    // Tell the browser we want to receive (not send) video
    pc.addTransceiver("video", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Wait for ICE gathering.  On LAN with no STUN this completes almost
    // instantly (only host candidates are gathered).  Timeout after 3 s.
    await new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") { resolve(); return; }
        pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") resolve();
        };
        setTimeout(resolve, 3000);
    });

    // Send offer to Pi and receive answer
    const resp = await fetch(offerUrl, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
            sdp:  pc.localDescription.sdp,
            type: pc.localDescription.type
        })
    });

    if (!resp.ok) throw new Error("HTTP " + resp.status + " from " + offerUrl);
    const answer = await resp.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));

    return pc;
}

/**
 * Connect a <video> element to a Pi camera WebRTC stream.
 * Automatically reconnects if the connection drops.
 */
async function connectCamera(videoEl, offerUrl, pillId) {
    while (true) {
        let pc = null;
        try {
            setPill(pillId, "connecting");
            setOverlay(pillId, true, "CONNECTING…");

            pc = await connectOnce(videoEl, offerUrl);

            // Wait for 'connected' state (ICE + DTLS handshake done)
            await new Promise((resolve, reject) => {
                const check = () => {
                    if (pc.connectionState === "connected") resolve();
                    else if (["failed","closed","disconnected"].includes(pc.connectionState))
                        reject(new Error("State: " + pc.connectionState));
                };
                pc.onconnectionstatechange = check;
                check();
                setTimeout(() => reject(new Error("Connection timeout")), 15000);
            });

            setPill(pillId, true);
            setOverlay(pillId, false);

            // Watch for disconnect
            await new Promise((resolve) => {
                pc.onconnectionstatechange = () => {
                    if (["failed","closed","disconnected"].includes(pc.connectionState))
                        resolve();
                };
            });

        } catch (err) {
            console.warn("[WebRTC]", pillId, err.message || err);
        } finally {
            if (pc) { try { pc.close(); } catch(_) {} }
            videoEl.srcObject = null;
            setPill(pillId, false);
            setOverlay(pillId, true, "RECONNECTING…");
        }

        // Back-off before retry
        await sleep(3000);
    }
}

// ── Capture & Health (unchanged from dashboard.py JS) ────────────────────────

async function captureAll() {
    const row  = parseInt(document.getElementById("greenhouse_row").value, 10);
    const dist = parseFloat(document.getElementById("distance_from_start").value);
    if (isNaN(row) || isNaN(dist)) {
        showToast("Enter valid row and distance values.", "error");
        return;
    }
    setStatus("Capturing…", "busy");
    try {
        const res = await fetch("/capture_all", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ greenhouse_row: row, distanceFromRowStart: dist }),
        });
        const data = await res.json();
        if (data.ok) {
            setStatus("Captured · Row " + row + " · " + dist + " m", "");
            showToast("Capture saved — " + data.timestamp, "success");
        } else {
            setStatus("Capture completed with errors", "error");
            showToast("Some captures failed — check logs", "error");
        }
        // Reload D435 MJPEG streams after capture (WebRTC is continuous)
        reloadD435();
    } catch (err) {
        setStatus("Capture failed", "error");
        showToast("Error: " + err, "error");
    }
}

async function checkHealth() {
    setStatus("Checking health…", "busy");
    try {
        const res  = await fetch("/health_all");
        const data = await res.json();
        setPill("d435rgb",   data.d435?.ok ?? false);
        setPill("d435depth", data.d435?.ok ?? false);
        // Pi WebRTC pills are managed by connectCamera(); only override on explicit health check
        if (data.pi_zeros?.left?.ok  === false) setPill("left",  false);
        if (data.pi_zeros?.right?.ok === false) setPill("right", false);
        if (data.ok) {
            setStatus("All devices online", "");
        } else {
            const offline = [];
            if (!data.d435?.ok)              offline.push("D435");
            if (!data.pi_zeros?.left?.ok)    offline.push("Left");
            if (!data.pi_zeros?.right?.ok)   offline.push("Right");
            setStatus("Offline: " + offline.join(", "), "error");
        }
    } catch (err) {
        setStatus("Health check failed", "error");
    }
}

function reloadD435() {
    const ts = Date.now();
    document.getElementById("d435rgb").src   =
        "{{ d435_base }}/stream/rgb.mjpg?t="   + ts;
    document.getElementById("d435depth").src =
        "{{ d435_base }}/stream/depth.mjpg?t=" + ts;
}

// ── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener("load", () => {
    // Start WebRTC connections to Pi cameras (auto-reconnecting)
    connectCamera(
        document.getElementById("pi_left"),
        "{{ pi_left_url }}/offer",
        "left"
    );
    connectCamera(
        document.getElementById("pi_right"),
        "{{ pi_right_url }}/offer",
        "right"
    );

    checkHealth();
    setInterval(checkHealth, 10000);
});
</script>
</body>
</html>
"""


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(
        HTML,
        d435_base    = D435_BASE,
        pi_left_url  = PI_ZERO_CAMS["left"],
        pi_right_url = PI_ZERO_CAMS["right"],
        now_ts       = int(datetime.now().timestamp()),
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


# ── Capture helpers (identical to dashboard.py) ───────────────────────────────

def capture_d435(round_dir: Path, trigger_at_ms: int):
    r = HTTP.post(
        f"{D435_BASE}/capture",
        json={"trigger_at_ms": trigger_at_ms},
        timeout=25,
    )
    if r.status_code != 200:
        raise RuntimeError(f"D435 capture failed {r.status_code}: {r.text}")

    rgb_size    = int(r.headers.get("X-RGB-Size",    "0"))
    depth_scale = float(r.headers.get("X-Depth-Scale","0"))
    fx          = float(r.headers.get("X-FX",        "0"))
    fy          = float(r.headers.get("X-FY",        "0"))
    cx          = float(r.headers.get("X-CX",        "0"))
    cy          = float(r.headers.get("X-CY",        "0"))
    width       = int(r.headers.get("X-Width",       "0"))
    height      = int(r.headers.get("X-Height",      "0"))
    d435_ts     = r.headers.get("X-Timestamp", "")

    payload = r.content
    if rgb_size <= 0 or rgb_size >= len(payload):
        raise RuntimeError("Invalid D435 payload sizes")

    rgb_bytes       = payload[:rgb_size]
    depth_png_bytes = payload[rgb_size:]

    d435_dir = round_dir / "d435"
    d435_dir.mkdir(exist_ok=True)

    rgb_path   = d435_dir / "d435_rgb.jpg"
    depth_path = d435_dir / "d435_depth_raw.png"
    npy_path   = d435_dir / "d435_depth_raw.npy"
    meta_path  = d435_dir / "d435_meta.json"

    rgb_path.write_bytes(rgb_bytes)
    depth_path.write_bytes(depth_png_bytes)

    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    np.save(npy_path, depth_img)

    meta = {
        "camera_name":           "d435",
        "timestamp":             d435_ts,
        "depth_format":          "RS2_FORMAT_Z16 stored as 16-bit PNG",
        "depth_dtype":           str(depth_img.dtype)  if depth_img is not None else None,
        "depth_shape":           list(depth_img.shape) if depth_img is not None else None,
        "depth_scale_m_per_unit": depth_scale,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "width": width, "height": height},
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "rgb_path":   str(rgb_path),
        "depth_path": str(depth_path),
        "npy_path":   str(npy_path),
        "meta_path":  str(meta_path),
        "timestamp":  d435_ts,
    }


def capture_pi_zero(cam_name: str, base_url: str, round_dir: Path, trigger_at_ms: int):
    r = HTTP.post(
        f"{base_url}/capture",
        json={"trigger_at_ms": trigger_at_ms},
        timeout=25,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{cam_name} capture failed {r.status_code}: {r.text}")

    pi_ts             = r.headers.get("X-Timestamp", "")
    returned_cam_name = r.headers.get("X-Camera-Name", cam_name)
    width             = int(r.headers.get("X-Width",  "0"))
    height            = int(r.headers.get("X-Height", "0"))

    pi_dir = round_dir / cam_name
    pi_dir.mkdir(exist_ok=True)

    img_path  = pi_dir / f"{returned_cam_name}_rgb.jpg"
    meta_path = pi_dir / f"{returned_cam_name}_meta.json"

    img_path.write_bytes(r.content)
    meta_path.write_text(json.dumps({
        "camera_name": returned_cam_name,
        "timestamp":   pi_ts,
        "width":       width,
        "height":      height,
        "base_url":    base_url,
    }, indent=2))

    return {
        "rgb_path":    str(img_path),
        "meta_path":   str(meta_path),
        "timestamp":   pi_ts,
        "camera_name": returned_cam_name,
    }


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def upload_rgb_pair(saved: dict, greenhouse_row: int, distance: float, timestamp: str):
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
        data={"timestamp": timestamp,
              "greenhouse_row": greenhouse_row,
              "distanceFromRowStart": distance},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def upload_d435_data(saved: dict, greenhouse_row: int, distance: float, timestamp: str):
    d435_info = saved.get("d435")
    if not d435_info:
        raise RuntimeError("No D435 capture data available to upload")

    meta     = json.loads(Path(d435_info["meta_path"]).read_text())
    rgb_path = Path(d435_info["rgb_path"])
    npy_path = Path(d435_info["npy_path"])

    r = HTTP.post(
        f"{API_BASE}/uploadData",
        files=[
            ("images",      (rgb_path.name, rgb_path.read_bytes(), "image/jpeg")),
            ("depth_image", (npy_path.name, npy_path.read_bytes(), "application/octet-stream")),
        ],
        data={
            "timestamp":            timestamp,
            "greenhouse_row":       greenhouse_row,
            "distanceFromRowStart": distance,
            "fx":                   meta["intrinsics"]["fx"],
            "fy":                   meta["intrinsics"]["fy"],
            "depth_scale":          meta["depth_scale_m_per_unit"],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


@app.route("/capture_all", methods=["POST"])
def capture_all():
    body          = request.get_json(silent=True) or {}
    greenhouse_row = int(body.get("greenhouse_row", 1))
    distance       = float(body.get("distanceFromRowStart", 0.0))

    now            = datetime.now(timezone.utc)
    round_timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    api_ts_rgb     = _fmt_ts(now)
    api_ts_d435    = _fmt_ts(now + timedelta(seconds=1))

    round_dir = SAVE_DIR / round_timestamp
    round_dir.mkdir(parents=True, exist_ok=True)

    trigger_at_ms = (time.time_ns() // 1_000_000) + 1500

    result = {
        "ok":                  True,
        "timestamp":           round_timestamp,
        "trigger_at_ms":       trigger_at_ms,
        "save_dir":            str(round_dir),
        "greenhouse_row":      greenhouse_row,
        "distanceFromRowStart": distance,
        "saved":               {"d435": None, "pi_zeros": {}},
        "uploads":             {},
    }

    # Phase 1: parallel capture
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures[executor.submit(capture_d435, round_dir, trigger_at_ms)] = ("d435", "d435")
        for cam_name, base_url in PI_ZERO_CAMS.items():
            futures[executor.submit(
                capture_pi_zero, cam_name, base_url, round_dir, trigger_at_ms
            )] = ("pi_zero", cam_name)

        for future in as_completed(futures):
            kind, name = futures[future]
            try:
                info = future.result()
                if kind == "d435":
                    result["saved"]["d435"] = info
                else:
                    result["saved"]["pi_zeros"][name] = info
            except Exception as e:
                result["ok"] = False
                result[f"{name}_error"] = str(e)

    # Phase 2: sequential uploads
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
