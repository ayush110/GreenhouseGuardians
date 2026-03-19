#!/usr/bin/env python3
"""
app2.py  —  Pi Zero camera server with WebRTC H.264 live-preview
─────────────────────────────────────────────────────────────────
Drop-in replacement for app.py.

WHAT CHANGED: live-preview transport only.
  - MJPEG  /stream/rgb.mjpg  →  WebRTC  POST /offer
  - Browser connects directly; H.264 is negotiated automatically.
  - All capture logic (still mode, trigger sync, headers) is unchanged.

WHAT IS THE SAME:
  GET  /health      — identical response to app.py
  POST /capture     — identical logic, headers, timing

Install on Pi:
  pip install aiohttp aiortc av picamera2

  On Pi Zero W (ARMv6) you may need to build av/aiortc from source:
    sudo apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
    pip install av aiortc aiohttp

Run:
  CAM_NAME=pi_left  python3 app2.py
  CAM_NAME=pi_right PORT=8001 python3 app2.py
"""

import asyncio
import io
import json
import logging
import os
import threading
import time

import av
import numpy as np
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

try:
    from picamera2 import Picamera2
    from PIL import Image
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

# ── Config ────────────────────────────────────────────────────────────────────
CAM_NAME       = os.environ.get("CAM_NAME", "pi_zero_unknown")
PORT           = int(os.environ.get("PORT", "8001"))

# Preview resolution — lower if Pi Zero W struggles (e.g. 640×480)
STREAM_WIDTH   = int(os.environ.get("STREAM_WIDTH",  "1280"))
STREAM_HEIGHT  = int(os.environ.get("STREAM_HEIGHT", "720"))
STREAM_FPS     = int(os.environ.get("STREAM_FPS",    "15"))

# High-res still capture (unchanged from app.py)
STILL_WIDTH    = 4608
STILL_HEIGHT   = 2592
JPEG_QUALITY   = 95

# ── Globals ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_picam2           = None
_camera_mode      = "stream"    # "stream" | "capturing"
_camera_lock      = threading.Lock()

# All active WebRTC tracks (one per connected browser tab)
_active_tracks    : list["PiCameraTrack"] = []
_tracks_lock      = threading.Lock()

# Set to the running asyncio event loop once the server starts
_event_loop : asyncio.AbstractEventLoop | None = None

# ── WebRTC video track ────────────────────────────────────────────────────────

class PiCameraTrack(MediaStreamTrack):
    """
    One video track per peer connection.
    The camera capture thread calls push_frame() from a non-async context;
    recv() is called by aiortc from within the event loop.
    """
    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        # maxsize=2 keeps latency low; older frames are dropped
        self._queue: asyncio.Queue[av.VideoFrame] = asyncio.Queue(maxsize=2)

    # ── called from background thread ──────────────────────────────────────
    def _enqueue(self, frame: av.VideoFrame) -> None:
        """Thread-safe enqueue; drops frame if queue is full."""
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # drop old frame to keep latency low

    def push_frame(self, bgr: np.ndarray) -> None:
        if _event_loop is None:
            return
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        _event_loop.call_soon_threadsafe(self._enqueue, frame)

    # ── called from asyncio event loop (aiortc) ────────────────────────────
    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            # Return a blank frame so aiortc doesn't stall
            blank = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(blank, format="bgr24")
        frame.pts       = pts
        frame.time_base = time_base
        return frame


# ── Camera capture thread ─────────────────────────────────────────────────────

def _camera_loop() -> None:
    """
    Background daemon thread.
    Initialises picamera2 in video mode and continuously pushes raw BGR frames
    to every active WebRTC track.  Pauses (without holding the lock) while a
    still capture is in progress.
    """
    global _picam2

    if not HAS_PICAMERA:
        log.warning("picamera2 not available — sending test pattern")
        _synthetic_loop()
        return

    _picam2 = Picamera2()
    cfg = _picam2.create_video_configuration(
        main={"format": "RGB888", "size": (STREAM_WIDTH, STREAM_HEIGHT)},
        controls={"FrameRate": STREAM_FPS},
    )
    _picam2.configure(cfg)
    _picam2.start()
    log.info("Camera streaming %dx%d @ %d fps", STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS)

    frame_period = 1.0 / STREAM_FPS

    while True:
        # Pause while a still capture is reconfiguring the camera
        if _camera_mode != "stream":
            time.sleep(0.05)
            continue

        with _camera_lock:
            # Re-check inside the lock so we don't race with _do_still_capture
            if _camera_mode != "stream":
                continue
            try:
                arr_rgb = _picam2.capture_array("main")  # RGB888, non-blocking
            except Exception as exc:
                log.warning("capture_array error: %s", exc)
                time.sleep(0.1)
                continue

        arr_bgr = arr_rgb[:, :, ::-1].copy()   # RGB → BGR (aiortc/av expects BGR)

        with _tracks_lock:
            for track in _active_tracks:
                track.push_frame(arr_bgr)

        time.sleep(frame_period)


def _synthetic_loop() -> None:
    """Animated colour-bar pattern for development without a camera."""
    t = 0.0
    while True:
        arr = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
        arr[:, :, 0] = int(127 + 127 * np.sin(t))
        arr[:, :, 2] = int(127 + 127 * np.cos(t))
        arr[10:30, 10:160] = 180   # crude "label" bar
        with _tracks_lock:
            for track in _active_tracks:
                track.push_frame(arr)
        t += 0.15
        time.sleep(1.0 / STREAM_FPS)


# ── Still capture helpers (identical logic to app.py) ────────────────────────

def timestamp_now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def wait_until_trigger(trigger_at_ms) -> None:
    """Busy-wait for precise multi-camera synchronisation (unchanged from app.py)."""
    if trigger_at_ms is None:
        return
    while True:
        now_ms      = time.time_ns() // 1_000_000
        remaining   = trigger_at_ms - now_ms
        if remaining <= 0:
            return
        if remaining > 20:
            time.sleep((remaining - 10) / 1000.0)
        else:
            time.sleep(0.001)


def _do_still_capture(trigger_at_ms):
    """
    Blocking function (runs in thread-pool via run_in_executor).
    Stops the streaming loop, switches to still mode, captures, restores stream.
    Returns (jpeg_bytes, timestamp_str).
    """
    global _camera_mode

    # Signal the streaming loop to pause, then wait for it to release the lock
    _camera_mode = "capturing"
    time.sleep(0.15)   # ≥ one frame period at 15 fps gives the loop time to exit

    with _camera_lock:
        # Stop current config
        _picam2.stop()

        # High-res still configuration
        still_cfg = _picam2.create_still_configuration(
            main={"size": (STILL_WIDTH, STILL_HEIGHT)}
        )
        _picam2.configure(still_cfg)
        _picam2.start()

        time.sleep(0.25)          # let AE / AWB settle (unchanged from app.py)

        wait_until_trigger(trigger_at_ms)   # synchronised capture

        frame = _picam2.capture_array("main")   # RGB888
        captured_ts = timestamp_now()

        # Restore streaming configuration
        _picam2.stop()
        stream_cfg = _picam2.create_video_configuration(
            main={"format": "RGB888", "size": (STREAM_WIDTH, STREAM_HEIGHT)},
            controls={"FrameRate": STREAM_FPS},
        )
        _picam2.configure(stream_cfg)
        _picam2.start()
        _camera_mode = "stream"

    img = Image.fromarray(frame, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), captured_ts, frame.shape[1], frame.shape[0]


# ── CORS helper ───────────────────────────────────────────────────────────────

def _cors() -> dict:
    return {
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


# ── Route handlers ────────────────────────────────────────────────────────────

async def handle_options(request: web.Request) -> web.Response:
    """CORS preflight for all routes."""
    return web.Response(headers=_cors())


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok":            True,
            "camera_name":   CAM_NAME,
            "stream_width":  STREAM_WIDTH,
            "stream_height": STREAM_HEIGHT,
            "still_width":   STILL_WIDTH,
            "still_height":  STILL_HEIGHT,
            "webrtc":        True,
        },
        headers=_cors(),
    )


async def handle_offer(request: web.Request) -> web.Response:
    """
    WebRTC signalling endpoint.
    Browser sends SDP offer → we add a video track → return SDP answer.
    ICE is gathered server-side before returning (trickle-free for simplicity).
    """
    body  = await request.json()
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc    = RTCPeerConnection()
    track = PiCameraTrack()

    with _tracks_lock:
        _active_tracks.append(track)

    @pc.on("connectionstatechange")
    async def _on_state_change():
        log.info("[%s] WebRTC state → %s", CAM_NAME, pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            with _tracks_lock:
                if track in _active_tracks:
                    _active_tracks.remove(track)

    pc.addTrack(track)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Wait for ICE gathering to complete so candidates are embedded in the SDP.
    # On a local network with no STUN this is nearly instant.
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        headers=_cors(),
    )


async def handle_capture(request: web.Request) -> web.Response:
    """
    Synchronized still capture — logic identical to app.py /capture.
    Accepts optional JSON body: { "trigger_at_ms": <int> }
    Returns JPEG with X-Timestamp, X-Camera-Name, X-Width, X-Height headers.
    """
    if not HAS_PICAMERA:
        return web.Response(status=503, text="No camera available")

    try:
        body = await request.json()
    except Exception:
        body = {}

    trigger_at_ms = body.get("trigger_at_ms")

    loop = asyncio.get_event_loop()
    try:
        jpg, ts, w, h = await loop.run_in_executor(
            None, _do_still_capture, trigger_at_ms
        )
    except Exception as exc:
        log.exception("Still capture failed")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"ok": False, "error": str(exc), "camera_name": CAM_NAME}),
        )

    return web.Response(
        body=jpg,
        content_type="image/jpeg",
        headers={
            **_cors(),
            "X-Timestamp":     ts,
            "X-Camera-Name":   CAM_NAME,
            "X-Width":         str(w),
            "X-Height":        str(h),
            "Cache-Control":   "no-store, no-cache, must-revalidate, max-age=0",
        },
    )


# ── App factory ───────────────────────────────────────────────────────────────

def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get( "/health",         handle_health)
    app.router.add_post("/offer",          handle_offer)
    app.router.add_post("/capture",        handle_capture)
    app.router.add_options("/{tail:.*}",   handle_options)
    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Grab the event loop before starting the server so push_frame() can use it
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _event_loop = loop

    # Camera capture runs in a daemon thread — aiohttp owns the event loop
    cam_thread = threading.Thread(target=_camera_loop, daemon=True, name="camera-loop")
    cam_thread.start()

    log.info("Starting %s (WebRTC) on 0.0.0.0:%d", CAM_NAME, PORT)
    web.run_app(build_app(), host="0.0.0.0", port=PORT, loop=loop)
