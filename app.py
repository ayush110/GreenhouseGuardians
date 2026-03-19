from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import threading
import time
from datetime import datetime
import io
import os
import subprocess

app = Flask(__name__)

CAM_NAME = os.environ.get("CAM_NAME", "pi_zero_unknown")
DASHBOARD_IP = os.environ.get("DASHBOARD_IP", "172.20.10.6")
STREAM_PORT = int(os.environ.get("STREAM_PORT", "5001"))

STREAM_WIDTH = 640
STREAM_HEIGHT = 360

PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180

STILL_WIDTH = 4608
STILL_HEIGHT = 2592

JPEG_QUALITY_CAPTURE = 95

picam2 = Picamera2()
camera_lock = threading.Lock()
gst_proc = None
stream_running = False

_preview_jpg = b""
_preview_lock = threading.Lock()


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def wait_until_trigger(trigger_at_ms):
    if trigger_at_ms is None:
        return
    while True:
        now_ms = time.time_ns() // 1_000_000
        remaining_ms = trigger_at_ms - now_ms
        if remaining_ms <= 0:
            return
        if remaining_ms > 20:
            time.sleep((remaining_ms - 10) / 1000.0)
        else:
            time.sleep(0.001)


def configure_stream():
    config = picam2.create_video_configuration(
        main={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "YUV420"},
        lores={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(config)


def start_gst_stream():
    global gst_proc, stream_running
    gst_proc = subprocess.Popen(
        [
            "gst-launch-1.0", "fdsrc", "!",
            "h264parse", "!",
            "rtph264pay", "config-interval=1", "pt=96", "!",
            "udpsink", f"host={DASHBOARD_IP}", f"port={STREAM_PORT}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    encoder = H264Encoder(bitrate=2_000_000)
    picam2.start_recording(encoder, FileOutput(gst_proc.stdin))
    stream_running = True


def stop_gst_stream():
    global gst_proc, stream_running
    stream_running = False
    try:
        picam2.stop_recording()
    except Exception:
        try:
            picam2.stop()
        except Exception:
            pass
    if gst_proc is not None:
        try:
            gst_proc.stdin.close()
        except Exception:
            pass
        try:
            gst_proc.terminate()
            gst_proc.wait(timeout=3)
        except Exception:
            pass
        gst_proc = None


def _preview_worker():
    global _preview_jpg
    from PIL import Image
    while True:
        if not stream_running:
            time.sleep(0.1)
            continue
        try:
            with picam2.capture_request() as req:
                frame = req.make_array("lores")  # RGB888
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            with _preview_lock:
                _preview_jpg = buf.getvalue()
        except Exception:
            time.sleep(0.1)
        time.sleep(0.05)  # ~15 fps


def init_camera():
    configure_stream()
    start_gst_stream()
    threading.Thread(target=_preview_worker, daemon=True).start()
    time.sleep(2)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "camera_name": CAM_NAME,
        "stream_running": stream_running,
        "stream_host": DASHBOARD_IP,
        "stream_port": STREAM_PORT,
        "stream_width": STREAM_WIDTH,
        "stream_height": STREAM_HEIGHT,
        "still_width": STILL_WIDTH,
        "still_height": STILL_HEIGHT,
    })


@app.route("/stream/rgb.mjpg")
def stream_rgb():
    def generate():
        last = b""
        while True:
            with _preview_lock:
                jpg = _preview_jpg
            if jpg and jpg is not last:
                last = jpg
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg + b"\r\n"
                )
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def capture():
    ts_requested = timestamp_now()

    try:
        body = request.get_json(silent=True) or {}
        trigger_at_ms = body.get("trigger_at_ms")

        with camera_lock:
            stop_gst_stream()

            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT), "format": "RGB888"}
            )
            picam2.configure(still_config)
            picam2.start()

            # let AE settle a bit before the synced trigger
            time.sleep(0.25)

            wait_until_trigger(trigger_at_ms)

            frame = picam2.capture_array("main")
            captured_ts = timestamp_now()

            picam2.stop()
            configure_stream()
            start_gst_stream()
            time.sleep(0.2)

        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.asarray(frame))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY_CAPTURE)
        jpg = buf.getvalue()

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = captured_ts
        resp.headers["X-Requested-Timestamp"] = ts_requested
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(frame.shape[1])
        resp.headers["X-Height"] = str(frame.shape[0])
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "camera_name": CAM_NAME}), 500


if __name__ == "__main__":
    init_camera()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)
