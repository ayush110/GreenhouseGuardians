from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import threading
import time
import io
import os
from datetime import datetime

app = Flask(__name__)

CAM_NAME = os.environ.get("CAM_NAME", "pi_zero_unknown")

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

STILL_WIDTH = 4608
STILL_HEIGHT = 2592

JPEG_QUALITY_CAPTURE = 95

picam2 = Picamera2()
camera_lock = threading.Lock()
stream_encoder = None
output = None
stream_running = False


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


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


def start_stream():
    global stream_encoder, output, stream_running
    output = StreamingOutput()
    stream_config = picam2.create_video_configuration(
        main={"size": (STREAM_WIDTH, STREAM_HEIGHT)}
    )
    picam2.configure(stream_config)
    stream_encoder = MJPEGEncoder()
    picam2.start_recording(stream_encoder, FileOutput(output))
    stream_running = True


def stop_stream():
    global stream_running
    stream_running = False
    try:
        picam2.stop_recording()
    except Exception:
        try:
            picam2.stop()
        except Exception:
            pass


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "camera_name": CAM_NAME,
        "stream_running": stream_running,
        "stream_width": STREAM_WIDTH,
        "stream_height": STREAM_HEIGHT,
        "still_width": STILL_WIDTH,
        "still_height": STILL_HEIGHT,
    })


@app.route("/stream/rgb.mjpg")
def stream_rgb():
    def generate():
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def capture():
    try:
        body = request.get_json(silent=True) or {}
        trigger_at_ms = body.get("trigger_at_ms")

        with camera_lock:
            stop_stream()

            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT)}
            )
            picam2.configure(still_config)
            picam2.start()

            time.sleep(0.25)  # let AE settle

            wait_until_trigger(trigger_at_ms)

            frame = picam2.capture_array("main")
            captured_ts = timestamp_now()

            picam2.stop()
            start_stream()

        from PIL import Image

        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY_CAPTURE)
        jpg = buf.getvalue()

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = captured_ts
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(frame.shape[1])
        resp.headers["X-Height"] = str(frame.shape[0])
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "camera_name": CAM_NAME}), 500


if __name__ == "__main__":
    start_stream()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)
