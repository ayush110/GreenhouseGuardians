from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import threading
import time
from datetime import datetime
import io

app = Flask(__name__)

CAM_NAME = "pi_zero_left"

# Preview: small + YUV420 = fast MJPEG encoding, low CPU
PREVIEW_WIDTH  = 480
PREVIEW_HEIGHT = 270

# Full-resolution still
STILL_WIDTH  = 4608
STILL_HEIGHT = 2592

JPEG_QUALITY_CAPTURE = 92


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()
        self.timestamp = ""
        self.frame_counter = 0

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.timestamp = timestamp_now()
            self.frame_counter += 1
            self.condition.notify_all()


app_output = StreamingOutput()
picam2 = Picamera2()
camera_lock = threading.Lock()


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def now_ms():
    return int(time.time() * 1000)


def wait_until_ms(trigger_ms):
    while True:
        remaining = trigger_ms - now_ms()
        if remaining <= 0:
            return
        elif remaining > 20:
            time.sleep((remaining - 10) / 1000.0)
        else:
            time.sleep(0.001)


def configure_preview():
    config = picam2.create_video_configuration(
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "YUV420"},
        buffer_count=4,
        controls={"FrameDurationLimits": (33333, 33333)}  # cap at ~30 fps
    )
    picam2.configure(config)


def start_preview_stream():
    encoder = MJPEGEncoder()
    picam2.start_recording(encoder, FileOutput(app_output))


def stop_preview_stream():
    try:
        picam2.stop_recording()
    except Exception:
        try:
            picam2.stop()
        except Exception:
            pass


def init_camera():
    configure_preview()
    start_preview_stream()
    time.sleep(1)


def mjpeg_generator():
    while True:
        try:
            with app_output.condition:
                app_output.condition.wait(timeout=2.0)
                frame = app_output.frame
            if frame is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
        except GeneratorExit:
            break
        except Exception:
            time.sleep(0.02)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "camera_name": CAM_NAME,
        "frames_ready": app_output.frame is not None,
        "frame_counter": app_output.frame_counter,
        "preview_size": f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}",
        "still_size": f"{STILL_WIDTH}x{STILL_HEIGHT}",
    })


@app.route("/stream/rgb.mjpg")
def stream_rgb():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Accel-Buffering": "no",
        },
        direct_passthrough=True,
    )


@app.route("/capture", methods=["POST"])
def capture():
    ts = timestamp_now()

    # Optional trigger synchronization
    trigger_ms = request.args.get("trigger_at_ms", type=int)
    if trigger_ms:
        wait_until_ms(trigger_ms)

    try:
        with camera_lock:
            stop_preview_stream()

            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT), "format": "RGB888"},
                buffer_count=2,
            )
            picam2.configure(still_config)
            picam2.start()

            # Wait for 2 real frames instead of a fixed sleep.
            # Frame 1: AEC/AWB settling; frame 2: properly exposed.
            req = picam2.capture_request()
            req.release()
            req = picam2.capture_request()
            frame = req.make_array("main")
            req.release()

            picam2.stop()
            configure_preview()
            start_preview_stream()
            # No sleep needed — stream recovers as frames arrive

        from PIL import Image

        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY_CAPTURE)
        jpg = buf.getvalue()

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = ts
        resp.headers["X-Frame-Counter"] = str(app_output.frame_counter)
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(STILL_WIDTH)
        resp.headers["X-Height"] = str(STILL_HEIGHT)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    except Exception as e:
        # Best-effort recovery
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            configure_preview()
            start_preview_stream()
        except Exception:
            pass
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_camera()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)
