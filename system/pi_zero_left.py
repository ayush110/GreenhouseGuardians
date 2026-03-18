from flask import Flask, Response, jsonify
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import threading
import time
from datetime import datetime
import io

app = Flask(__name__)

CAM_NAME = "pi_zero_left"

# Preview stream resolution
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

# Full-resolution still capture
STILL_WIDTH = 4608
STILL_HEIGHT = 2592

JPEG_QUALITY_CAPTURE = 95


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


def configure_preview():
    config = picam2.create_video_configuration(
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
        buffer_count=3
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
    time.sleep(2)


def mjpeg_generator():
    while True:
        try:
            with app_output.condition:
                app_output.condition.wait()
                frame = app_output.frame
            if frame is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n\r\n"
                + frame +
                b"\r\n"
            )
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Stream generator error: {e}")
            time.sleep(0.05)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "camera_name": CAM_NAME,
        "frames_ready": app_output.frame is not None,
        "frame_counter": app_output.frame_counter,
        "preview_width": PREVIEW_WIDTH,
        "preview_height": PREVIEW_HEIGHT,
        "still_width": STILL_WIDTH,
        "still_height": STILL_HEIGHT,
        "latest_timestamp": app_output.timestamp
    })


@app.route("/stream/rgb.mjpg")
def stream_rgb():
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
        "Connection": "close",
    }
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
        direct_passthrough=True
    )


@app.route("/capture", methods=["POST"])
def capture():
    ts = timestamp_now()

    try:
        with camera_lock:
            stop_preview_stream()

            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT), "format": "RGB888"}
            )
            picam2.configure(still_config)
            picam2.start()
            time.sleep(0.6)

            frame = picam2.capture_array("main")

            picam2.stop()
            configure_preview()
            start_preview_stream()
            time.sleep(0.3)

        # Encode still as JPEG using picamera2/libcamera path via simplebuffer? fallback to cv2 avoided.
        # Since Flask needs bytes, use PIL here for lighter one-off encoding than per-frame streaming.
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.asarray(frame))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY_CAPTURE)
        jpg = buf.getvalue()

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = ts
        resp.headers["X-Frame-Counter"] = str(app_output.frame_counter)
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(frame.shape[1])
        resp.headers["X-Height"] = str(frame.shape[0])
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_camera()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)