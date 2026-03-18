from flask import Flask, Response, jsonify
from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import threading

app = Flask(__name__)

CAM_NAME = "pi_zero_cam"

# Preview resolution for dashboard
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

# Full-resolution still capture
STILL_WIDTH = 4608
STILL_HEIGHT = 2592

JPEG_QUALITY_PREVIEW = 70
JPEG_QUALITY_CAPTURE = 95

# Lower = more responsive, higher CPU/network
PREVIEW_INTERVAL_SEC = 0.03   # ~33 fps max target

picam2 = Picamera2()
camera_lock = threading.Lock()

latest_preview_jpg = None
latest_preview_ts = ""
preview_counter = 0
running = True


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def encode_jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def configure_preview():
    preview_config = picam2.create_video_configuration(
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
        buffer_count=2
    )
    picam2.configure(preview_config)


def init_camera():
    configure_preview()
    picam2.start()
    time.sleep(2)


def preview_loop():
    global latest_preview_jpg, latest_preview_ts, preview_counter, running

    while running:
        try:
            with camera_lock:
                frame = picam2.capture_array("main")

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            jpg = encode_jpeg(frame_bgr, JPEG_QUALITY_PREVIEW)
            if jpg is not None:
                latest_preview_jpg = jpg
                latest_preview_ts = timestamp_now()
                preview_counter += 1

            time.sleep(PREVIEW_INTERVAL_SEC)

        except Exception as e:
            print(f"Preview loop error: {e}")
            time.sleep(0.1)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "camera_name": CAM_NAME,
        "frames_ready": latest_preview_jpg is not None,
        "frame_counter": preview_counter,
        "preview_width": PREVIEW_WIDTH,
        "preview_height": PREVIEW_HEIGHT,
        "still_width": STILL_WIDTH,
        "still_height": STILL_HEIGHT,
        "latest_timestamp": latest_preview_ts
    })


@app.route("/frame.jpg")
def frame_jpg():
    if latest_preview_jpg is None:
        return jsonify({"ok": False, "error": "preview not ready"}), 503

    resp = Response(latest_preview_jpg, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Timestamp"] = latest_preview_ts
    resp.headers["X-Frame-Counter"] = str(preview_counter)
    resp.headers["X-Camera-Name"] = CAM_NAME
    return resp


@app.route("/capture", methods=["POST"])
def capture():
    ts = timestamp_now()

    try:
        with camera_lock:
            picam2.stop()

            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT), "format": "RGB888"},
                buffer_count=1
            )
            picam2.configure(still_config)
            picam2.start()
            time.sleep(0.6)

            frame = picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            picam2.stop()
            configure_preview()
            picam2.start()
            time.sleep(0.3)

        jpg = encode_jpeg(frame_bgr, JPEG_QUALITY_CAPTURE)
        if jpg is None:
            return jsonify({"ok": False, "error": "failed to encode full-res still"}), 500

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = ts
        resp.headers["X-Frame-Counter"] = str(preview_counter)
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(frame_bgr.shape[1])
        resp.headers["X-Height"] = str(frame_bgr.shape[0])
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_camera()
    t = threading.Thread(target=preview_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)