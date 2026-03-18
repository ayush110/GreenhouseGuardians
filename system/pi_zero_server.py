from flask import Flask, Response, jsonify
from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import threading

app = Flask(__name__)

CAM_NAME = "pi_zero_cam"

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

STILL_WIDTH = 4608
STILL_HEIGHT = 2592

JPEG_QUALITY = 95

picam2 = Picamera2()
camera_lock = threading.Lock()

latest_frame = None
latest_timestamp = ""
frame_counter = 0
running = True


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def init_preview_camera():
    preview_config = picam2.create_video_configuration(
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)


def capture_loop():
    global latest_frame, latest_timestamp, frame_counter, running

    while running:
        try:
            with camera_lock:
                frame = picam2.capture_array()

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            latest_frame = frame_bgr
            latest_timestamp = timestamp_now()
            frame_counter += 1

        except Exception as e:
            print(f"Capture loop error: {e}")
            time.sleep(0.2)


def encode_jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def mjpeg_generator():
    last_seen = -1
    while True:
        global latest_frame, frame_counter

        if latest_frame is None or frame_counter == last_seen:
            time.sleep(0.03)
            continue

        frame = latest_frame.copy()
        last_seen = frame_counter

        jpg = encode_jpeg(frame, 75)
        if jpg is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg +
            b"\r\n"
        )


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "camera_name": CAM_NAME,
        "frames_ready": latest_frame is not None,
        "frame_counter": frame_counter,
        "preview_width": PREVIEW_WIDTH,
        "preview_height": PREVIEW_HEIGHT,
        "still_width": STILL_WIDTH,
        "still_height": STILL_HEIGHT,
    })


@app.route("/stream/rgb.mjpg")
def stream_rgb():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture", methods=["POST"])
def capture():
    global latest_timestamp

    ts = timestamp_now()

    try:
        with camera_lock:
            picam2.stop()

            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT), "format": "RGB888"}
            )
            picam2.configure(still_config)
            picam2.start()
            time.sleep(0.5)

            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            picam2.stop()

            preview_config = picam2.create_video_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"}
            )
            picam2.configure(preview_config)
            picam2.start()
            time.sleep(0.3)

        jpg = encode_jpeg(frame_bgr, JPEG_QUALITY)
        if jpg is None:
            return jsonify({"ok": False, "error": "failed to encode image"}), 500

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = ts
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(frame_bgr.shape[1])
        resp.headers["X-Height"] = str(frame_bgr.shape[0])
        return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_preview_camera()
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)