from flask import Flask, Response, jsonify
from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import threading

app = Flask(__name__)

CAM_NAME = "pi_zero_cam"

# Live preview resolution for dashboard
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

# Full-resolution still capture
STILL_WIDTH = 4608
STILL_HEIGHT = 2592

# JPEG settings
JPEG_QUALITY_CAPTURE = 95
JPEG_QUALITY_STREAM = 60

# Limit preview FPS a bit to reduce lag on Pi Zero 2 W
FRAME_INTERVAL_SEC = 0.07

picam2 = Picamera2()
camera_lock = threading.Lock()

latest_frame = None
latest_timestamp = ""
frame_counter = 0
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
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(preview_config)


def init_camera():
    configure_preview()
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

            time.sleep(FRAME_INTERVAL_SEC)

        except Exception as e:
            print(f"Capture loop error: {e}")
            time.sleep(0.2)


def mjpeg_generator():
    global latest_frame, frame_counter, latest_timestamp
    last_seen = -1

    while True:
        try:
            if latest_frame is None or frame_counter == last_seen:
                time.sleep(0.01)
                continue

            frame = latest_frame.copy()
            ts = latest_timestamp
            fc = frame_counter
            last_seen = fc

            cv2.putText(
                frame,
                f"{CAM_NAME}  {ts}  fc={fc}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            jpg = encode_jpeg(frame, JPEG_QUALITY_STREAM)
            if jpg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n\r\n"
                + jpg +
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
        "frames_ready": latest_frame is not None,
        "frame_counter": frame_counter,
        "preview_width": PREVIEW_WIDTH,
        "preview_height": PREVIEW_HEIGHT,
        "still_width": STILL_WIDTH,
        "still_height": STILL_HEIGHT,
        "latest_timestamp": latest_timestamp
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
    global latest_frame, latest_timestamp, frame_counter

    ts = timestamp_now()

    try:
        with camera_lock:
            # Stop preview mode
            picam2.stop()

            # Switch to full-res still mode
            still_config = picam2.create_still_configuration(
                main={"size": (STILL_WIDTH, STILL_HEIGHT), "format": "RGB888"}
            )
            picam2.configure(still_config)
            picam2.start()
            time.sleep(0.6)

            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Switch back to preview mode
            picam2.stop()
            configure_preview()
            picam2.start()
            time.sleep(0.3)

        jpg = encode_jpeg(frame_bgr, JPEG_QUALITY_CAPTURE)
        if jpg is None:
            return jsonify({"ok": False, "error": "failed to encode image"}), 500

        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Timestamp"] = ts
        resp.headers["X-Frame-Counter"] = str(frame_counter)
        resp.headers["X-Camera-Name"] = CAM_NAME
        resp.headers["X-Width"] = str(frame_bgr.shape[1])
        resp.headers["X-Height"] = str(frame_bgr.shape[0])
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_camera()
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)