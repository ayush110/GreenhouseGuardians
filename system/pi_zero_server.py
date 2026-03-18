from flask import Flask, Response, jsonify
from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import threading

app = Flask(__name__)

CAM_NAME = "pi_zero_cam"
WIDTH = 640
HEIGHT = 480
JPEG_QUALITY_CAPTURE = 90
JPEG_QUALITY_STREAM = 55
FRAME_INTERVAL_SEC = 0.05  # ~20 FPS max

picam2 = Picamera2()
camera_lock = threading.Lock()

latest_frame = None
latest_timestamp = ""
frame_counter = 0
running = True


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def init_camera():
    global picam2
    config = picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
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


def encode_jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


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
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
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
        "width": WIDTH,
        "height": HEIGHT,
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

    if latest_frame is None:
        return jsonify({"ok": False, "error": "frames not ready"}), 503

    frame = latest_frame.copy()
    ts = latest_timestamp
    fc = frame_counter

    jpg = encode_jpeg(frame, JPEG_QUALITY_CAPTURE)
    if jpg is None:
        return jsonify({"ok": False, "error": "failed to encode image"}), 500

    resp = Response(jpg, mimetype="image/jpeg")
    resp.headers["X-Timestamp"] = ts
    resp.headers["X-Frame-Counter"] = str(fc)
    resp.headers["X-Camera-Name"] = CAM_NAME
    resp.headers["X-Width"] = str(frame.shape[1])
    resp.headers["X-Height"] = str(frame.shape[0])
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


if __name__ == "__main__":
    init_camera()
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)