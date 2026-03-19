import sys
import cv2
import numpy as np
import json
import requests
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

D435_BASE = "http://172.20.10.2:8000"

PI_ZERO_CAMS = {
    "left":   ("http://172.20.10.7:8001", 5001),
    "right":  ("http://172.20.10.4:8001", 5002),
    "bottom": ("http://172.20.10.5:8001", 5003),
}

SAVE_DIR = Path.home() / "multi_camera_captures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HTTP = requests.Session()
HTTP.mount("http://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))


def gst_rtp_pipeline(port: int) -> str:
    return (
        f"udpsrc port={port} "
        f"! application/x-rtp,encoding-name=H264,payload=96 "
        f"! rtph264depay ! h264parse ! avdec_h264 "
        f"! videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(bool)

    def __init__(self, cap_source, is_gstreamer=False, name="camera"):
        super().__init__()
        self.cap_source = cap_source
        self.is_gstreamer = is_gstreamer
        self.name = name
        self._running = True

    def run(self):
        while self._running:
            try:
                backend = cv2.CAP_GSTREAMER if self.is_gstreamer else cv2.CAP_ANY
                cap = cv2.VideoCapture(self.cap_source, backend)

                if not cap.isOpened():
                    self.status_changed.emit(False)
                    time.sleep(3)
                    continue

                self.status_changed.emit(True)

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame is not None:
                        self.frame_ready.emit(frame)

                cap.release()
                self.status_changed.emit(False)
                if self._running:
                    time.sleep(2)

            except Exception as e:
                print(f"[{self.name}] error: {e}")
                self.status_changed.emit(False)
                time.sleep(3)

    def stop(self):
        self._running = False
        self.wait()


class CameraCard(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: white; border-radius: 12px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self.title_label = QLabel(title)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: #111827; background: transparent;")

        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #ef4444; background: transparent; font-size: 14px;")

        title_row.addWidget(self.title_label)
        title_row.addStretch()
        title_row.addWidget(self.dot)
        layout.addLayout(title_row)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background: #e5e7eb; border-radius: 8px; border: 1px solid #d1d5db;"
        )
        layout.addWidget(self.image_label)

    def set_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qt_img.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

    def set_status(self, ok: bool):
        color = "#22c55e" if ok else "#ef4444"
        self.dot.setStyleSheet(f"color: {color}; background: transparent; font-size: 14px;")


class CaptureWorker(QThread):
    done = pyqtSignal(str)

    def run(self):
        try:
            result = run_capture_all()
            ok = result.get("ok", False)
            save_dir = result.get("save_dir", "?")
            self.done.emit(f"{'Saved' if ok else 'Partial save'}: {save_dir}")
        except Exception as e:
            self.done.emit(f"Capture error: {e}")


class HealthWorker(QThread):
    done = pyqtSignal(str)

    def run(self):
        results = {}
        try:
            r = HTTP.get(f"{D435_BASE}/health", timeout=3)
            results["d435"] = r.json()
        except Exception as e:
            results["d435"] = {"ok": False, "error": str(e)}

        for cam_name, (base_url, _) in PI_ZERO_CAMS.items():
            try:
                r = HTTP.get(f"{base_url}/health", timeout=3)
                results[cam_name] = r.json()
            except Exception as e:
                results[cam_name] = {"ok": False, "error": str(e)}

        self.done.emit(json.dumps(results, indent=2))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Greenhouse Guardians")
        self.resize(1280, 760)
        self._threads = []
        self._capture_worker = None
        self._health_worker = None
        self._build_ui()
        self._start_camera_threads()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet("background: #111827;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(12)

        title = QLabel("Greenhouse Guardians")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")

        self.capture_btn = QPushButton("Capture All")
        self.capture_btn.setStyleSheet(
            "background:#2563eb; color:white; border:none; padding:8px 16px;"
            " border-radius:8px; font-size:14px; font-weight:bold;"
        )
        self.capture_btn.clicked.connect(self.on_capture_all)

        self.health_btn = QPushButton("Health")
        self.health_btn.setStyleSheet(
            "background:#374151; color:white; border:none; padding:8px 16px;"
            " border-radius:8px; font-size:14px;"
        )
        self.health_btn.clicked.connect(self.on_health_check)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color:#9ca3af; font-size:13px;")

        hl.addWidget(title)
        hl.addStretch()
        hl.addWidget(self.capture_btn)
        hl.addWidget(self.health_btn)
        hl.addWidget(self.status_label)
        root.addWidget(header)

        # Content area
        content = QWidget()
        content.setStyleSheet("background:#f5f7fb;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.setSpacing(16)

        # Top row: D435 RGB + Depth
        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        self.card_d435_rgb = CameraCard("D435 RGB")
        self.card_d435_depth = CameraCard("D435 Depth")
        top_row.addWidget(self.card_d435_rgb)
        top_row.addWidget(self.card_d435_depth)
        cl.addLayout(top_row)

        # Bottom row: Pi Zeros
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)
        self.card_left = CameraCard("Pi Zero Left")
        self.card_right = CameraCard("Pi Zero Right")
        self.card_bottom = CameraCard("Pi Zero Bottom")
        bottom_row.addWidget(self.card_left)
        bottom_row.addWidget(self.card_right)
        bottom_row.addWidget(self.card_bottom)
        cl.addLayout(bottom_row)

        root.addWidget(content)

    def _start_camera_threads(self):
        cam_configs = [
            (f"{D435_BASE}/stream/rgb.mjpg",   False, "d435_rgb",   self.card_d435_rgb),
            (f"{D435_BASE}/stream/depth.mjpg", False, "d435_depth", self.card_d435_depth),
            (gst_rtp_pipeline(5001),           True,  "left",       self.card_left),
            (gst_rtp_pipeline(5002),           True,  "right",      self.card_right),
            (gst_rtp_pipeline(5003),           True,  "bottom",     self.card_bottom),
        ]
        for source, is_gst, name, card in cam_configs:
            t = CameraThread(source, is_gstreamer=is_gst, name=name)
            t.frame_ready.connect(card.set_frame)
            t.status_changed.connect(card.set_status)
            t.start()
            self._threads.append(t)

    @pyqtSlot()
    def on_capture_all(self):
        self.capture_btn.setEnabled(False)
        self.status_label.setText("Capturing...")
        self._capture_worker = CaptureWorker()
        self._capture_worker.done.connect(self._on_capture_done)
        self._capture_worker.start()

    @pyqtSlot(str)
    def _on_capture_done(self, msg: str):
        self.status_label.setText(msg)
        self.capture_btn.setEnabled(True)

    @pyqtSlot()
    def on_health_check(self):
        self.status_label.setText("Checking health...")
        self._health_worker = HealthWorker()
        self._health_worker.done.connect(self._on_health_done)
        self._health_worker.start()

    @pyqtSlot(str)
    def _on_health_done(self, text: str):
        self.status_label.setText("Health check complete.")
        msg = QMessageBox(self)
        msg.setWindowTitle("Health Check")
        msg.setText(text)
        msg.setFont(QFont("Courier", 10))
        msg.exec_()

    def closeEvent(self, event):
        for t in self._threads:
            t.stop()
        event.accept()


def run_capture_all() -> dict:
    round_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    round_dir = SAVE_DIR / round_timestamp
    round_dir.mkdir(parents=True, exist_ok=True)

    trigger_at_ms = (time.time_ns() // 1_000_000) + 1500

    result = {
        "ok": True,
        "timestamp": round_timestamp,
        "save_dir": str(round_dir),
        "saved": {"d435": None, "pi_zeros": {}},
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=1 + len(PI_ZERO_CAMS)) as executor:
        futures[executor.submit(_capture_d435, round_dir, trigger_at_ms)] = ("d435", "d435")
        for cam_name, (base_url, _) in PI_ZERO_CAMS.items():
            futures[executor.submit(
                _capture_pi_zero, cam_name, base_url, round_dir, trigger_at_ms
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

    (round_dir / "capture_summary.json").write_text(json.dumps(result, indent=2))
    return result


def _capture_d435(round_dir: Path, trigger_at_ms: int) -> dict:
    r = HTTP.post(f"{D435_BASE}/capture", json={"trigger_at_ms": trigger_at_ms}, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"D435 capture failed: {r.status_code}")

    rgb_size = int(r.headers.get("X-RGB-Size", "0"))
    payload = r.content
    if rgb_size <= 0 or rgb_size >= len(payload):
        raise RuntimeError("Invalid D435 payload")

    d435_dir = round_dir / "d435"
    d435_dir.mkdir(exist_ok=True)

    rgb_path = d435_dir / "rgb.jpg"
    depth_path = d435_dir / "depth_raw.png"
    npy_path = d435_dir / "depth_raw.npy"
    meta_path = d435_dir / "meta.json"

    rgb_path.write_bytes(payload[:rgb_size])
    depth_path.write_bytes(payload[rgb_size:])

    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    np.save(str(npy_path), depth_img)

    meta_path.write_text(json.dumps({
        "camera_name": "d435",
        "timestamp": r.headers.get("X-Timestamp", ""),
        "depth_scale_m_per_unit": float(r.headers.get("X-Depth-Scale", "0")),
        "intrinsics": {
            "fx": float(r.headers.get("X-FX", "0")),
            "fy": float(r.headers.get("X-FY", "0")),
            "cx": float(r.headers.get("X-CX", "0")),
            "cy": float(r.headers.get("X-CY", "0")),
            "width": int(r.headers.get("X-Width", "0")),
            "height": int(r.headers.get("X-Height", "0")),
        },
    }, indent=2))

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "timestamp": r.headers.get("X-Timestamp", ""),
    }


def _capture_pi_zero(cam_name: str, base_url: str, round_dir: Path, trigger_at_ms: int) -> dict:
    r = HTTP.post(f"{base_url}/capture", json={"trigger_at_ms": trigger_at_ms}, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"{cam_name} capture failed: {r.status_code}")

    pi_dir = round_dir / cam_name
    pi_dir.mkdir(exist_ok=True)

    img_path = pi_dir / "rgb.jpg"
    meta_path = pi_dir / "meta.json"

    img_path.write_bytes(r.content)
    meta_path.write_text(json.dumps({
        "camera_name": r.headers.get("X-Camera-Name", cam_name),
        "timestamp": r.headers.get("X-Timestamp", ""),
        "width": int(r.headers.get("X-Width", "0")),
        "height": int(r.headers.get("X-Height", "0")),
    }, indent=2))

    return {"rgb_path": str(img_path), "timestamp": r.headers.get("X-Timestamp", "")}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
