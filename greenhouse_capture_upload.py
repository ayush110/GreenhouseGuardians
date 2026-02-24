#!/usr/bin/env python3
"""
Greenhouse cart: capture images from Raspberry Pi Camera Module 3 and upload to an HTTP endpoint.

Design goals:
- "Every plant covered" -> configurable capture rate (Hz) / timelapse (ms)
- Robust over Wi-Fi hiccups -> disk-backed queue (outbox), retry with backoff, delete-on-success
- Single Python file -> runs both capture + upload loops in one process

Requirements:
  pip3 install requests

Works best on Raspberry Pi OS / Debian with rpicam-apps installed (rpicam-jpeg available).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests


# ----------------------------
# Utilities
# ----------------------------

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def disk_free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def list_jpegs_sorted(outbox: str) -> list[str]:
    return sorted(glob.glob(os.path.join(outbox, "*.jpg")))


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class Config:
    endpoint: str
    outbox: str
    cart_id: str
    file_field: str
    extra_fields_json: Optional[str]
    auth_header: Optional[str]

    width: int
    height: int
    quality: int
    timelapse_ms: int
    rpicam_bin: str

    # reliability / safety
    timeout_s: int
    max_outbox_mb: int
    min_free_mb: int
    upload_max_backoff_s: float
    upload_base_backoff_s: float

    # metadata handling
    sidecar_json: bool


# ----------------------------
# Capture manager
# ----------------------------

class CaptureManager:
    """
    Starts rpicam-jpeg in timelapse mode writing to outbox/frame_%06d.jpg.
    You can pause/resume capture based on storage pressure.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.running = False

    def start(self) -> None:
        safe_mkdir(self.cfg.outbox)

        # Use a numeric filename pattern; rpicam-jpeg will increment.
        pattern = os.path.join(self.cfg.outbox, "frame_%06d.jpg")

        cmd = [
            self.cfg.rpicam_bin,
            "-t", "0",
            "--timelapse", str(self.cfg.timelapse_ms),
            "--width", str(self.cfg.width),
            "--height", str(self.cfg.height),
            "-q", str(self.cfg.quality),
            "-o", pattern,
        ]

        # rpicam prints to stderr; keep it attached so journald/systemd captures it
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        self.running = True
        print(f"[capture] started: {' '.join(cmd)}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            print("[capture] stopping...")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.running = False
        print("[capture] stopped")

    def ensure_running(self) -> None:
        if not self.proc:
            self.start()
            return
        if self.proc.poll() is not None:
            # exited unexpectedly — restart
            code = self.proc.returncode
            print(f"[capture] exited (code={code}); restarting...")
            self.start()

    def pause_if_needed(self) -> bool:
        """
        Returns True if capture should be paused.
        Pauses capture when:
          - outbox exceeds max_outbox_mb, OR
          - free space on filesystem drops below min_free_mb
        """
        outbox_bytes = folder_size_bytes(self.cfg.outbox)
        outbox_mb = outbox_bytes / (1024 * 1024)
        free_mb = disk_free_bytes(self.cfg.outbox) / (1024 * 1024)

        if outbox_mb >= self.cfg.max_outbox_mb:
            print(f"[capture] PAUSE: outbox {outbox_mb:.1f}MB >= {self.cfg.max_outbox_mb}MB")
            return True
        if free_mb <= self.cfg.min_free_mb:
            print(f"[capture] PAUSE: free {free_mb:.1f}MB <= {self.cfg.min_free_mb}MB")
            return True
        return False


# ----------------------------
# Upload manager
# ----------------------------

class UploadManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.backoff = cfg.upload_base_backoff_s

        # Parse extra fields once
        self.extra_fields = {}
        if cfg.extra_fields_json:
            try:
                self.extra_fields = json.loads(cfg.extra_fields_json)
                if not isinstance(self.extra_fields, dict):
                    raise ValueError("extra-fields-json must decode to an object/dict")
            except Exception as e:
                raise SystemExit(f"Invalid --extra-fields-json: {e}")

    def _headers(self) -> dict:
        headers = {}
        if self.cfg.auth_header:
            # Example: "Authorization: Bearer <token>"
            k, sep, v = self.cfg.auth_header.partition(":")
            if not sep:
                raise SystemExit("Invalid --auth-header format. Use 'Header-Name: value'")
            headers[k.strip()] = v.strip()
        return headers

    def upload_one(self, jpg_path: str) -> Tuple[bool, str]:
        """
        Upload a single JPEG. On success returns (True, msg). On failure (False, msg).
        Deletes only on success (caller handles delete).
        """
        fname = os.path.basename(jpg_path)
        data = {
            "cart_id": self.cfg.cart_id,
            "filename": fname,
            "timestamp_utc": utc_iso(),
            **self.extra_fields,
        }

        # Optional sidecar metadata file
        if self.cfg.sidecar_json:
            meta_path = os.path.splitext(jpg_path)[0] + ".json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if isinstance(meta, dict):
                        # Merge but don't let it override required fields unless you want that behavior
                        for k, v in meta.items():
                            data.setdefault(k, v)
                except Exception:
                    pass

        try:
            with open(jpg_path, "rb") as f:
                files = {self.cfg.file_field: (fname, f, "image/jpeg")}
                r = self.session.post(
                    self.cfg.endpoint,
                    data=data,
                    files=files,
                    headers=self._headers(),
                    timeout=self.cfg.timeout_s,
                )
        except requests.RequestException as e:
            return False, f"request error: {e}"

        if 200 <= r.status_code < 300:
            return True, f"ok ({r.status_code})"

        # Keep server response short
        body = (r.text or "")[:200].replace("\n", " ")
        return False, f"server {r.status_code}: {body}"

    def run_once(self) -> bool:
        """
        Attempts to upload the oldest file. Returns True if it uploaded at least one.
        """
        paths = list_jpegs_sorted(self.cfg.outbox)
        if not paths:
            return False

        jpg = paths[0]
        ok, msg = self.upload_one(jpg)
        if ok:
            # delete jpg and optional json sidecar
            try:
                os.remove(jpg)
            except OSError as e:
                print(f"[upload] delete failed {jpg}: {e}")

            if self.cfg.sidecar_json:
                side = os.path.splitext(jpg)[0] + ".json"
                if os.path.exists(side):
                    try:
                        os.remove(side)
                    except OSError:
                        pass

            print(f"[upload] {os.path.basename(jpg)} -> {msg}")
            self.backoff = self.cfg.upload_base_backoff_s
            return True

        print(f"[upload] FAIL {os.path.basename(jpg)} -> {msg}")
        time.sleep(self.backoff)
        self.backoff = min(self.backoff * 2, self.cfg.upload_max_backoff_s)
        return False


# ----------------------------
# Main loop
# ----------------------------

STOP = False


def handle_stop(signum, frame):
    global STOP
    STOP = True
    print(f"\n[sys] received signal {signum}, stopping...")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture images with rpicam and upload to HTTP endpoint with retries (single file)."
    )

    # Backend
    p.add_argument("--endpoint", required=True, help="HTTP endpoint URL to POST images to.")
    p.add_argument("--cart-id", default="greenhouse-cart-01", help="Identifier for the cart/device.")
    p.add_argument("--file-field", default="image", help="Multipart field name for the image file (e.g. image/file/frame).")
    p.add_argument("--extra-fields-json", default=None, help='Extra form fields as JSON object, e.g. \'{"row":"A1","side":"left"}\'.')
    p.add_argument("--auth-header", default=None, help='Optional single header, e.g. "Authorization: Bearer TOKEN".')

    # Storage
    p.add_argument("--outbox", default="/data/outbox", help="Folder for captured images (disk queue).")
    p.add_argument("--max-outbox-mb", type=int, default=2048, help="Pause capture if outbox exceeds this size (MB).")
    p.add_argument("--min-free-mb", type=int, default=512, help="Pause capture if free disk space drops below this (MB).")

    # Capture settings
    p.add_argument("--timelapse-ms", type=int, default=250, help="Capture interval in milliseconds (250ms=4Hz).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--quality", type=int, default=80, help="JPEG quality (0-100).")
    p.add_argument("--rpicam-bin", default="rpicam-jpeg", help="Binary name/path for rpicam jpeg tool.")

    # Upload settings
    p.add_argument("--timeout-s", type=int, default=15, help="HTTP request timeout seconds.")
    p.add_argument("--upload-base-backoff-s", type=float, default=1.0)
    p.add_argument("--upload-max-backoff-s", type=float, default=30.0)

    # Optional metadata sidecars
    p.add_argument("--sidecar-json", action="store_true",
                   help="If set, merges fields from frame_XXXXXX.json (if exists) into upload form.")
    return p


def main():
    global STOP

    args = build_argparser().parse_args()

    cfg = Config(
        endpoint=args.endpoint,
        outbox=args.outbox,
        cart_id=args.cart_id,
        file_field=args.file_field,
        extra_fields_json=args.extra_fields_json,
        auth_header=args.auth_header,
        width=args.width,
        height=args.height,
        quality=args.quality,
        timelapse_ms=args.timelapse_ms,
        rpicam_bin=args.rpicam_bin,
        timeout_s=args.timeout_s,
        max_outbox_mb=args.max_outbox_mb,
        min_free_mb=args.min_free_mb,
        upload_max_backoff_s=args.upload_max_backoff_s,
        upload_base_backoff_s=args.upload_base_backoff_s,
        sidecar_json=args.sidecar_json,
    )

    safe_mkdir(cfg.outbox)

    # Quick sanity checks
    if not (0 <= cfg.quality <= 100):
        raise SystemExit("--quality must be 0..100")
    if cfg.timelapse_ms < 50:
        print("[warn] very high capture rate; you may overwhelm SD card / Wi-Fi. Consider >= 100ms.")

    # Ensure rpicam exists
    if shutil_which(cfg.rpicam_bin) is None:
        raise SystemExit(
            f"Cannot find '{cfg.rpicam_bin}' in PATH. "
            f"Install rpicam-apps or pass full path via --rpicam-bin."
        )

    cap = CaptureManager(cfg)
    up = UploadManager(cfg)

    # Start capture
    cap.start()

    print("[sys] running. Ctrl+C to stop.")
    last_pause_state = False

    while not STOP:
        # If storage pressure, pause capture (stop rpicam process)
        should_pause = cap.pause_if_needed()

        if should_pause and cap.running:
            cap.stop()
            last_pause_state = True
        elif (not should_pause):
            # Ensure capture is running
            if not cap.running:
                print("[capture] resuming...")
                cap.start()
            else:
                cap.ensure_running()
            last_pause_state = False

        # Upload at least one file if available
        uploaded = up.run_once()

        # If nothing uploaded (outbox empty), sleep briefly
        if not uploaded:
            time.sleep(0.1)

    # Cleanup
    cap.stop()
    print("[sys] exited cleanly.")


def shutil_which(cmd: str) -> Optional[str]:
    # Small re-implementation of shutil.which to avoid importing full module in minimal environments
    if os.path.isabs(cmd) and os.access(cmd, os.X_OK):
        return cmd
    path = os.environ.get("PATH", "")
    for p in path.split(os.pathsep):
        candidate = os.path.join(p, cmd)
        if os.access(candidate, os.X_OK):
            return candidate
    return None


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    main()