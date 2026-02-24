#!/usr/bin/env python3
"""Capture images at fixed intervals into an outbox folder."""

import subprocess
import os
import signal
import sys
import time
import argparse

OUTBOX = "/home/pi/Documents/outbox"
INTERVAL_S = 0.25
WIDTH = 1280
HEIGHT = 720
QUALITY = 80

process = None
shutting_down = False


def parse_args():
    parser = argparse.ArgumentParser(description="Capture images into outbox at a fixed interval.")
    parser.add_argument("--outbox", default=OUTBOX, help="Directory where images are written")
    parser.add_argument(
        "--interval",
        type=float,
        default=INTERVAL_S,
        help="Seconds between captures (example: 1.0 = 1 image/sec)",
    )
    parser.add_argument("--width", type=int, default=WIDTH, help="Image width")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Image height")
    parser.add_argument("--quality", type=int, default=QUALITY, help="JPEG quality (1-100)")
    return parser.parse_args()


def start_capture(args):
    global process
    os.makedirs(args.outbox, exist_ok=True)
    timelapse_ms = max(1, int(args.interval * 1000))

    cmd = [
        "rpicam-jpeg",
        "-t", "0",
        "--timelapse",
        str(timelapse_ms),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "-q",
        str(args.quality),
        "-o",
        f"{args.outbox}/frame_%06d.jpg",
    ]

    print(f"Starting capture every {args.interval}s -> {args.outbox}")
    # Start in its own process group so we can kill everything on Ctrl+C.
    process = subprocess.Popen(cmd, preexec_fn=os.setsid)


def stop_capture(signum=None, frame=None):
    global process, shutting_down
    if shutting_down:
        return
    shutting_down = True
    print("Stopping capture...")
    if process and process.poll() is None:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("Capture did not exit in time, force killing...")
            os.killpg(pgid, signal.SIGKILL)
            process.wait()
    sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    signal.signal(signal.SIGINT, stop_capture)
    signal.signal(signal.SIGTERM, stop_capture)
    start_capture(args)
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError("Capture process exited unexpectedly")
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_capture()
