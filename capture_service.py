#!/usr/bin/env python3
import subprocess
import os
import signal
import sys

OUTBOX = "/home/pi/Documents/outbox"
TIMELAPSE_MS = 250      # 4 Hz
WIDTH = 1280
HEIGHT = 720
QUALITY = 80

process = None

def start_capture():
    global process
    os.makedirs(OUTBOX, exist_ok=True)

    cmd = [
        "rpicam-jpeg",
        "-t", "0",
        "--timelapse", str(TIMELAPSE_MS),
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "-q", str(QUALITY),
        "-o", f"{OUTBOX}/frame_%06d.jpg"
    ]

    print("Starting capture...")
    process = subprocess.Popen(cmd)

def stop_capture(signum=None, frame=None):
    global process
    print("Stopping capture...")
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("Capture didn't exist, foce killing...")
            process.kill()
            process.wait()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_capture)
    signal.signal(signal.SIGTERM, stop_capture)
    start_capture()
    process.wait()