#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import select
import sys

SAVE_DIR = Path("./captures")
SAVE_DIR.mkdir(exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

align = rs.align(rs.stream.color)

pipeline.start(config)

print("Press ENTER to capture image")
print("Press q in the preview window to quit")

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

try:

    # warmup
    for _ in range(30):
        pipeline.wait_for_frames()

    while True:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET
        )

        preview = np.hstack((color, depth_colormap))

        cv2.imshow("RealSense Preview (RGB | Depth)", preview)

        # check keyboard for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # check if ENTER pressed in terminal
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()

            ts = timestamp()

            cv2.imwrite(str(SAVE_DIR / f"{ts}_color.png"), color)
            cv2.imwrite(str(SAVE_DIR / f"{ts}_depth.png"), depth)

            np.save(str(SAVE_DIR / f"{ts}_depth.npy"), depth)

            print(f"Saved capture {ts}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


"""
cd ~/librealsense/build
rm -f CMakeCache.txt

# Run cmake and capture the full output to see why Python bindings are disabled
cmake .. -DBUILD_PYTHON_BINDINGS:BOOL=ON -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i -A2 -B2 "python\|binding\|pybind\|wrap"
"""