import cv2
import json
import numpy as np

capture_dir = "captures/20260318_133425_078"

depth = cv2.imread(f"{capture_dir}/depth_raw.png", cv2.IMREAD_UNCHANGED)

with open(f"{capture_dir}/meta.json", "r") as f:
    meta = json.load(f)

scale = meta["depth_scale_m_per_unit"]
fx = meta["intrinsics"]["fx"]
fy = meta["intrinsics"]["fy"]
cx = meta["intrinsics"]["cx"]
cy = meta["intrinsics"]["cy"]

u, v = 334, 260
raw_value = int(depth[v, u])
z_m = raw_value * scale

x_m = (u - cx) * z_m / fx
y_m = (v - cy) * z_m / fy

print("raw depth:", raw_value)
print("depth meters:", z_m)
print("3D point:", (x_m, y_m, z_m))