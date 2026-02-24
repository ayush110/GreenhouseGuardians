#!/usr/bin/env python3
import os
import time
import glob
import requests
from datetime import datetime, timezone

OUTBOX = "/data/outbox"
ENDPOINT = "https://YOUR_BACKEND/upload"   # CHANGE THIS
CART_ID = "cart-01"
FILE_FIELD = "image"
TIMEOUT = 15

session = requests.Session()

def utc_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def upload_file(path):
    filename = os.path.basename(path)

    data = {
        "cart_id": CART_ID,
        "filename": filename,
        "timestamp_utc": utc_iso()
    }

    try:
        with open(path, "rb") as f:
            files = {FILE_FIELD: (filename, f, "image/jpeg")}
            r = session.post(ENDPOINT, data=data, files=files, timeout=TIMEOUT)
    except requests.RequestException as e:
        print(f"Upload error: {e}")
        return False

    if 200 <= r.status_code < 300:
        print(f"Uploaded {filename}")
        return True

    print(f"Server error {r.status_code}: {r.text[:100]}")
    return False

def main():
    os.makedirs(OUTBOX, exist_ok=True)
    backoff = 1

    while True:
        files = sorted(glob.glob(f"{OUTBOX}/*.jpg"))

        if not files:
            time.sleep(0.2)
            continue

        path = files[0]

        if upload_file(path):
            os.remove(path)
            backoff = 1
        else:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

if __name__ == "__main__":
    main()