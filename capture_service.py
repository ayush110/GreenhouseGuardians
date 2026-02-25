from picamera2 import Picamera2
import time
import os
import requests
import mimetypes
from pathlib import Path

INTERVAL = 1.0   # seconds
SAVE_DIR = Path("/home/pi/Documents/captures")
API_URL = "https://deenp03-capstone-backend.hf.space/api/classify"


def upload_image(image_path: Path) -> bool:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, mime_type)}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        print(f"  Uploaded successfully.")
        return True

    print(f"  Upload failed: {response.status_code} - {response.text}")
    return False


def upload_pending():
    """Upload any images left over from previous runs."""
    pending = sorted(SAVE_DIR.glob("*.jpg"))
    if not pending:
        return
    print(f"Found {len(pending)} pending image(s), uploading...")
    for image_path in pending:
        print(f"  Sending {image_path.name}...")
        try:
            if upload_image(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"  Error: {e}")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    upload_pending()

    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (2304, 1296)}
    )
    picam2.configure(config)
    picam2.start()

    # Let camera auto-exposure settle
    time.sleep(2)

    next_capture_time = time.monotonic()
    image_count = 0

    try:
        while True:
            now = time.monotonic()
            if now < next_capture_time:
                time.sleep(next_capture_time - now)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = SAVE_DIR / f"img_{timestamp}_{image_count:05d}.jpg"

            picam2.capture_file(str(filepath))
            print(f"Captured {filepath.name}")

            try:
                if upload_image(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"  Upload error (image kept on disk): {e}")

            image_count += 1
            next_capture_time += INTERVAL

    except KeyboardInterrupt:
        print("Stopping capture")

    finally:
        picam2.stop()


if __name__ == "__main__":
    main()