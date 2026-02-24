from picamera2 import Picamera2
import time
from pathlib import Path

INTERVAL = 1.0   # seconds
SAVE_DIR = "captures"

def main():
    # Create output folder
    Path(SAVE_DIR).mkdir(exist_ok=True)

    # Initialize camera
    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (2304, 1296)}  # good balance speed/quality
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

            # Wait until next scheduled capture
            if now < next_capture_time:
                time.sleep(next_capture_time - now)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/img_{timestamp}_{image_count:05d}.jpg"

            # Capture image
            picam2.capture_file(filename)
            print(f"Saved {filename}")

            image_count += 1
            next_capture_time += INTERVAL

    except KeyboardInterrupt:
        print("Stopping capture")

    finally:
        picam2.stop()

if __name__ == "__main__":
    main()