from picamera2 import Picamera2, Preview
from datetime import datetime
from pathlib import Path
import time

# Folder to save captured images
SAVE_DIR = Path("/home/pi/Documents/single_captures")
SAVE_DIR.mkdir(exist_ok=True)

def main():
    picam2 = Picamera2()

    # Preview/display configuration
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        lores={"size": (640, 480)},
        display="main"
    )
    picam2.configure(config)

    # Start camera + preview
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    # Let camera settle
    time.sleep(2)

    print("Camera preview started.")
    print("Commands:")
    print("  c + Enter  -> capture image")
    print("  q + Enter  -> quit")

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "c":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = SAVE_DIR / f"image_{timestamp}.jpg"
                picam2.capture_file(str(filename))
                print(f"Saved: {filename}")

            elif cmd == "q":
                print("Exiting.")
                break

            else:
                print("Unknown command. Use 'c' to capture or 'q' to quit.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        picam2.stop_preview()
        picam2.stop()

if __name__ == "__main__":
    main()