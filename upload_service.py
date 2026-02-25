import requests
from pathlib import Path
import os

API_URL = "https://deenp03-capstone-backend.hf.space/api/classify"
IMAGE_DIR = Path("/home/pi/Documents/captures")

for image_path in IMAGE_DIR.glob("*.jpg"):
    print(f"Sending {image_path.name}...")

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            print("Uploaded successfully. Deleting image.")
            os.remove(image_path)
        else:
            print("Upload failed:", response.status_code)

    except Exception as e:
        print("Error sending image:", e)

print("Done.")