import requests
from pathlib import Path
import os
import mimetypes

API_URL = "https://deenp03-capstone-backend.hf.space/api/classify"
IMAGE_DIR = Path("/home/pi/Documents/captures")

for image_path in IMAGE_DIR.glob("*.jpg"):
    print(f"Sending {image_path.name}...")

    try:
        mime_type = mimetypes.guess_type(str(image_path))[0]
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, mime_type)}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            print("Uploaded successfully. Deleting image.")
            os.remove(image_path)
        else:
            print(f"Upload failed: {response.status_code} - {response.text}")

    except Exception as e:
        print("Error sending image:", e)

print("Done.")