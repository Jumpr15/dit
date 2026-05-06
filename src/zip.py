import os
import zipfile
from huggingface_hub import hf_hub_download

# --- Configuration ---
REPO_ID = "aipracticecafe/anime-faces-256px"  # Replace with your actual repo ID
FILENAME = "dataset.zip" # Replace with your actual zip filename
OUTPUT_DIR = "./danbooru_faces"
# ---------------------

# 1. Download the zip file
print(f"Downloading {FILENAME} from {REPO_ID}...")
zip_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")

# 2. Unzip the content
print(f"Extracting to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(OUTPUT_DIR)