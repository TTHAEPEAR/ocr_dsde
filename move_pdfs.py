import os
import shutil
from pathlib import Path

source_dir = Path("data/drive_download")
dest_dir = Path("data/raw_pdfs")

if not source_dir.exists():
    print(f"Source directory {source_dir} not found. Ensure download is finished.")
    exit(1)

count = {
    "5_18": 0, "5_18_party": 0,
    "5_16": 0, "5_16_party": 0,
    "5_17": 0, "5_17_party": 0,
}

print("Scanning downloaded files and organizing them...")

for root, dirs, files in os.walk(source_dir):
    for filename in files:
        if filename.lower().endswith('.pdf'):
            name = filename.replace("ทับ", "/").replace(" ", "").replace("_", "")
            target_folder = None
            
            if "18บช" in name:
                target_folder = "5_18_party"
            elif "18" in name:
                target_folder = "5_18"
            elif "16บช" in name:
                target_folder = "5_16_party"
            elif "16" in name:
                target_folder = "5_16"
            elif "17บช" in name:
                target_folder = "5_17_party"
            elif "17" in name:
                target_folder = "5_17"
            
            if target_folder:
                # Prepend parent folder path to avoid name collisions
                parent_folder = Path(root).name
                new_filename = f"{parent_folder}_{filename}"
                dest_path = dest_dir / target_folder / new_filename
                
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(os.path.join(root, filename), dest_path)
                count[target_folder] += 1

print("=== File Organization Summary ===")
for folder, c in count.items():
    print(f"  {folder}: {c} files copied")
print("Done! You can now run python 02_preprocess/preprocess_images.py")
