"""
Dev-only helper (not part of the pipeline): generates a tiny synthetic
original/edited image pair set + metadata.csv so the pipeline can be
smoke-tested end-to-end locally without any real dataset or GPU.
"""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

IMAGES_DIR = config.IMAGES_ROOT
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {"pair_id": "p001", "edit_type": "object_addition",
     "prompt": "add a red hat on the person's head", "kind": "local", "resize_edited": False},
    {"pair_id": "p002", "edit_type": "object_removal",
     "prompt": "remove the blue car from the street", "kind": "local", "resize_edited": False},
    {"pair_id": "p003", "edit_type": "style_transfer",
     "prompt": "apply a vintage sepia color grade to the whole image", "kind": "global", "resize_edited": False},
    {"pair_id": "p004", "edit_type": "attribute_change",
     "prompt": "change the shirt color to green", "kind": "local", "resize_edited": True},
    {"pair_id": "p005", "edit_type": "lighting",
     "prompt": "make the overall lighting brighter and more cinematic", "kind": "global", "resize_edited": False},
]

metadata_rows = []
for r in rows:
    orig = Image.new("RGB", (256, 256), color=(120, 140, 160))
    d = ImageDraw.Draw(orig)
    d.rectangle([80, 80, 176, 176], fill=(200, 60, 60))  # a "subject" block
    orig_path = IMAGES_DIR / f"{r['pair_id']}_original.png"
    orig.save(orig_path)

    edited = orig.copy()
    d2 = ImageDraw.Draw(edited)
    d2.ellipse([90, 60, 166, 110], fill=(255, 215, 0))  # simulate a localized edit (e.g. "hat")
    if r["resize_edited"]:
        edited = edited.resize((240, 260))  # force an alignment mismatch case
    edited_path = IMAGES_DIR / f"{r['pair_id']}_edited.png"
    edited.save(edited_path)

    metadata_rows.append({
        "pair_id": r["pair_id"],
        "original_path": str(orig_path),
        "edited_path": str(edited_path),
        "edit_type": r["edit_type"],
        "prompt": r["prompt"],
    })

with open(config.INPUT_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["pair_id", "original_path", "edited_path", "edit_type", "prompt"])
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Wrote {len(metadata_rows)} synthetic pairs to {config.INPUT_METADATA_CSV}")
print(f"Images in {IMAGES_DIR}")
