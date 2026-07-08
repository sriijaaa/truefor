"""
Dev-only helper (not part of the pipeline): builds tiny synthetic pos.zip /
neg.zip so 00_build_metadata_from_pico_zips.py can be smoke-tested locally
without the real ~1.8GB Drive download.
"""
import io
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

config.ensure_dirs()
raw_dir = config.GDRIVE_RAW_DIR
raw_dir.mkdir(parents=True, exist_ok=True)

ids = ["000000", "000001", "000002", "000003", "000004"]
# one id present in pos.zip only, to test the intersection logic
pos_only_id = "999999"

with zipfile.ZipFile(raw_dir / "pos.zip", "w") as zf_pos, \
     zipfile.ZipFile(raw_dir / "neg.zip", "w") as zf_neg:
    for i, pid in enumerate(ids + [pos_only_id]):
        orig = Image.new("RGB", (200, 200), color=(100 + i * 10, 120, 140))
        d = ImageDraw.Draw(orig)
        d.rectangle([60, 60, 140, 140], fill=(180, 60, 60))
        buf = io.BytesIO()
        orig.save(buf, format="JPEG")
        zf_pos.writestr(f"{pid}.jpg", buf.getvalue())

        if pid == pos_only_id:
            continue  # deliberately absent from neg.zip

        edited = orig.copy()
        d2 = ImageDraw.Draw(edited)
        d2.ellipse([70, 40, 130, 80], fill=(255, 215, 0))
        buf2 = io.BytesIO()
        edited.save(buf2, format="JPEG")
        zf_neg.writestr(f"{pid}.jpg", buf2.getvalue())

print(f"Wrote synthetic pos.zip ({len(ids) + 1} files) and neg.zip ({len(ids)} files) to {raw_dir}")
