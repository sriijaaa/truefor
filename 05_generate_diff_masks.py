"""
05_generate_diff_masks.py

Secondary VALIDATION signal only (per spec, not the primary mask source):
computes a per-pixel SSIM difference map between the aligned original/edited
pair, Otsu-thresholds it into a binary "changed pixels" mask, and saves it.
Step 06 will cross-check this against the Grounding-DINO+SAM2 mask via IoU.

Pure CPU (skimage), no GPU model involved -- runs on ALIGNED_MANIFEST (a
superset of the rows that made it through grounding) so 06 has a diff mask
to compare against for every grounded mask that exists.

Usage:
  python 05_generate_diff_masks.py --dry_run
  python 05_generate_diff_masks.py --limit 10 --resume
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim

import config
import pipeline_utils as pu


def compute_diff_mask(orig_img: Image.Image, edited_img: Image.Image) -> tuple[np.ndarray, float, float]:
    orig_gray = np.array(orig_img.convert("L"), dtype=np.float64)
    edited_gray = np.array(edited_img.convert("L"), dtype=np.float64)

    score, full_ssim = ssim(orig_gray, edited_gray, full=True, data_range=255.0)
    diff_map = 1.0 - full_ssim  # 0 = identical, 1 = maximally different

    try:
        thresh = threshold_otsu(diff_map)
    except ValueError:
        # Otsu fails on a constant image (identical pair); treat as "no change".
        thresh = 1.0

    mask = diff_map > thresh
    return mask, float(score), float(thresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.ALIGNED_MANIFEST),
                         help="Manifest from step 02 (default: aligned_manifest.csv).")
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("05_generate_diff_masks")
    if args.dry_run:
        logger.setLevel("DEBUG")

    limit = pu.resolve_limit(args.limit, args.dry_run)
    cost_est = pu.CostEstimator(logger)
    logger.info(f"dry_run={args.dry_run} limit={limit} input={args.input}")

    t0 = time.monotonic()
    rows = pu.read_csv_rows(args.input)
    rows = pu.apply_limit(rows, limit)
    logger.info(f"Processing {len(rows)} rows")

    out_fields = list(rows[0].keys()) + ["diff_mask_path", "ssim_score", "otsu_threshold"] \
        if rows else []
    writer = pu.IncrementalCSVWriter(config.DIFF_MASK_MANIFEST, out_fields, resume=args.resume)

    n_ok = n_errors = n_skipped = 0
    for row in rows:
        pair_id = row.get("pair_id", "")
        if writer.already_done(pair_id):
            n_skipped += 1
            continue

        try:
            orig_path = pu.resolve_image_path(row["original_path"])
            edited_path = pu.resolve_image_path(row.get("aligned_edited_path") or row["edited_path"])

            with Image.open(orig_path) as orig_img, Image.open(edited_path) as edited_img:
                mask, score, thresh = compute_diff_mask(orig_img, edited_img)

            mask_path = config.DIFF_MASKS_DIR / f"{pair_id}_diff_mask.png"
            Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(mask_path)

            out_row = dict(row)
            out_row.update({
                "diff_mask_path": str(mask_path),
                "ssim_score": round(score, 4),
                "otsu_threshold": round(thresh, 4),
            })
            writer.write_row(out_row)
            n_ok += 1
            logger.debug(f"{pair_id}: ssim={score:.4f} otsu_thresh={thresh:.4f} "
                         f"changed_px={int(mask.sum())}")

        except Exception as e:
            n_errors += 1
            logger.error(f"{pair_id}: unexpected error -- {e}")

        cost_est.tick(total_planned=len(rows))

    writer.close()

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "05_generate_diff_masks", n_ok, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.DIFF_MASK_MANIFEST}")


if __name__ == "__main__":
    main()
