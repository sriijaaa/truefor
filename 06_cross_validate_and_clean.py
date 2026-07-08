"""
06_cross_validate_and_clean.py

Cross-validates the Grounding-DINO+SAM2 mask (primary) against the SSIM
diff mask (secondary/validation) via IoU. Pairs at or above
config.IOU_ACCEPT_THRESHOLD are accepted: the grounded mask (still the
source of truth) gets morphological open+close and small-connected-
component removal (< config.MIN_COMPONENT_AREA_PX px) and is written to
outputs/masks/accepted/. Pairs below threshold are rejected: both raw masks
are copied to outputs/masks/review/ for manual inspection and the pair is
recorded in rejected_manifest.csv with the reason.

Pure CPU (skimage), no GPU model.

Usage:
  python 06_cross_validate_and_clean.py --dry_run
  python 06_cross_validate_and_clean.py --limit 10
"""

from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
from PIL import Image
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

import config
import pipeline_utils as pu


def load_bool_mask(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        # Shouldn't happen post-alignment, but don't crash a whole run over it.
        h = min(mask_a.shape[0], mask_b.shape[0])
        w = min(mask_a.shape[1], mask_b.shape[1])
        mask_a = mask_a[:h, :w]
        mask_b = mask_b[:h, :w]
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    selem = disk(3)
    cleaned = binary_opening(mask, selem)
    cleaned = binary_closing(cleaned, selem)
    cleaned = remove_small_objects(cleaned, min_size=min_area)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounded_input", default=str(config.GROUNDED_MASK_MANIFEST))
    parser.add_argument("--diff_input", default=str(config.DIFF_MASK_MANIFEST))
    parser.add_argument("--iou_threshold", type=float, default=config.IOU_ACCEPT_THRESHOLD)
    parser.add_argument("--min_component_area", type=int, default=config.MIN_COMPONENT_AREA_PX)
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("06_cross_validate_and_clean")
    if args.dry_run:
        logger.setLevel("DEBUG")

    limit = pu.resolve_limit(args.limit, args.dry_run)
    logger.info(f"dry_run={args.dry_run} limit={limit} iou_threshold={args.iou_threshold} "
                f"min_component_area={args.min_component_area}")

    t0 = time.monotonic()
    grounded_rows = pu.read_csv_rows(args.grounded_input)
    diff_rows = pu.read_csv_rows(args.diff_input)
    diff_by_id = {r["pair_id"]: r for r in diff_rows}

    grounded_rows = pu.apply_limit(grounded_rows, limit)
    logger.info(f"Processing {len(grounded_rows)} grounded rows against {len(diff_rows)} diff rows")

    accepted_fields = list(grounded_rows[0].keys()) + ["diff_mask_path", "iou_score", "final_mask_path"] \
        if grounded_rows else []
    rejected_fields = accepted_fields + ["reason"]

    accepted_writer = pu.IncrementalCSVWriter(config.CROSS_VALIDATED_MANIFEST, accepted_fields, resume=args.resume)
    rejected_writer = pu.IncrementalCSVWriter(config.REJECTED_MANIFEST, rejected_fields, resume=args.resume)

    n_accepted = n_rejected = n_errors = n_skipped = 0
    for row in grounded_rows:
        pair_id = row.get("pair_id", "")
        if accepted_writer.already_done(pair_id) or rejected_writer.already_done(pair_id):
            n_skipped += 1
            continue

        try:
            diff_row = diff_by_id.get(pair_id)
            if diff_row is None or not diff_row.get("diff_mask_path"):
                rejected_writer.write_row({**row, "diff_mask_path": "", "iou_score": "",
                                            "final_mask_path": "", "reason": "no diff mask available"})
                n_rejected += 1
                logger.warning(f"{pair_id}: REJECTED -- no diff mask available")
                continue

            grounded_mask = load_bool_mask(row["grounded_mask_path"])
            diff_mask = load_bool_mask(diff_row["diff_mask_path"])
            score = iou(grounded_mask, diff_mask)

            if score >= args.iou_threshold:
                cleaned = clean_mask(grounded_mask, args.min_component_area)
                final_path = config.ACCEPTED_MASKS_DIR / f"{pair_id}_final_mask.png"
                Image.fromarray((cleaned.astype(np.uint8)) * 255, mode="L").save(final_path)

                out_row = dict(row)
                out_row.update({
                    "diff_mask_path": diff_row["diff_mask_path"],
                    "iou_score": round(score, 4),
                    "final_mask_path": str(final_path),
                })
                accepted_writer.write_row(out_row)
                n_accepted += 1
                logger.debug(f"{pair_id}: ACCEPTED iou={score:.4f}")
            else:
                shutil.copy2(row["grounded_mask_path"],
                              config.REVIEW_MASKS_DIR / f"{pair_id}_grounded_rejected.png")
                shutil.copy2(diff_row["diff_mask_path"],
                              config.REVIEW_MASKS_DIR / f"{pair_id}_diff_rejected.png")
                out_row = dict(row)
                out_row.update({
                    "diff_mask_path": diff_row["diff_mask_path"],
                    "iou_score": round(score, 4),
                    "final_mask_path": "",
                    "reason": f"IoU {score:.4f} below threshold {args.iou_threshold}",
                })
                rejected_writer.write_row(out_row)
                n_rejected += 1
                logger.warning(f"{pair_id}: REJECTED iou={score:.4f} < {args.iou_threshold}")

        except Exception as e:
            n_errors += 1
            logger.error(f"{pair_id}: unexpected error -- {e}")

    accepted_writer.close()
    rejected_writer.close()

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "06_cross_validate_and_clean", n_accepted + n_rejected, n_skipped, n_errors, elapsed)
    logger.info(f"accepted={n_accepted} rejected={n_rejected} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.CROSS_VALIDATED_MANIFEST}")
    logger.info(f"-> {config.REJECTED_MANIFEST}")


if __name__ == "__main__":
    main()
