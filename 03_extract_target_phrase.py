"""
03_extract_target_phrase.py

DATA SOURCE NOTE: this data source (pos.zip/neg.zip pairs, see
00_build_metadata_from_pico_zips.py) has NO prompt text, so this step no
longer grounds a given sentence. Instead it runs Florence-2 UNPROMPTED
(config.FLORENCE2_REGION_TASK, <DENSE_REGION_CAPTION>) on the EDITED image
to get candidate (phrase, box) region proposals, then picks whichever
candidate has the highest IoU against the SSIM diff mask's bounding box
(from step 05) as the "extracted phrase" + rough box. This is why 05 (diff
masks) must run BEFORE this step now -- see run_pipeline.sh's reordering.

If the diff mask is empty (no detected change) or no candidate region
overlaps it well enough (config.REGION_MATCH_MIN_IOU), the row is written
with an empty phrase and a status describing why, so step 04 can skip it
without crashing and it stays visible in the manifest for review.

--dry_run stubs Florence-2 entirely (no torch/transformers import, no GPU):
it fabricates a candidate region equal to the diff bbox itself, so the
IoU-matching logic still gets exercised for free.

Usage:
  python 03_extract_target_phrase.py --dry_run
  python 03_extract_target_phrase.py --limit 10 --resume --time_budget_minutes 15
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import config
import pipeline_utils as pu


def diff_mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stub_propose_regions(target_bbox: tuple[float, float, float, float]) -> dict:
    """Used only in --dry_run. No model, no GPU -- fabricates a candidate
    region equal to the diff bbox so the IoU-matching logic still runs."""
    return {"bboxes": [list(target_bbox)], "labels": ["stubbed_region[DRY_RUN]"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.DIFF_MASK_MANIFEST),
                         help="Manifest from step 05 (default: diff_mask_manifest.csv) -- "
                              "05 must run before this step for this data source.")
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("03_extract_target_phrase")
    if args.dry_run:
        logger.setLevel("DEBUG")

    limit = pu.resolve_limit(args.limit, args.dry_run)
    budget = pu.TimeBudget(args.time_budget_minutes)
    cost_est = pu.CostEstimator(logger)
    logger.info(f"dry_run={args.dry_run} limit={limit} input={args.input} "
                f"time_budget_minutes={args.time_budget_minutes}")

    t0 = time.monotonic()
    rows = pu.read_csv_rows(args.input)
    rows = pu.apply_limit(rows, limit)
    logger.info(f"Processing {len(rows)} rows")

    out_fields = list(rows[0].keys()) + ["extracted_phrase", "rough_box_x1y1x2y2", "phrase_status"] \
        if rows else []
    writer = pu.IncrementalCSVWriter(config.PHRASE_MANIFEST, out_fields, resume=args.resume)

    florence = None
    if not args.dry_run and rows:
        logger.info("Loading Florence-2 (real run, this is the only model this script loads)...")
        import models
        florence = models.Florence2Wrapper(logger=logger)

    n_ok = n_no_match = n_errors = n_skipped = 0
    try:
        from PIL import Image

        for row in rows:
            pair_id = row.get("pair_id", "")
            if writer.already_done(pair_id):
                n_skipped += 1
                continue
            if budget.expired():
                logger.warning(f"Time budget of {args.time_budget_minutes} min reached. "
                                f"Stopping early; progress is saved.")
                break

            try:
                edited_path = pu.resolve_image_path(
                    row.get("aligned_edited_path") or row["edited_path"]
                )
                diff_mask = np.array(Image.open(row["diff_mask_path"]).convert("L")) > 127
                target_bbox = diff_mask_bbox(diff_mask)

                out_row = dict(row)

                if target_bbox is None:
                    out_row["extracted_phrase"] = ""
                    out_row["rough_box_x1y1x2y2"] = ""
                    out_row["phrase_status"] = "no_change_detected"
                    n_no_match += 1
                    logger.warning(f"{pair_id}: diff mask is empty, nothing to ground")
                    writer.write_row(out_row)
                    cost_est.tick(total_planned=len(rows))
                    continue

                with Image.open(edited_path) as edited_img:
                    edited_img = edited_img.convert("RGB")
                    if args.dry_run:
                        logger.debug(f"{pair_id}: [DRY RUN] stubbing Florence-2 call")
                        parsed = stub_propose_regions(target_bbox)
                    else:
                        parsed = florence.propose_regions(edited_img)

                bboxes = parsed.get("bboxes", [])
                labels = parsed.get("labels", [])

                if not bboxes:
                    out_row["extracted_phrase"] = ""
                    out_row["rough_box_x1y1x2y2"] = ""
                    out_row["phrase_status"] = "no_regions_found"
                    n_no_match += 1
                    logger.warning(f"{pair_id}: Florence-2 proposed no regions at all")
                    writer.write_row(out_row)
                    cost_est.tick(total_planned=len(rows))
                    continue

                ious = [box_iou(tuple(b), target_bbox) for b in bboxes]
                best_idx = int(np.argmax(ious))
                best_iou = ious[best_idx]

                if best_iou < config.REGION_MATCH_MIN_IOU:
                    out_row["extracted_phrase"] = ""
                    out_row["rough_box_x1y1x2y2"] = ""
                    out_row["phrase_status"] = "no_region_match"
                    n_no_match += 1
                    logger.warning(f"{pair_id}: best candidate region IoU {best_iou:.3f} "
                                    f"below threshold {config.REGION_MATCH_MIN_IOU} vs diff bbox")
                else:
                    out_row["extracted_phrase"] = labels[best_idx]
                    out_row["rough_box_x1y1x2y2"] = ",".join(f"{c:.1f}" for c in bboxes[best_idx])
                    out_row["phrase_status"] = "ok"
                    n_ok += 1
                    logger.debug(f"{pair_id}: phrase='{labels[best_idx]}' "
                                 f"box={bboxes[best_idx]} iou_vs_diff={best_iou:.3f}")

                writer.write_row(out_row)

            except Exception as e:
                n_errors += 1
                logger.error(f"{pair_id}: unexpected error -- {e}")

            cost_est.tick(total_planned=len(rows))
    finally:
        writer.close()
        if florence is not None:
            import models
            models.free_model(florence.model, florence.processor)
            logger.info("Florence-2 unloaded, CUDA cache cleared.")

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "03_extract_target_phrase", n_ok + n_no_match, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} no_match={n_no_match} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.PHRASE_MANIFEST}")


if __name__ == "__main__":
    main()
