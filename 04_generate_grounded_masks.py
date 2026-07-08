"""
04_generate_grounded_masks.py

Grounding DINO (open-weights grounding-dino-tiny, no API token) finds a box
for the phrase extracted in step 03, then SAM 2.1 hiera-tiny turns that box
into a segmentation mask. Detection is attempted on the EDITED image first
(the phrase usually names what was added/changed there); if nothing is
found, we retry on the ORIGINAL image (covers removal edits, where the
named object only exists pre-edit) and reuse that box against the edited
image -- valid because step 02 already aligned both images to identical
dimensions. Rows with no detection on either image are logged to
config.NO_MATCH_LOG_CSV with a reason instead of crashing the run.

Loads Grounding DINO, then SAM 2.1, one at a time (never Florence-2
simultaneously -- that model already exited with script 03's process).
Both stay resident together for the duration of this script since each
image needs box-then-mask in sequence; explicitly freed at the end.

--dry_run stubs both models (no torch/transformers import, no GPU).

Usage:
  python 04_generate_grounded_masks.py --dry_run
  python 04_generate_grounded_masks.py --limit 10 --resume --time_budget_minutes 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

import config
import pipeline_utils as pu


def stub_detect(img_w: int, img_h: int, phrase: str) -> list[dict]:
    bw, bh = img_w * 0.4, img_h * 0.4
    x1, y1 = (img_w - bw) / 2, (img_h - bh) / 2
    return [{"box": [x1, y1, x1 + bw, y1 + bh], "score": 0.99, "label": phrase}]


def stub_mask_from_box(img_w: int, img_h: int, box: list[float]) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=bool)
    x1, y1, x2, y2 = [int(round(c)) for c in box]
    mask[max(y1, 0):min(y2, img_h), max(x1, 0):min(x2, img_w)] = True
    return mask


def save_mask(mask: np.ndarray, path) -> None:
    Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.PHRASE_MANIFEST),
                         help="Manifest from step 03 (default: phrase_manifest.csv).")
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("04_generate_grounded_masks")
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

    out_fields = list(rows[0].keys()) + [
        "grounded_mask_path", "detection_source", "detection_score", "detection_box",
    ] if rows else []
    no_match_fields = (list(rows[0].keys()) + ["reason"]) if rows else ["pair_id", "reason"]

    writer = pu.IncrementalCSVWriter(config.GROUNDED_MASK_MANIFEST, out_fields, resume=args.resume)
    no_match_writer = pu.IncrementalCSVWriter(config.NO_MATCH_LOG_CSV, no_match_fields, resume=args.resume)

    dino = sam2 = None
    if not args.dry_run and rows:
        import models
        logger.info("Loading Grounding DINO...")
        dino = models.GroundingDinoWrapper(logger=logger)
        logger.info("Loading SAM 2.1 hiera-tiny...")
        sam2 = models.Sam2Wrapper(logger=logger)

    n_ok = n_no_match = n_errors = n_skipped = 0
    try:
        for row in rows:
            pair_id = row.get("pair_id", "")
            if writer.already_done(pair_id) or no_match_writer.already_done(pair_id):
                n_skipped += 1
                continue
            if budget.expired():
                logger.warning(f"Time budget of {args.time_budget_minutes} min reached. "
                                f"Stopping early; progress is saved.")
                break

            phrase = row.get("extracted_phrase", "")
            if row.get("phrase_status") != "ok" or not phrase:
                no_match_writer.write_row({**row, "reason": "no phrase from step 03"})
                n_no_match += 1
                cost_est.tick(total_planned=len(rows))
                continue

            try:
                edited_path = pu.resolve_image_path(
                    row.get("aligned_edited_path") or row["edited_path"]
                )
                orig_path = pu.resolve_image_path(row["original_path"])

                with Image.open(edited_path) as edited_img:
                    edited_img = edited_img.convert("RGB")
                    w, h = edited_img.size

                    if args.dry_run:
                        dets = stub_detect(w, h, phrase)
                        source = "edited[DRY_RUN]"
                    else:
                        dets = dino.detect(edited_img, phrase)
                        source = "edited"

                    if not dets:
                        with Image.open(orig_path) as orig_img:
                            orig_img = orig_img.convert("RGB")
                            if args.dry_run:
                                dets = stub_detect(w, h, phrase)
                                source = "original[DRY_RUN]"
                            else:
                                dets = dino.detect(orig_img, phrase)
                                source = "original"

                    if not dets:
                        no_match_writer.write_row({
                            **row, "reason": f"Grounding DINO found no box for '{phrase}' "
                                              f"on either edited or original image"
                        })
                        n_no_match += 1
                        logger.warning(f"{pair_id}: NO MATCH for phrase '{phrase}'")
                        cost_est.tick(total_planned=len(rows))
                        continue

                    best = max(dets, key=lambda d: d["score"])

                    if args.dry_run:
                        mask = stub_mask_from_box(w, h, best["box"])
                    else:
                        mask = sam2.mask_from_box(edited_img, best["box"])

                    mask_path = config.GROUNDED_MASKS_DIR / f"{pair_id}_grounded_mask.png"
                    save_mask(mask, mask_path)

                out_row = dict(row)
                out_row.update({
                    "grounded_mask_path": str(mask_path),
                    "detection_source": source,
                    "detection_score": round(best["score"], 4),
                    "detection_box": ",".join(f"{c:.1f}" for c in best["box"]),
                })
                writer.write_row(out_row)
                n_ok += 1
                logger.debug(f"{pair_id}: mask saved, source={source}, score={best['score']:.3f}")

            except Exception as e:
                n_errors += 1
                logger.error(f"{pair_id}: unexpected error -- {e}")

            cost_est.tick(total_planned=len(rows))
    finally:
        writer.close()
        no_match_writer.close()
        if dino is not None or sam2 is not None:
            import models
            models.free_model(dino, sam2)
            logger.info("Grounding DINO + SAM 2.1 unloaded, CUDA cache cleared.")

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "04_generate_grounded_masks", n_ok, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} no_match={n_no_match} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.GROUNDED_MASK_MANIFEST}")
    logger.info(f"-> {config.NO_MATCH_LOG_CSV}")


if __name__ == "__main__":
    main()
