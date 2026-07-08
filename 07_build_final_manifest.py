"""
07_build_final_manifest.py

Builds the final training manifest: original_path, edited_path, mask_path,
edit_type, prompt, iou_score -- one row per accepted pair from step 06.
This is the file the (separate, later) TruFor fine-tuning step will consume.

Usage:
  python 07_build_final_manifest.py --dry_run
  python 07_build_final_manifest.py --limit 10
"""

from __future__ import annotations

import argparse
import time

import config
import pipeline_utils as pu


OUTPUT_COLUMNS = ["pair_id", "original_path", "edited_path", "mask_path",
                   "edit_type", "prompt", "iou_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.CROSS_VALIDATED_MANIFEST),
                         help="Manifest from step 06 (default: cross_validated_manifest.csv).")
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("07_build_final_manifest")
    if args.dry_run:
        logger.setLevel("DEBUG")

    limit = pu.resolve_limit(args.limit, args.dry_run)
    logger.info(f"dry_run={args.dry_run} limit={limit} input={args.input}")

    t0 = time.monotonic()
    rows = pu.read_csv_rows(args.input)
    rows = pu.apply_limit(rows, limit)
    logger.info(f"Processing {len(rows)} accepted rows")

    writer = pu.IncrementalCSVWriter(config.FINAL_TRAINING_MANIFEST, OUTPUT_COLUMNS, resume=args.resume)

    n_ok = n_skipped = n_errors = 0
    for row in rows:
        pair_id = row.get("pair_id", "")
        if writer.already_done(pair_id):
            n_skipped += 1
            continue
        try:
            out_row = {
                "pair_id": pair_id,
                "original_path": row["original_path"],
                "edited_path": row["edited_path"],
                "mask_path": row["final_mask_path"],
                "edit_type": row.get("edit_type", ""),
                "prompt": row.get("prompt", ""),
                "iou_score": row.get("iou_score", ""),
            }
            writer.write_row(out_row)
            n_ok += 1
            logger.debug(f"{pair_id}: added to final manifest")
        except Exception as e:
            n_errors += 1
            logger.error(f"{pair_id}: unexpected error -- {e}")

    writer.close()

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "07_build_final_manifest", n_ok, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.FINAL_TRAINING_MANIFEST}")


if __name__ == "__main__":
    main()
