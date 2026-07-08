"""
03_extract_target_phrase.py

Runs Florence-2's <CAPTION_TO_PHRASE_GROUNDING> task against the ORIGINAL
image using the full edit `prompt` as input text. Florence-2 parses the
prompt itself and returns the noun phrase(s) it can ground in the image,
each with a box -- this is what gives us a clean "target phrase" + rough
box out of a full natural-language sentence, instead of writing our own NLP.

We keep the highest-confidence (first-returned) phrase/box pair. If
Florence-2 returns nothing groundable, the row is still written with an
empty phrase and status="no_phrase_found" so step 04 can skip it without
crashing and it stays visible in the manifest for review.

--dry_run stubs Florence-2 entirely (no torch/transformers import, no GPU)
using a trivial heuristic so path/config/CSV-schema bugs surface for free.

Usage:
  python 03_extract_target_phrase.py --dry_run
  python 03_extract_target_phrase.py --limit 10 --resume --time_budget_minutes 15
"""

from __future__ import annotations

import argparse
import time

import config
import pipeline_utils as pu


def stub_ground_phrase(prompt: str, img_w: int, img_h: int) -> dict:
    """Used only in --dry_run. No model, no GPU -- just enough structure to
    exercise the CSV schema and downstream code paths for free."""
    words = [w.strip(".,!?") for w in (prompt or "").split() if len(w) > 3]
    phrase = " ".join(words[:3]) if words else "object"
    # fake centered box covering ~25% of the image
    bw, bh = img_w * 0.5, img_h * 0.5
    x1, y1 = (img_w - bw) / 2, (img_h - bh) / 2
    return {"bboxes": [[x1, y1, x1 + bw, y1 + bh]], "labels": [phrase]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.ALIGNED_MANIFEST),
                         help="Manifest from step 02 (default: aligned_manifest.csv).")
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

    n_ok = n_no_phrase = n_errors = n_skipped = 0
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
                orig_path = pu.resolve_image_path(row["original_path"])
                prompt_text = row.get("prompt", "")

                if args.dry_run:
                    with Image.open(orig_path) as img:
                        w, h = img.size
                    logger.debug(f"{pair_id}: [DRY RUN] stubbing Florence-2 call")
                    parsed = stub_ground_phrase(prompt_text, w, h)
                else:
                    with Image.open(orig_path) as img:
                        img = img.convert("RGB")
                        parsed = florence.ground_phrase(img, prompt_text)

                bboxes = parsed.get("bboxes", [])
                labels = parsed.get("labels", [])

                out_row = dict(row)
                if bboxes and labels:
                    out_row["extracted_phrase"] = labels[0]
                    out_row["rough_box_x1y1x2y2"] = ",".join(f"{c:.1f}" for c in bboxes[0])
                    out_row["phrase_status"] = "ok"
                    n_ok += 1
                    logger.debug(f"{pair_id}: phrase='{labels[0]}' box={bboxes[0]}")
                else:
                    out_row["extracted_phrase"] = ""
                    out_row["rough_box_x1y1x2y2"] = ""
                    out_row["phrase_status"] = "no_phrase_found"
                    n_no_phrase += 1
                    logger.warning(f"{pair_id}: no groundable phrase found in prompt")

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
    pu.summarize_run(logger, "03_extract_target_phrase", n_ok + n_no_phrase, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} no_phrase={n_no_phrase} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.PHRASE_MANIFEST}")


if __name__ == "__main__":
    main()
