"""
02_align_pairs.py

Ensures each (original, edited) pair has matching pixel dimensions before
anything downstream tries to compute per-pixel diffs or reuse a box detected
on one image against the other. If the edited image's size differs from the
original's, it is resized (LANCZOS) to match and the resized copy is written
to outputs/aligned_images/. Pairs whose aspect ratio differs by more than
config.ASPECT_RATIO_MISMATCH_PCT are additionally flagged to a review CSV --
a large aspect-ratio mismatch usually means the resize will distort content,
which matters for anyone manually auditing the resulting masks later.

No models loaded -- this is pure PIL I/O, so --dry_run just caps rows at 3
and turns on verbose per-pair logging (still hits real disk paths, which is
exactly the kind of bug --dry_run exists to catch for free).

Usage:
  python 02_align_pairs.py --dry_run
  python 02_align_pairs.py --limit 10 --resume
"""

from __future__ import annotations

import argparse
import time

from PIL import Image

import config
import pipeline_utils as pu


def aspect_ratio(w: int, h: int) -> float:
    return w / h if h else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.LOCAL_EDITS_MANIFEST),
                         help="Manifest to align (default: local_edits_manifest.csv from step 01).")
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("02_align_pairs")
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

    out_fields = list(rows[0].keys()) + ["aligned_edited_path", "was_resized",
                                          "orig_w", "orig_h", "edited_w", "edited_h",
                                          "aspect_mismatch_pct"] if rows else []
    review_fields = out_fields + ["reason"]

    writer = pu.IncrementalCSVWriter(config.ALIGNED_MANIFEST, out_fields, resume=args.resume)
    review_writer = pu.IncrementalCSVWriter(config.ALIGNMENT_REVIEW_CSV, review_fields, resume=args.resume)

    n_ok = n_resized = n_flagged = n_errors = n_skipped = 0
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
            edited_path = pu.resolve_image_path(row["edited_path"])

            with Image.open(orig_path) as orig_img, Image.open(edited_path) as edited_img:
                ow, oh = orig_img.size
                ew, eh = edited_img.size

                ar_o = aspect_ratio(ow, oh)
                ar_e = aspect_ratio(ew, eh)
                mismatch_pct = abs(ar_e - ar_o) / ar_o * 100 if ar_o else 0.0

                was_resized = False
                aligned_edited_path = str(edited_path)
                if (ow, oh) != (ew, eh):
                    resized = edited_img.convert("RGB").resize((ow, oh), Image.LANCZOS)
                    out_path = config.ALIGNED_IMAGES_DIR / f"{pair_id}_edited_aligned.png"
                    resized.save(out_path)
                    aligned_edited_path = str(out_path)
                    was_resized = True

            out_row = dict(row)
            out_row.update({
                "aligned_edited_path": aligned_edited_path,
                "was_resized": was_resized,
                "orig_w": ow, "orig_h": oh,
                "edited_w": ew, "edited_h": eh,
                "aspect_mismatch_pct": round(mismatch_pct, 3),
            })
            writer.write_row(out_row)
            n_ok += 1
            if was_resized:
                n_resized += 1

            if mismatch_pct > config.ASPECT_RATIO_MISMATCH_PCT:
                review_row = dict(out_row)
                review_row["reason"] = (
                    f"aspect ratio mismatch {mismatch_pct:.2f}% "
                    f"exceeds threshold {config.ASPECT_RATIO_MISMATCH_PCT}%"
                )
                review_writer.write_row(review_row)
                n_flagged += 1
                logger.warning(f"{pair_id}: FLAGGED aspect mismatch {mismatch_pct:.2f}%")

            logger.debug(f"{pair_id}: orig={ow}x{oh} edited={ew}x{eh} "
                         f"resized={was_resized} mismatch={mismatch_pct:.2f}%")

        except FileNotFoundError as e:
            n_errors += 1
            logger.error(f"{pair_id}: file not found -- {e}")
        except Exception as e:
            n_errors += 1
            logger.error(f"{pair_id}: unexpected error -- {e}")

        cost_est.tick(total_planned=len(rows))

    writer.close()
    review_writer.close()

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "02_align_pairs", n_ok, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} resized={n_resized} flagged_review={n_flagged} "
                f"errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.ALIGNED_MANIFEST}")
    logger.info(f"-> {config.ALIGNMENT_REVIEW_CSV}")


if __name__ == "__main__":
    main()
