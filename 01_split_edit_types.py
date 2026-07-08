"""
01_split_edit_types.py

Classifies each metadata row as "local" (a spatially confined edit -- the
kind Florence-2/Grounding DINO/SAM2 can produce a meaningful box+mask for)
or "global" (whole-image style/tone/lighting changes with no single
localized region -- these are logged but NOT carried into the grounded
masking pipeline; TruFor's localization head needs local manipulation
masks, so a "global" edit has no ground truth to generate here).

Classification is pure keyword matching against `prompt` + `edit_type`
(see config.LOCAL_EDIT_KEYWORDS / GLOBAL_EDIT_KEYWORDS) -- no model calls,
so --dry_run and a real run behave identically except for row count.

Usage:
  python 01_split_edit_types.py --dry_run
  python 01_split_edit_types.py --limit 10
"""

from __future__ import annotations

import argparse
import time

import config
import pipeline_utils as pu


def classify(prompt: str, edit_type: str) -> tuple[str, int, int]:
    text = f"{prompt or ''} {edit_type or ''}".lower()
    local_hits = sum(1 for kw in config.LOCAL_EDIT_KEYWORDS if kw in text)
    global_hits = sum(1 for kw in config.GLOBAL_EDIT_KEYWORDS if kw in text)
    # Tie/no-match defaults to "local": a false-positive local classification
    # just means the grounding stage logs a no-match later (cheap, logged,
    # non-fatal); a false-positive "global" silently drops a pair we could
    # have masked, which is worse for a small test batch.
    label = "global" if global_hits > local_hits else "local"
    return label, local_hits, global_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.INPUT_METADATA_CSV),
                         help="Path to source metadata CSV.")
    pu.add_common_args(parser)
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("01_split_edit_types")
    if args.dry_run:
        logger.setLevel("DEBUG")

    limit = pu.resolve_limit(args.limit, args.dry_run)
    logger.info(f"dry_run={args.dry_run} limit={limit} input={args.input}")

    t0 = time.monotonic()
    rows = pu.read_csv_rows(args.input)
    logger.info(f"Loaded {len(rows)} rows from {args.input}")
    rows = pu.apply_limit(rows, limit)
    logger.info(f"Processing {len(rows)} rows (after --limit)")

    required_cols = {"pair_id", "original_path", "edited_path", "edit_type", "prompt"}
    if rows:
        missing = required_cols - set(rows[0].keys())
        if missing:
            logger.error(f"Metadata CSV is missing required columns: {missing}. Aborting.")
            raise SystemExit(1)

    fieldnames = list(rows[0].keys()) + ["local_global_label", "local_kw_hits", "global_kw_hits"] \
        if rows else ["pair_id", "original_path", "edited_path", "edit_type", "prompt",
                       "local_global_label", "local_kw_hits", "global_kw_hits"]

    local_writer = pu.IncrementalCSVWriter(config.LOCAL_EDITS_MANIFEST, fieldnames, resume=args.resume)
    global_writer = pu.IncrementalCSVWriter(config.GLOBAL_EDITS_MANIFEST, fieldnames, resume=args.resume)

    n_local = n_global = n_skipped = 0
    for row in rows:
        pair_id = row.get("pair_id", "")
        if local_writer.already_done(pair_id) or global_writer.already_done(pair_id):
            n_skipped += 1
            continue

        label, lhits, ghits = classify(row.get("prompt", ""), row.get("edit_type", ""))
        out_row = dict(row)
        out_row["local_global_label"] = label
        out_row["local_kw_hits"] = lhits
        out_row["global_kw_hits"] = ghits

        logger.debug(f"{pair_id}: '{row.get('prompt','')[:60]}...' -> {label} "
                     f"(local_hits={lhits}, global_hits={ghits})")

        if label == "local":
            local_writer.write_row(out_row)
            n_local += 1
        else:
            global_writer.write_row(out_row)
            n_global += 1

    local_writer.close()
    global_writer.close()

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "01_split_edit_types", n_local + n_global, n_skipped, 0, elapsed)
    logger.info(f"local={n_local} global={n_global} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.LOCAL_EDITS_MANIFEST}")
    logger.info(f"-> {config.GLOBAL_EDITS_MANIFEST}")


if __name__ == "__main__":
    main()
