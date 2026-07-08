"""
00_build_metadata_from_pico_zips.py

ONE-OFF ADAPTER for this specific data source -- NOT part of the generic
8-step pipeline. Builds data/metadata.csv directly from the pos.zip/neg.zip
pair downloaded from the shared Google Drive folder.

Confirmed 2026-07-08: pos/{id}.jpg = ORIGINAL image, neg/{id}.jpg = EDITED
image. There is no prompt/edit_type text for these pairs -- sft.jsonl (which
has prompt text) was verified to share no join key with these zips (its only
keys are edit_type/open_image_input_url/output_image/summarized_text/text,
none of which reference this zip's numeric ids), so this script leaves
edit_type/prompt blank. 01_split_edit_types.py already defaults blank rows
to "local"; 03_extract_target_phrase.py runs Florence-2 unprompted for rows
with no prompt text instead of grounding a sentence.

A pair is only used if its numeric id appears in BOTH pos.zip and neg.zip
(some ids are apparently missing from one side or the other -- 8,906 files
in pos.zip vs 8,821 in neg.zip). Only the ids actually needed (respecting
--limit) are decompressed from the zips -- listing uses Python's `zipfile`
(reads just the central directory, confirmed reliable), but actual
extraction shells out to the system `unzip` binary instead of
`zipfile.open()`. These particular archives trip a known Python `zipfile`
bug ("Bad magic number for file header") where it trusts stale byte offsets
in the central directory for some members near the start of the file;
`unzip` (Info-ZIP) auto-corrects for this and was confirmed working against
the same archives. Neither approach requires unzipping the full ~1.8GB
archive -- both do targeted single-member extraction.

Usage:
  python 00_build_metadata_from_pico_zips.py --dry_run
  python 00_build_metadata_from_pico_zips.py --limit 10
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import config
import pipeline_utils as pu

OUTPUT_COLUMNS = ["pair_id", "original_path", "edited_path", "edit_type", "prompt"]


class MemberExtractionError(Exception):
    pass


def extract_member(zip_path: str, member_name: str, dest_path: Path) -> None:
    """Extract a single member via the system `unzip` binary (see module
    docstring for why -- Python's zipfile module fails on these archives)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["unzip", "-p", str(zip_path), member_name],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        raise MemberExtractionError(
            f"unzip -p {zip_path} {member_name} failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')[:300]}"
        )
    dest_path.write_bytes(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos_zip", default=str(config.POS_ZIP_PATH),
                         help="Zip containing ORIGINAL images (default: data/gdrive_raw/pos.zip).")
    parser.add_argument("--neg_zip", default=str(config.NEG_ZIP_PATH),
                         help="Zip containing EDITED images (default: data/gdrive_raw/neg.zip).")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT,
                         help=f"Max pairs to extract (default {config.DEFAULT_LIMIT}). "
                              f"Never omit this on a real run -- there are ~8.8K available pairs.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Cap at 3 pairs with verbose logging. Still does real (tiny) extraction -- "
                              "this script has no GPU model to stub.")
    parser.add_argument("--resume", action="store_true",
                         help="Skip pair_ids already present in data/metadata.csv.")
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("00_build_metadata_from_pico_zips")
    if args.dry_run:
        logger.setLevel("DEBUG")

    if shutil.which("unzip") is None:
        logger.error("`unzip` binary not found on PATH -- required for extraction (see module "
                      "docstring for why we don't use Python's zipfile module here). "
                      "Install it, e.g. `apt-get install -y unzip`.")
        raise SystemExit(1)

    limit = pu.resolve_limit(args.limit, args.dry_run)
    logger.info(f"dry_run={args.dry_run} limit={limit} pos_zip={args.pos_zip} neg_zip={args.neg_zip}")

    for p in (args.pos_zip, args.neg_zip):
        if not Path(p).exists():
            logger.error(f"Zip not found: {p}. Did you download it into data/gdrive_raw/ first?")
            raise SystemExit(1)

    t0 = time.monotonic()

    with zipfile.ZipFile(args.pos_zip) as zf_pos, zipfile.ZipFile(args.neg_zip) as zf_neg:
        pos_names = {n for n in zf_pos.namelist() if not n.endswith("/")}
        neg_names = {n for n in zf_neg.namelist() if not n.endswith("/")}
        common = sorted(pos_names & neg_names)

        logger.info(f"pos.zip: {len(pos_names)} files, neg.zip: {len(neg_names)} files, "
                    f"{len(common)} ids present in BOTH (usable pairs)")

        if not common:
            logger.error("No overlapping filenames between pos.zip and neg.zip -- "
                         "check that these are really the paired archives.")
            raise SystemExit(1)

        selected = pu.apply_limit(common, limit)
        logger.info(f"Extracting {len(selected)} pairs (of {len(common)} available)")

        writer = pu.IncrementalCSVWriter(config.INPUT_METADATA_CSV, OUTPUT_COLUMNS, resume=args.resume)

        n_ok = n_errors = n_skipped = 0
        for member_name in selected:
            pair_id = Path(member_name).stem
            if writer.already_done(pair_id):
                n_skipped += 1
                continue

            try:
                ext = Path(member_name).suffix or ".jpg"
                orig_dest = config.IMAGES_ROOT / f"{pair_id}_original{ext}"
                edited_dest = config.IMAGES_ROOT / f"{pair_id}_edited{ext}"

                extract_member(args.pos_zip, member_name, orig_dest)
                extract_member(args.neg_zip, member_name, edited_dest)

                writer.write_row({
                    "pair_id": pair_id,
                    "original_path": str(orig_dest),
                    "edited_path": str(edited_dest),
                    "edit_type": "",
                    "prompt": "",
                })
                n_ok += 1
                logger.debug(f"{pair_id}: extracted original + edited")

            except MemberExtractionError as e:
                n_errors += 1
                logger.error(f"{pair_id}: extraction failed -- {e}")
            except Exception as e:
                n_errors += 1
                logger.error(f"{pair_id}: unexpected error -- {e}")

        writer.close()

    elapsed = time.monotonic() - t0
    pu.summarize_run(logger, "00_build_metadata_from_pico_zips", n_ok, n_skipped, n_errors, elapsed)
    logger.info(f"ok={n_ok} errors={n_errors} skipped(resume)={n_skipped}")
    logger.info(f"-> {config.INPUT_METADATA_CSV}")
    logger.info(f"-> images extracted to {config.IMAGES_ROOT}")


if __name__ == "__main__":
    main()
