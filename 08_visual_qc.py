"""
08_visual_qc.py

Builds a side-by-side grid PNG for a random sample of accepted pairs:
  original | edited | edited+grounded-mask overlay (red) | edited+diff-mask overlay (blue)

This is the fastest way to eyeball whether the whole pipeline is actually
producing sane masks before spending more budget on a bigger batch.

Usage:
  python 08_visual_qc.py --dry_run
  python 08_visual_qc.py --sample_size 10
"""

from __future__ import annotations

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
import pipeline_utils as pu


def overlay_mask(base_img: Image.Image, mask_path, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = np.array(base_img.convert("RGB")).astype(np.float32)
    mask = np.array(Image.open(mask_path).convert("L").resize(base_img.size)) > 127

    overlay = base.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_arr
    return Image.fromarray(overlay.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(config.CROSS_VALIDATED_MANIFEST),
                         help="Manifest with original/edited/grounded/diff paths (default: cross_validated_manifest.csv).")
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--output", default=str(config.QC_GRID_PNG))
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--dry_run", action="store_true",
                         help="Cap sample size at 3 with verbose logging.")
    args = parser.parse_args()

    config.ensure_dirs()
    logger = pu.setup_logging("08_visual_qc")
    if args.dry_run:
        logger.setLevel("DEBUG")

    sample_size = min(args.sample_size, 3) if args.dry_run else args.sample_size
    logger.info(f"dry_run={args.dry_run} sample_size={sample_size} input={args.input}")

    rows = pu.read_csv_rows(args.input)
    if not rows:
        logger.error("No accepted rows to visualize. Nothing written.")
        raise SystemExit(1)

    random.seed(args.seed)
    sample = random.sample(rows, min(sample_size, len(rows)))
    logger.info(f"Sampled {len(sample)} of {len(rows)} accepted pairs")

    fig, axes = plt.subplots(len(sample), 4, figsize=(16, 4 * len(sample)))
    if len(sample) == 1:
        axes = axes.reshape(1, 4)

    col_titles = ["original", "edited", "grounded mask overlay", "diff mask overlay"]

    for i, row in enumerate(sample):
        pair_id = row.get("pair_id", "")
        try:
            orig_path = pu.resolve_image_path(row["original_path"])
            edited_path = pu.resolve_image_path(row.get("aligned_edited_path") or row["edited_path"])

            with Image.open(orig_path) as orig_img, Image.open(edited_path) as edited_img:
                orig_img = orig_img.convert("RGB")
                edited_img = edited_img.convert("RGB")

                grounded_overlay = overlay_mask(edited_img, row["grounded_mask_path"], color=(255, 0, 0))
                diff_overlay = overlay_mask(edited_img, row["diff_mask_path"], color=(0, 0, 255))

            for ax, img in zip(axes[i], [orig_img, edited_img, grounded_overlay, diff_overlay]):
                ax.imshow(img)
                ax.axis("off")

            axes[i][0].set_ylabel(pair_id, fontsize=8)
            logger.debug(f"{pair_id}: added to QC grid")

        except Exception as e:
            logger.error(f"{pair_id}: failed to render QC row -- {e}")
            for ax in axes[i]:
                ax.axis("off")

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    plt.close(fig)

    logger.info(f"-> {args.output}")


if __name__ == "__main__":
    main()
