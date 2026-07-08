"""
Central configuration for the Pico-Banana-400K ground-truth mask pipeline.

Everything that a script might need to know about paths, model checkpoints,
or thresholds lives here so that behavior can be tuned in one place without
hunting through 8 numbered scripts.

MODEL SOURCING NOTE (read before running setup_environment.sh):
  - Grounding DINO: we use the OPEN-WEIGHTS "Grounding DINO 1.0" model
    (SwinT-OGC checkpoint), Apache-2.0, no API token. Grounding DINO 1.5
    (Pro/Edge) is API-gated through DeepDataSpace and requires a paid API
    token -- explicitly avoided per budget constraints.
  - Per explicit instruction, Grounding DINO and SAM 2.1 are loaded from the
    ORIGINAL github.com/IDEA-Research/Grounded-SAM-2 repo (cloned into
    third_party/Grounded-SAM-2 by setup_environment.sh), not the HF
    `transformers` ports. This means GroundingDINO's custom CUDA op gets
    compiled at install time -- setup_environment.sh preflight-checks
    nvcc/CUDA_HOME before attempting it and degrades to GroundingDINO's
    built-in pure-PyTorch attention fallback (slower, but functional) if
    compilation fails, rather than hard-aborting setup.
  - Florence-2 is unaffected by this -- it's not part of Grounded-SAM-2 and
    still loads via `transformers` as microsoft/Florence-2-base.

DATA SOURCE NOTE (2026-07-08): the actual data available for this project is
a Google Drive export with TWO UNRELATED parts confirmed to share no join
key: (a) pos.zip/neg.zip + manifest CSVs -- ~8.9K real local original/edited
image pairs (pos/{id}.jpg = ORIGINAL, neg/{id}.jpg = EDITED) but NO prompt
text; (b) sft.jsonl -- 257K records WITH prompt/edit_type text, but
referencing different images we don't have locally. We're building around
(a). This means the pipeline no longer has a prompt sentence to ground a
phrase from, so:
  - 00_build_metadata_from_pico_zips.py is a new one-off adapter that builds
    data/metadata.csv directly from pos.zip/neg.zip (see that script).
  - 01_split_edit_types.py needs no change: blank edit_type/prompt already
    default to "local" per its existing tie-breaking rule.
  - 03_extract_target_phrase.py now runs Florence-2 UNPROMPTED
    (<DENSE_REGION_CAPTION>, see FLORENCE2_REGION_TASK below) on the edited
    image, then picks whichever candidate region has the highest IoU against
    the SSIM diff mask's bounding box as the "extracted phrase" -- so 05
    (diff masks) must run BEFORE 03 now. run_pipeline.sh reflects this
    reordering; the scripts keep their original numeric filenames so nothing
    else has to be renamed.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGES_ROOT = DATA_DIR / "images"  # used to resolve relative paths in the CSV

OUTPUT_DIR = BASE_DIR / "outputs"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
MASKS_DIR = OUTPUT_DIR / "masks"
GROUNDED_MASKS_DIR = MASKS_DIR / "grounded"
DIFF_MASKS_DIR = MASKS_DIR / "diff"
ACCEPTED_MASKS_DIR = MASKS_DIR / "accepted"
REVIEW_MASKS_DIR = MASKS_DIR / "review"
LOG_DIR = OUTPUT_DIR / "logs"
QC_DIR = OUTPUT_DIR / "qc"

MODELS_DIR = BASE_DIR / "models"  # Florence-2's HF cache lives here (set as HF_HOME);
                                   # also where we drop the GroundingDINO/SAM2 checkpoint
                                   # files we download ourselves (see below).
ALIGNED_IMAGES_DIR = OUTPUT_DIR / "aligned_images"  # only resized edited images land here

THIRD_PARTY_DIR = BASE_DIR / "third_party"
GROUNDED_SAM2_DIR = THIRD_PARTY_DIR / "Grounded-SAM-2"  # cloned by setup_environment.sh

ALL_DIRS = [
    DATA_DIR, IMAGES_ROOT, OUTPUT_DIR, MANIFEST_DIR, MASKS_DIR,
    GROUNDED_MASKS_DIR, DIFF_MASKS_DIR, ACCEPTED_MASKS_DIR, REVIEW_MASKS_DIR,
    LOG_DIR, QC_DIR, MODELS_DIR, ALIGNED_IMAGES_DIR, THIRD_PARTY_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Input / manifest files (chained pipeline: each script reads the previous
# script's output manifest and writes its own)
# ---------------------------------------------------------------------------
INPUT_METADATA_CSV = DATA_DIR / "metadata.csv"  # pair_id, original_path, edited_path, edit_type, prompt

LOCAL_EDITS_MANIFEST = MANIFEST_DIR / "local_edits_manifest.csv"      # 01
GLOBAL_EDITS_MANIFEST = MANIFEST_DIR / "global_edits_manifest.csv"    # 01

ALIGNED_MANIFEST = MANIFEST_DIR / "aligned_manifest.csv"              # 02
ALIGNMENT_REVIEW_CSV = MANIFEST_DIR / "alignment_review.csv"          # 02

PHRASE_MANIFEST = MANIFEST_DIR / "phrase_manifest.csv"                # 03

GROUNDED_MASK_MANIFEST = MANIFEST_DIR / "grounded_mask_manifest.csv"  # 04
NO_MATCH_LOG_CSV = LOG_DIR / "no_match_log.csv"                       # 04

DIFF_MASK_MANIFEST = MANIFEST_DIR / "diff_mask_manifest.csv"          # 05

CROSS_VALIDATED_MANIFEST = MANIFEST_DIR / "cross_validated_manifest.csv"  # 06
REJECTED_MANIFEST = MANIFEST_DIR / "rejected_manifest.csv"                # 06

FINAL_TRAINING_MANIFEST = MANIFEST_DIR / "training_manifest.csv"      # 07

QC_GRID_PNG = QC_DIR / "visual_qc_grid.png"                           # 08

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"          # base, not large; via `transformers`
FLORENCE2_REGION_TASK = "<DENSE_REGION_CAPTION>"          # unprompted region proposals (no prompt text available)

# Grounding DINO (original IDEA-Research/GroundingDINO code, vendored inside
# Grounded-SAM-2 at third_party/Grounded-SAM-2/grounding_dino/). Config is a
# real filesystem path; checkpoint is downloaded directly by
# setup_environment.sh (NOT via their bulk download_ckpts.sh, which pulls
# extra checkpoints we don't need).
GROUNDING_DINO_CONFIG_PATH = (
    GROUNDED_SAM2_DIR / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = MODELS_DIR / "groundingdino_swint_ogc.pth"
GROUNDING_DINO_CHECKPOINT_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

# SAM 2.1 hiera-tiny (original facebookresearch/sam2 code, vendored inside
# Grounded-SAM-2 at third_party/Grounded-SAM-2/sam2/). IMPORTANT: SAM2_CONFIG_NAME
# is NOT a filesystem path -- build_sam2() resolves it through Hydra relative
# to the installed `sam2` package's own config dir, so it must stay in this
# "configs/..." form even though everything else in this file is a real path.
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT_PATH = MODELS_DIR / "sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

# ---------------------------------------------------------------------------
# Detection / masking thresholds
# ---------------------------------------------------------------------------
GROUNDING_DINO_BOX_THRESHOLD = 0.30    # min confidence to keep a box
GROUNDING_DINO_TEXT_THRESHOLD = 0.25   # min text-token confidence

IOU_ACCEPT_THRESHOLD = 0.30            # grounded-vs-diff mask agreement to accept
MIN_COMPONENT_AREA_PX = 100            # connected components smaller than this are dropped
ASPECT_RATIO_MISMATCH_PCT = 5.0        # flag pairs whose aspect ratio differs by more than this
REGION_MATCH_MIN_IOU = 0.05            # min IoU a Florence-2 region proposal must have against the
                                        # diff-mask bbox to be picked as the "extracted phrase" target

# ---------------------------------------------------------------------------
# Local vs. global edit keyword lists (used by 01_split_edit_types.py)
# Edit lists here, not code, if you need to retune classification.
# ---------------------------------------------------------------------------
LOCAL_EDIT_KEYWORDS = [
    "remove", "delete", "erase", "add", "insert", "place",
    "replace", "swap", "change the color", "recolor",
    "hat", "glasses", "shirt", "logo", "text", "sign", "watermark",
    "object", "person", "face", "eyes", "hair", "background object",
    "crop out", "cover", "cover up", "blur the", "hide the",
]

GLOBAL_EDIT_KEYWORDS = [
    "style", "filter", "color grade", "color grading", "tone",
    "lighting", "brightness", "contrast", "saturation", "vibrance",
    "black and white", "sepia", "vintage", "cinematic", "hdr",
    "sharpen", "blur the whole", "whole image", "entire image",
    "overall", "atmosphere", "mood", "weather", "season",
    "day to night", "night to day", "resolution", "upscale",
    "denoise", "art style", "painting", "sketch", "cartoon", "anime",
]

# ---------------------------------------------------------------------------
# Data source: pos/neg zip adapter (00_build_metadata_from_pico_zips.py)
# ---------------------------------------------------------------------------
GDRIVE_RAW_DIR = DATA_DIR / "gdrive_raw"
POS_ZIP_PATH = GDRIVE_RAW_DIR / "pos.zip"   # pos/{id}.jpg = ORIGINAL (per user confirmation)
NEG_ZIP_PATH = GDRIVE_RAW_DIR / "neg.zip"   # neg/{id}.jpg = EDITED

# ---------------------------------------------------------------------------
# Budget / run-control defaults
# ---------------------------------------------------------------------------
DEFAULT_LIMIT = 10
DEFAULT_TIME_BUDGET_MINUTES = 30
COST_PER_HOUR_USD = 0.20  # RunPod Community Cloud ballpark; override with --cost_per_hour
COST_ESTIMATE_AFTER_N = 5  # print elapsed/item + projected cost after this many images

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def configure_hf_cache() -> None:
    """Point HF/transformers caches at MODELS_DIR so setup_environment.sh's
    pre-downloaded weights are actually reused (no surprise re-downloads)."""
    os.environ.setdefault("HF_HOME", str(MODELS_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


if __name__ == "__main__":
    ensure_dirs()
    print(f"Base dir: {BASE_DIR}")
    print("Directories ensured:")
    for d in ALL_DIRS:
        print(f"  {d}")
