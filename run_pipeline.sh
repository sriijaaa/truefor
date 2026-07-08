#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_pipeline.sh
#
# Orchestrates 01 -> 08 in sequence. A single --limit / --time_budget_minutes
# is passed through to every step. The remaining time budget is recomputed
# before each step; once it's exhausted, remaining steps are SKIPPED
# entirely (no model load attempted) rather than launched and immediately
# cut off, and whatever manifests exist so far are used for the final
# summary. `set -euo pipefail` means any step that hard-fails (bad config,
# missing columns, etc.) stops the whole run immediately -- fail fast rather
# than silently limping forward on broken output.
#
# Usage:
#   ./run_pipeline.sh --dry_run
#   ./run_pipeline.sh --limit 10 --time_budget_minutes 30
#   ./run_pipeline.sh --limit 10 --time_budget_minutes 30 --resume
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configurable cost assumption for the final summary (RunPod Community Cloud ballpark).
COST_PER_HOUR_USD="${COST_PER_HOUR_USD:-0.20}"

LIMIT=10
TIME_BUDGET_MINUTES=30
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --time_budget_minutes) TIME_BUDGET_MINUTES="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        --resume) RESUME=1; shift ;;
        --cost_per_hour) COST_PER_HOUR_USD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

COMMON_FLAGS=(--limit "$LIMIT")
[[ $RESUME -eq 1 ]] && COMMON_FLAGS+=(--resume)
[[ $DRY_RUN -eq 1 ]] && COMMON_FLAGS+=(--dry_run)

PIPE_START=$(date +%s)

echo "============================================================"
echo " run_pipeline.sh: limit=$LIMIT time_budget_minutes=$TIME_BUDGET_MINUTES dry_run=$DRY_RUN resume=$RESUME"
echo "============================================================"

remaining_minutes() {
    local now elapsed_s elapsed_min
    now=$(date +%s)
    elapsed_s=$(( now - PIPE_START ))
    elapsed_min=$(( elapsed_s / 60 ))
    echo $(( TIME_BUDGET_MINUTES - elapsed_min ))
}

# HALTED=1 once the time budget runs out; every later step becomes a no-op
# (still printed) instead of spending time attempting anything further.
HALTED=0

# $1 = step label, remaining args = the command to run. Aborts the whole
# script (exit 1) on a genuine failure -- fail fast rather than limping
# forward on broken output. This does NOT rely on `set -e` propagating out
# of an `if`/`||` context (it doesn't); failure is detected and handled
# explicitly below.
step() {
    local label="$1"; shift
    if [[ $HALTED -eq 1 ]]; then
        echo ">>> [$label] SKIPPED (time budget already exhausted)"
        return
    fi
    local rem
    rem=$(remaining_minutes)
    if [[ $rem -le 0 ]]; then
        echo ">>> [$label] SKIPPED -- time budget of ${TIME_BUDGET_MINUTES} min exhausted. Progress from prior steps is saved."
        HALTED=1
        return
    fi
    echo ">>> [$label] (remaining budget: ${rem} min)"
    if ! "$@"; then
        echo "ERROR: [$label] failed. Aborting pipeline (fail-fast). Progress from completed steps is saved." >&2
        exit 1
    fi
}

# Same as step(), but a failure only warns instead of aborting the whole run.
step_soft() {
    local label="$1"; shift
    if [[ $HALTED -eq 1 ]]; then
        echo ">>> [$label] SKIPPED (time budget already exhausted)"
        return
    fi
    local rem
    rem=$(remaining_minutes)
    if [[ $rem -le 0 ]]; then
        echo ">>> [$label] SKIPPED -- time budget of ${TIME_BUDGET_MINUTES} min exhausted."
        HALTED=1
        return
    fi
    echo ">>> [$label] (remaining budget: ${rem} min)"
    if ! "$@"; then
        echo "WARNING: [$label] failed (non-fatal, continuing to summary)." >&2
    fi
}

step "01_split_edit_types" \
    python3 01_split_edit_types.py "${COMMON_FLAGS[@]}"

step "02_align_pairs" \
    python3 02_align_pairs.py "${COMMON_FLAGS[@]}" --time_budget_minutes "$(remaining_minutes)"

step "03_extract_target_phrase" \
    python3 03_extract_target_phrase.py "${COMMON_FLAGS[@]}" --time_budget_minutes "$(remaining_minutes)"

step "04_generate_grounded_masks" \
    python3 04_generate_grounded_masks.py "${COMMON_FLAGS[@]}" --time_budget_minutes "$(remaining_minutes)"

step "05_generate_diff_masks" \
    python3 05_generate_diff_masks.py "${COMMON_FLAGS[@]}"

step "06_cross_validate_and_clean" \
    python3 06_cross_validate_and_clean.py "${COMMON_FLAGS[@]}"

step "07_build_final_manifest" \
    python3 07_build_final_manifest.py "${COMMON_FLAGS[@]}"

QC_FLAGS=(--sample_size "$LIMIT")
[[ $DRY_RUN -eq 1 ]] && QC_FLAGS+=(--dry_run)
step_soft "08_visual_qc" \
    python3 08_visual_qc.py "${QC_FLAGS[@]}"  # QC is nice-to-have; don't fail the run over it

PIPE_END=$(date +%s)
ELAPSED_S=$(( PIPE_END - PIPE_START ))
ELAPSED_MIN=$(python3 -c "print(f'{${ELAPSED_S}/60:.2f}')")
EST_COST=$(python3 -c "print(f'{${ELAPSED_S}/3600*${COST_PER_HOUR_USD}:.4f}')")

FINAL_MANIFEST="outputs/manifests/training_manifest.csv"
N_IMAGES=0
if [[ -f "$FINAL_MANIFEST" ]]; then
    N_IMAGES=$(( $(wc -l < "$FINAL_MANIFEST") - 1 ))
fi

echo ""
echo "============================================================"
echo " PIPELINE SUMMARY"
echo "============================================================"
echo " Images in final training manifest : $N_IMAGES"
echo " Total elapsed wall-clock time      : ${ELAPSED_S}s (${ELAPSED_MIN} min)"
echo " Assumed cost rate                  : \$${COST_PER_HOUR_USD}/hr"
echo " Estimated cost this run            : \$${EST_COST}"
echo " Final manifest                     : ${FINAL_MANIFEST}"
echo " QC grid                            : outputs/qc/visual_qc_grid.png"
echo "============================================================"
