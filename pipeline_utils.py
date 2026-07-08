"""
Shared helpers used by every numbered pipeline script:
  - logging setup (console + per-script log file)
  - --resume support (skip already-processed pair_ids, load from existing output)
  - --time_budget_minutes wall-clock cutoff
  - incremental (per-row) CSV writing so no completed work is ever batched-and-lost
  - cost/time-per-image estimate printed after the first N images
  - small path/image helpers shared across scripts

Nothing here touches GPU/model code -- that lives in models.py so this module
stays importable with zero heavy dependencies (pandas + Pillow only).
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(script_name: str) -> logging.Logger:
    config.ensure_dirs()
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = config.LOG_DIR / f"{script_name}.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------
class TimeBudget:
    """Wall-clock cutoff. Call .expired() before starting each item's work."""

    def __init__(self, minutes: Optional[float]):
        self.minutes = minutes
        self.start = time.monotonic()

    def expired(self) -> bool:
        if self.minutes is None:
            return False
        return (time.monotonic() - self.start) >= self.minutes * 60

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start


# ---------------------------------------------------------------------------
# Cost / time-per-image estimator
# ---------------------------------------------------------------------------
class CostEstimator:
    """Prints an elapsed/item + projected cost line once `after_n` images
    have been processed, so the user can decide to keep going or bail
    before committing GPU-hours to the rest of the batch."""

    def __init__(self, logger: logging.Logger, after_n: int = None, cost_per_hour: float = None):
        self.logger = logger
        self.after_n = after_n or config.COST_ESTIMATE_AFTER_N
        self.cost_per_hour = cost_per_hour if cost_per_hour is not None else config.COST_PER_HOUR_USD
        self.start = time.monotonic()
        self.count = 0
        self._printed = False

    def tick(self, total_planned: Optional[int] = None) -> None:
        self.count += 1
        if self.count == self.after_n and not self._printed:
            elapsed = time.monotonic() - self.start
            per_item = elapsed / self.count
            self._printed = True
            msg = (
                f"[COST ESTIMATE] {self.count} images in {elapsed:.1f}s "
                f"({per_item:.2f}s/image). "
                f"Projected: {per_item / 3600 * self.cost_per_hour:.4f} USD/image "
                f"at ${self.cost_per_hour:.2f}/hr."
            )
            if total_planned:
                remaining = max(total_planned - self.count, 0)
                proj_remaining_s = remaining * per_item
                proj_remaining_cost = proj_remaining_s / 3600 * self.cost_per_hour
                msg += (
                    f" Remaining {remaining} images -> ~{proj_remaining_s / 60:.1f} min, "
                    f"~${proj_remaining_cost:.4f}."
                )
            self.logger.info(msg)


# ---------------------------------------------------------------------------
# Incremental CSV writer with --resume support
# ---------------------------------------------------------------------------
class IncrementalCSVWriter:
    """Writes one row at a time, flushing immediately, and can resume a
    previous run by reporting which pair_ids already have a completed row.

    Usage:
        writer = IncrementalCSVWriter(out_path, fieldnames, resume=args.resume)
        for row in rows:
            if writer.already_done(row["pair_id"]):
                continue
            ... do work ...
            writer.write_row(result_dict)
        writer.close()
    """

    def __init__(self, path: Path, fieldnames: list[str], resume: bool = False):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self._done_ids: set[str] = set()

        file_exists = self.path.exists()

        if resume and file_exists:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("pair_id")
                    if pid:
                        self._done_ids.add(pid)
            self._fh = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
            self._writer.writeheader()
            self._fh.flush()

    def already_done(self, pair_id: str) -> bool:
        return pair_id in self._done_ids

    def write_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()
        pid = row.get("pair_id")
        if pid:
            self._done_ids.add(pid)

    def close(self) -> None:
        self._fh.close()

    @property
    def n_already_done(self) -> int:
        return len(self._done_ids)


# ---------------------------------------------------------------------------
# Misc shared helpers
# ---------------------------------------------------------------------------
def apply_limit(rows: list, limit: Optional[int]) -> list:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def resolve_limit(limit: Optional[int], dry_run: bool) -> int:
    """--dry_run always caps at 3 items regardless of --limit, so a mistaken
    `--dry_run --limit 500` still can't burn real time/GPU budget."""
    if dry_run:
        return min(limit, 3) if limit else 3
    return limit if limit is not None else config.DEFAULT_LIMIT


def add_common_args(parser) -> None:
    """Shared CLI flags every numbered script exposes."""
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT,
                         help=f"Max rows to process (default {config.DEFAULT_LIMIT}). "
                              f"Never omit this on a real run.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Run full logic on <=3 samples with verbose logging; "
                              "stubs heavy model calls where applicable. No GPU time spent.")
    parser.add_argument("--resume", action="store_true",
                         help="Skip pair_ids already present in this script's output manifest.")
    parser.add_argument("--time_budget_minutes", type=float, default=None,
                         help="Stop gracefully (progress saved) after this many wall-clock minutes.")


def resolve_image_path(raw_path: str) -> Path:
    """CSV paths may be absolute or relative to data/images/."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = config.IMAGES_ROOT / raw_path
    if candidate.exists():
        return candidate
    candidate2 = config.DATA_DIR / raw_path
    if candidate2.exists():
        return candidate2
    # Fall through: return the raw path so callers get a clear FileNotFoundError
    # with the exact path they tried, rather than a silent mismatch.
    return p


def read_csv_rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Expected manifest not found: {path}\n"
            f"Did the previous pipeline step run successfully?"
        )
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_run(logger: logging.Logger, script_name: str, n_processed: int,
                   n_skipped_resume: int, n_errors: int, elapsed_seconds: float,
                   cost_per_hour: float = None) -> None:
    cost_per_hour = cost_per_hour if cost_per_hour is not None else config.COST_PER_HOUR_USD
    est_cost = elapsed_seconds / 3600 * cost_per_hour
    logger.info(
        f"[{script_name}] DONE. processed={n_processed} "
        f"skipped(resume)={n_skipped_resume} errors={n_errors} "
        f"elapsed={elapsed_seconds:.1f}s est_cost=${est_cost:.4f}"
    )
