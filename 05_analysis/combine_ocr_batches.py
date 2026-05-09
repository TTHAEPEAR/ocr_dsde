"""
Combine OCR split outputs from multiple batches without overwriting originals.

Inputs default to:
    data/ocr_raw/raw_all_forms_split.csv
    data/ocr_raw/friend1.csv
    data/ocr_raw/friend2.csv
    data/ocr_raw/friend3.csv

Outputs:
    data/ocr_raw/raw_all_forms_split_combined.csv
    data/ocr_raw/combined_batch_summary.csv
    data/ocr_raw/combined_missing_ballot_kind.csv
    data/ocr_raw/combined_review_queue.csv

Use --activate to back up the current raw_all_forms_split.csv and replace it
with the combined file for existing clean/analysis/dashboard scripts.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import OCR_RAW_DIR


DEFAULT_BATCHES = {
    "mine": OCR_RAW_DIR / "raw_all_forms_split.csv",
    "friend1": OCR_RAW_DIR / "friend1.csv",
    "friend2": OCR_RAW_DIR / "friend2.csv",
    "friend3": OCR_RAW_DIR / "friend3.csv",
}


def _to_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _num(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def vote_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("candidate_") and c.endswith("_votes")],
        key=lambda c: int(c.split("_")[1]),
    )


def fill_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill quality columns missing from older friend exports."""
    df = df.copy()
    votes_sum = df[vote_columns(df)].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    default_zero = pd.Series(0, index=df.index)
    good = pd.to_numeric(df["good_ballots"] if "good_ballots" in df.columns else default_zero, errors="coerce").fillna(0)
    bad = pd.to_numeric(df["bad_ballots"] if "bad_ballots" in df.columns else default_zero, errors="coerce").fillna(0)
    no_vote = pd.to_numeric(
        df["no_vote_ballots"] if "no_vote_ballots" in df.columns else default_zero,
        errors="coerce",
    ).fillna(0)
    total = pd.to_numeric(df["total_ballots"] if "total_ballots" in df.columns else default_zero, errors="coerce").fillna(0)
    has_total_votes_sum = "total_votes_sum" in df.columns
    total_votes = pd.to_numeric(
        df["total_votes_sum"] if has_total_votes_sum else default_zero,
        errors="coerce",
    ).fillna(0)

    if "votes_sum" not in df.columns:
        df["votes_sum"] = votes_sum
    if "vote_sum_match_good_ballots" not in df.columns:
        df["vote_sum_match_good_ballots"] = (good > 0) & (votes_sum == good)
    if "vote_sum_match_total_votes" not in df.columns:
        df["vote_sum_match_total_votes"] = (
            (total_votes > 0) & (votes_sum == total_votes)
            if has_total_votes_sum
            else pd.Series(pd.NA, index=df.index)
        )
    if "vote_sum_match" not in df.columns:
        df["vote_sum_match"] = df["vote_sum_match_good_ballots"].map(lambda v: _to_bool(v))
    if "summary_votes_match" not in df.columns:
        df["summary_votes_match"] = (
            (good > 0) & (total_votes > 0) & (good == total_votes)
            if has_total_votes_sum
            else pd.Series(pd.NA, index=df.index)
        )
    if "total_votes_sum_match" not in df.columns:
        df["total_votes_sum_match"] = df["vote_sum_match_total_votes"]
    if "ballot_sum_match" not in df.columns:
        df["ballot_sum_match"] = (total > 0) & ((good + bad + no_vote) == total)
    if "needs_review" not in df.columns:
        df["needs_review"] = ~(
            df["vote_sum_match"].map(lambda v: _to_bool(v))
            & df["ballot_sum_match"].map(lambda v: _to_bool(v))
        )
    return df


def review_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not _to_bool(row.get("ocr_success", True), default=True):
        reasons.append("ocr_failed")
    if _to_bool(row.get("needs_review", False), default=False):
        reasons.append("needs_review")
    if not _to_bool(row.get("vote_sum_match", True), default=True):
        reasons.append("vote_sum_mismatch")
    if not _to_bool(row.get("ballot_sum_match", True), default=True):
        reasons.append("ballot_sum_mismatch")
    if pd.notna(row.get("summary_votes_match")) and not _to_bool(row.get("summary_votes_match"), default=True):
        reasons.append("good_vs_total_votes_mismatch")
    return ";".join(dict.fromkeys(reasons))


def combine_batches(batch_paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch, path in batch_paths.items():
        if not path.exists():
            print(f"SKIP missing: {batch} -> {path}")
            continue
        df = pd.read_csv(path)
        df["source_batch"] = batch
        df["source_batch_file"] = str(path)
        frames.append(fill_quality_columns(df))

    if not frames:
        raise FileNotFoundError("No batch CSVs found to combine.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "ballot_record_id" not in combined.columns:
        raise ValueError("Combined data must include ballot_record_id.")

    combined = combined.drop_duplicates(subset=["ballot_record_id"], keep="last")
    combined["combined_review_reason"] = combined.apply(review_reason, axis=1)
    return combined


def write_outputs(combined: pd.DataFrame, activate: bool = False) -> None:
    OCR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OCR_RAW_DIR / "raw_all_forms_split_combined.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    summary = (
        combined.groupby("source_batch", dropna=False)
        .agg(
            rows=("ballot_record_id", "count"),
            polling_units=("polling_unit_id", "nunique"),
            ballot_records=("ballot_record_id", "nunique"),
            needs_review=("needs_review", lambda s: sum(_to_bool(v) for v in s)),
            vote_sum_false=("vote_sum_match", lambda s: sum(not _to_bool(v, default=True) for v in s)),
            ballot_sum_false=("ballot_sum_match", lambda s: sum(not _to_bool(v, default=True) for v in s)),
        )
        .reset_index()
    )
    summary.to_csv(OCR_RAW_DIR / "combined_batch_summary.csv", index=False, encoding="utf-8-sig")

    missing_rows = []
    for unit, group in combined.groupby("polling_unit_id", dropna=False):
        kinds = set(group["ballot_kind"].dropna().astype(str))
        if not {"constituency", "party_list"}.issubset(kinds):
            missing_rows.append(
                {
                    "polling_unit_id": unit,
                    "present_kinds": "|".join(sorted(kinds)),
                    "source_batch": "|".join(sorted(set(group["source_batch"].astype(str)))),
                    "source_file": "|".join(sorted(set(group["source_file"].astype(str)))),
                }
            )
    pd.DataFrame(
        missing_rows,
        columns=["polling_unit_id", "present_kinds", "source_batch", "source_file"],
    ).to_csv(
        OCR_RAW_DIR / "combined_missing_ballot_kind.csv", index=False, encoding="utf-8-sig"
    )

    review = combined[combined["combined_review_reason"].astype(str).ne("")].copy()
    review.to_csv(OCR_RAW_DIR / "combined_review_queue.csv", index=False, encoding="utf-8-sig")

    if activate:
        active_path = OCR_RAW_DIR / "raw_all_forms_split.csv"
        if active_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = OCR_RAW_DIR / f"raw_all_forms_split.before_combined_{stamp}.csv"
            shutil.copy2(active_path, backup)
            print(f"Backed up active file -> {backup}")
        shutil.copy2(combined_path, active_path)
        shutil.copy2(combined_path, OCR_RAW_DIR / "raw_election_split.csv")
        print("Activated combined file as raw_all_forms_split.csv and raw_election_split.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", action="store_true", help="Back up and replace active raw split files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = combine_batches(DEFAULT_BATCHES)
    write_outputs(combined, activate=args.activate)
    print("\n=== Combined OCR Batches ===")
    print(f"rows: {len(combined)}")
    print(f"polling_units: {combined['polling_unit_id'].nunique()}")
    print(f"ballot_records: {combined['ballot_record_id'].nunique()}")
    print(combined["ballot_kind"].value_counts(dropna=False).to_string())
    print("\nWrote:")
    print(f"- {OCR_RAW_DIR / 'raw_all_forms_split_combined.csv'}")
    print(f"- {OCR_RAW_DIR / 'combined_batch_summary.csv'}")
    print(f"- {OCR_RAW_DIR / 'combined_missing_ballot_kind.csv'}")
    print(f"- {OCR_RAW_DIR / 'combined_review_queue.csv'}")


if __name__ == "__main__":
    main()
