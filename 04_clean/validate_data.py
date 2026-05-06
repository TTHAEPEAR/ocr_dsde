"""
Phase 3B: Cross-validation of election data.
Ensures internal consistency and compares against reference data.

Usage:
    python 04_clean/validate_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import re
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, REFERENCE_DIR


class DataValidator:
    """Validate cleaned election data for correctness."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_all(self, df: pd.DataFrame) -> bool:
        """
        Run all validation checks.

        Returns True if all critical checks pass.
        """
        logger.info("=== Starting Data Validation ===")

        self._check_ballot_sum_consistency(df)
        self._check_vote_totals(df)
        self._check_station_completeness(df)
        self._check_negative_values(df)
        self._check_duplicates(df)
        self._compare_with_reference(df)
        self._write_review_queue(df)

        # Report results
        logger.info(f"\nValidation complete:")
        logger.info(f"  Errors:   {len(self.errors)}")
        logger.info(f"  Warnings: {len(self.warnings)}")

        if self.errors:
            logger.error("CRITICAL ERRORS FOUND:")
            for e in self.errors:
                logger.error(f"  - {e}")

        if self.warnings:
            logger.warning("WARNINGS:")
            for w in self.warnings:
                logger.warning(f"  - {w}")

        # Save validation report
        report = {
            "errors": self.errors,
            "warnings": self.warnings,
            "total_records": len(df),
            "passed": len(self.errors) == 0
        }
        report_path = CLEANED_DIR / "validation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Validation Report ===\n\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Status: {'PASSED' if report['passed'] else 'FAILED'}\n\n")
            f.write("ERRORS:\n")
            for e in self.errors:
                f.write(f"  - {e}\n")
            f.write("\nWARNINGS:\n")
            for w in self.warnings:
                f.write(f"  - {w}\n")

        return len(self.errors) == 0

    def _check_ballot_sum_consistency(self, df: pd.DataFrame):
        """Check: good_ballots + bad_ballots + no_vote_ballots ≈ total_ballots"""
        required = ["good_ballots", "bad_ballots", "no_vote_ballots", "total_ballots"]
        if not all(c in df.columns for c in required):
            self.warnings.append("Missing ballot columns, skipping sum check")
            return

        df["_computed_total"] = (
            df["good_ballots"] + df["bad_ballots"] + df["no_vote_ballots"]
        )
        mismatches = df[df["_computed_total"] != df["total_ballots"]]

        if len(mismatches) > 0:
            pct = len(mismatches) / len(df) * 100
            if pct > 10:
                self.errors.append(
                    f"Ballot sum mismatch in {len(mismatches)} records ({pct:.1f}%)"
                )
            else:
                self.warnings.append(
                    f"Ballot sum mismatch in {len(mismatches)} records ({pct:.1f}%)"
                )

        df.drop("_computed_total", axis=1, inplace=True)

    def _check_vote_totals(self, df: pd.DataFrame):
        """Check: sum of candidate/party votes ≤ good_ballots."""
        vote_cols = [
            c for c in df.columns
            if re.match(r"^candidate_\d+_votes$", c) or re.match(r"^party_\d+_votes$", c)
        ]
        if not vote_cols or "good_ballots" not in df.columns:
            return

        df["_sum_votes"] = df[vote_cols].sum(axis=1)
        over_count = df[df["_sum_votes"] > df["good_ballots"]]

        if len(over_count) > 0:
            self.warnings.append(
                f"Vote sum exceeds good_ballots in {len(over_count)} records"
            )

        df.drop("_sum_votes", axis=1, inplace=True)

    def _check_station_completeness(self, df: pd.DataFrame):
        """Check that we have data for all expected polling stations."""
        if "station_id" not in df.columns:
            return

        for form_type in df["form_type"].unique():
            form_df = df[df["form_type"] == form_type]
            n_stations = form_df["station_id"].nunique()
            logger.info(f"  Form {form_type}: {n_stations} unique stations")

            if form_type in ["5_18", "5_18_party"] and n_stations < 250:
                self.warnings.append(
                    f"Form {form_type} has only {n_stations} stations (need ≥250)"
                )

    def _check_negative_values(self, df: pd.DataFrame):
        """No vote counts should be negative."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                self.errors.append(f"Negative values in '{col}': {neg_count} records")

    def _check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate station records within same form type."""
        if "source_file" in df.columns and "form_type" in df.columns:
            dupes = df.duplicated(subset=["source_file", "form_type"], keep=False)
            label = "source_file+form"
        elif "station_id" in df.columns and "form_type" in df.columns:
            dupes = df.duplicated(subset=["station_id", "form_type"], keep=False)
            label = "station+form"
        else:
            return

        if dupes.sum() > 0:
            self.warnings.append(
                f"Duplicate {label} entries: {dupes.sum()} records"
            )

    def _compare_with_reference(self, df: pd.DataFrame):
        """Compare our OCR results against P'PanJ reference data if available."""
        ref_path = REFERENCE_DIR / "reference_results.csv"
        if not ref_path.exists():
            self.warnings.append("No reference data found for comparison")
            return

        ref_df = pd.read_csv(ref_path)
        logger.info(f"Comparing against reference data ({len(ref_df)} records)")

        # Merge on station_id and form_type for comparison
        if "station_id" in ref_df.columns and "station_id" in df.columns:
            merged = df.merge(
                ref_df, on=["station_id", "form_type"],
                suffixes=("_ours", "_ref"), how="inner"
            )

            # Compare vote columns
            vote_cols_ours = [c for c in merged.columns if c.endswith("_votes_ours")]
            for col in vote_cols_ours:
                ref_col = col.replace("_ours", "_ref")
                if ref_col in merged.columns:
                    diffs = (merged[col] != merged[ref_col]).sum()
                    if diffs > 0:
                        self.warnings.append(
                            f"Mismatch vs reference in {col}: {diffs} records"
                        )

    def _write_review_queue(self, df: pd.DataFrame):
        """Save row-level validation failures for targeted re-OCR/review."""
        reasons = pd.Series("", index=df.index, dtype="object")

        required = ["good_ballots", "bad_ballots", "no_vote_ballots", "total_ballots"]
        if all(c in df.columns for c in required):
            computed_total = df["good_ballots"] + df["bad_ballots"] + df["no_vote_ballots"]
            mask = computed_total != df["total_ballots"]
            reasons.loc[mask] += "ballot_sum_mismatch;"

        vote_cols = [
            c for c in df.columns
            if re.match(r"^candidate_\d+_votes$", c) or re.match(r"^party_\d+_votes$", c)
        ]
        if vote_cols and "good_ballots" in df.columns:
            mask = df[vote_cols].sum(axis=1) > df["good_ballots"]
            reasons.loc[mask] += "vote_sum_exceeds_good_ballots;"

        if "station_id" in df.columns:
            mask = pd.to_numeric(df["station_id"], errors="coerce").fillna(0) <= 0
            reasons.loc[mask] += "missing_station_id;"

        queue = df.loc[reasons != ""].copy()
        if queue.empty:
            return
        queue.insert(0, "review_reason", reasons.loc[queue.index].str.rstrip(";"))
        review_cols = [
            c for c in [
                "review_reason", "source_file", "form_type", "station_id",
                "good_ballots", "bad_ballots", "no_vote_ballots", "total_ballots",
                "ocr_confidence", "raw_text_preview"
            ]
            if c in queue.columns
        ]
        queue[review_cols].to_csv(CLEANED_DIR / "review_queue.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    cleaned_path = CLEANED_DIR / "election_results_cleaned.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        validator = DataValidator()
        passed = validator.validate_all(df)
        print(f"\nValidation {'PASSED' if passed else 'FAILED'}")
    else:
        print("No cleaned data found. Run clean_data.py first.")
