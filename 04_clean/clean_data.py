"""
Phase 3A: Clean OCR output — fix common errors, normalize party names,
handle missing values.

Usage:
    python 04_clean/clean_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz, process
from loguru import logger
import re
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    OCR_RAW_DIR,
    CLEANED_DIR,
    REFERENCE_DIR,
    PARTY_NAMES,
    CONSTITUENCY_NUMBER,
    PROVINCE,
)


class DataCleaner:
    """Clean and normalize raw OCR election data."""

    # Common OCR misreads for Thai/numeric text
    OCR_CORRECTIONS = {
        "๑": "1", "๒": "2", "๓": "3", "๔": "4", "๕": "5",
        "๖": "6", "๗": "7", "๘": "8", "๙": "9", "๐": "0",
        "o": "0", "O": "0", "l": "1", "I": "1",
        "S": "5", "B": "8", "Z": "2", "G": "6",
    }

    def __init__(self):
        self.party_map = self._load_party_reference()
        self.party_name_list = list(self.party_map.values())

    def _load_party_reference(self) -> dict[int, str]:
        """Load official party-list numbers from CSV, falling back to config."""
        ref_path = REFERENCE_DIR / "party_reference.csv"
        if not ref_path.exists():
            return PARTY_NAMES

        try:
            ref = pd.read_csv(ref_path)
            required = {"party_number", "party_name"}
            if not required.issubset(ref.columns):
                logger.warning(f"Party reference missing columns {required}: {ref_path}")
                return PARTY_NAMES
            ref = ref.dropna(subset=["party_number", "party_name"])
            party_map = {
                int(row.party_number): str(row.party_name).strip()
                for row in ref.itertuples(index=False)
                if str(row.party_name).strip()
            }
            if party_map:
                logger.info(f"Loaded {len(party_map)} party names from {ref_path}")
                return party_map
        except Exception as exc:
            logger.warning(f"Could not load party reference {ref_path}: {exc}")
        return PARTY_NAMES

    def clean_all(self) -> pd.DataFrame:
        """Run full cleaning pipeline on all raw OCR data."""
        # Prefer form-level split data when available. It has one row for
        # constituency and one row for party-list per PDF, unlike the legacy
        # one-row-per-PDF raw output.
        split_path = OCR_RAW_DIR / "raw_all_forms_split.csv"
        raw_path = split_path if split_path.exists() else OCR_RAW_DIR / "raw_all_forms.csv"
        if not raw_path.exists():
            logger.error("No raw data found. Run OCR pipeline first.")
            return pd.DataFrame()

        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} raw records from {raw_path}")

        # Remove failed OCR records
        df = df[df.get("ocr_success", True) == True].copy()

        # Drop records with no usable extraction (empty raw_text_preview, all vote/ballot cols empty,
        # or zero confidence). These propagate as garbage rows downstream otherwise.
        if "raw_text_preview" in df.columns:
            empty_preview = df["raw_text_preview"].astype(str).str.strip().isin(["", "{}"])
            df = df[~empty_preview].copy()
        if "ocr_confidence" in df.columns:
            df = df[df["ocr_confidence"].fillna(0) > 0].copy()
        logger.info(f"After dropping empty/error rows: {len(df)} records")

        # Collapse legacy page-level OCR rows (foo_page1.png/foo_page2.png)
        # into one polling-unit row before validation.
        df = self._merge_page_records(df)

        # Filter to target constituency only (skip if PROVINCE is None)
        before = len(df)
        if PROVINCE is not None:
            if "constituency_number" in df.columns:
                if CONSTITUENCY_NUMBER is not None:
                    mask = df["constituency_number"].fillna(0).astype(int).isin([0, CONSTITUENCY_NUMBER])
                    df = df[mask].copy()
            if "province" in df.columns:
                prov = df["province"].fillna("").astype(str)
                mask = (prov == "") | prov.str.contains(PROVINCE, na=False)
                df = df[mask].copy()
            dropped = before - len(df)
            if dropped:
                logger.warning(f"Dropped {dropped} rows outside target {PROVINCE} เขต {CONSTITUENCY_NUMBER}")

        # Step 0: Reclassify form_type from source_file (fix "election" bucket)
        df = self._reclassify_form_type(df)

        # Step 0b: Force province/constituency to project scope
        if PROVINCE is not None and "province" in df.columns:
            df["province"] = PROVINCE
        if CONSTITUENCY_NUMBER is not None and "constituency_number" in df.columns:
            df["constituency_number"] = CONSTITUENCY_NUMBER

        # Step 1: Fix numeric OCR errors
        df = self._fix_numeric_errors(df)

        # Step 2: Normalize party names
        df = self._normalize_party_names(df)

        # Step 3: Handle missing values
        df = self._handle_missing(df)

        # Step 4: Enforce data types
        df = self._enforce_types(df)

        # Step 4b: Sanity check — drop impossible vote counts
        df = self._sanity_check_votes(df)

        # Step 5: Derive computed columns
        df = self._compute_derived(df)

        # Step 6: Map candidate numbers to Party Names
        df = self._map_party_votes(df)

        # Save cleaned data
        cleaned_path = CLEANED_DIR / "election_results_cleaned.csv"
        df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        logger.info(f"Cleaned data saved: {len(df)} records → {cleaned_path}")

        return df

    def _merge_page_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge rows split by PDF page into one record per source document."""
        if "source_file" not in df.columns:
            return df

        def doc_key(name: str) -> str:
            s = str(name)
            s = re.sub(r"_page\d+(?=\.[^.]+$)", "", s)
            return re.sub(r"\.[^.]+$", "", s)

        df = df.copy()
        df["_doc_key"] = df["source_file"].apply(doc_key)
        groups = df.groupby("_doc_key", sort=False, dropna=False)
        if groups.ngroups == len(df):
            return df.drop(columns=["_doc_key"])

        merged_rows = []
        vote_cols = [
            c for c in df.columns
            if re.match(r"^candidate_\d+_votes$", c) or re.match(r"^party_\d+_votes$", c)
        ]
        max_cols = ["good_ballots", "bad_ballots", "no_vote_ballots", "total_ballots", "ocr_confidence"]

        for _, g in groups:
            if len(g) == 1:
                merged_rows.append(g.iloc[0].drop(labels=["_doc_key"]).to_dict())
                continue

            row = g.iloc[0].drop(labels=["_doc_key"]).to_dict()
            row["source_file"] = f"{g.iloc[0]['_doc_key']}.pdf"
            if "polling_unit_id" in g.columns:
                row["polling_unit_id"] = str(g.iloc[0].get("polling_unit_id") or g.iloc[0]["_doc_key"])
            else:
                row["polling_unit_id"] = str(g.iloc[0]["_doc_key"])
            row["n_pages"] = int(len(g))

            for col in vote_cols:
                row[col] = pd.to_numeric(g[col], errors="coerce").fillna(0).sum()
            for col in max_cols:
                if col in g.columns:
                    row[col] = pd.to_numeric(g[col], errors="coerce").max()
            for col in ["station_id", "constituency_number"]:
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce")
                    nonzero = vals[vals.fillna(0) > 0]
                    row[col] = int(nonzero.iloc[0]) if len(nonzero) else 0
            for col in ["province", "form_type"]:
                if col in g.columns:
                    vals = g[col].dropna().astype(str)
                    vals = vals[~vals.str.lower().isin(["", "nan"])]
                    if len(vals):
                        row[col] = vals.iloc[0]
            if "raw_text_preview" in g.columns:
                row["raw_text_preview"] = " | ".join(
                    g["raw_text_preview"].dropna().astype(str).head(3)
                )[:200]
            if "ocr_success" in g.columns:
                row["ocr_success"] = bool(g["ocr_success"].astype(bool).any())
            merged_rows.append(row)

        logger.info(f"Merged page-level rows: {len(df)} rows -> {len(merged_rows)} document rows")
        return pd.DataFrame(merged_rows)

    def _reclassify_form_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-derive form_type from source_file/group_id when it's missing or
        bucketed as 'election'. The OCR pipeline sometimes drops every PDF
        from one folder under one label even when the folder mixes 5_17 / 5_18
        and constituency vs party-list (บช) variants.

        Rules (case-insensitive):
            *5_18*บช* or *5_18bช* or *5_18_บช*  → 5_18_party
            *5_17*บช*                            → 5_17_party
            *5_16*บช*                            → 5_16_party
            *5_18*                               → 5_18
            *5_17*                               → 5_17
            *5_16*                               → 5_16
        """
        if "source_file" not in df.columns and "group_id" not in df.columns:
            return df
        key = "source_file" if "source_file" in df.columns else "group_id"

        def classify(name: str) -> str | None:
            s = str(name)
            is_party = "บช" in s or "_bช" in s.lower() or "bช" in s
            for ft in ("5_18", "5_17", "5_16"):
                if ft in s:
                    return f"{ft}_party" if is_party else ft
            return None

        before = df["form_type"].astype(str).copy() if "form_type" in df.columns else None
        new_ft = df[key].apply(classify)
        if "ballot_kind" in df.columns:
            kind = df["ballot_kind"].fillna("").astype(str).str.lower()
            is_party_kind = (
                kind.isin(["party_list", "party-list", "party", "บัญชีรายชื่อ"])
                | kind.str.contains("บัญชี", na=False)
            )
            is_const_kind = kind.isin(["constituency", "เขต", "แบ่งเขต"]) | kind.str.contains("เขต", na=False)
            no_name_class = new_ft.isna()
            new_ft.loc[no_name_class & is_party_kind] = "5_18_party"
            new_ft.loc[no_name_class & is_const_kind] = "5_18"
        # Only overwrite when current is missing/blank/'election' or differs from a confident detection
        if "form_type" not in df.columns:
            df["form_type"] = new_ft
        else:
            need_fix = df["form_type"].isna() | df["form_type"].astype(str).str.lower().isin(
                ["", "nan", "election", "unknown"]
            )
            df.loc[need_fix & new_ft.notna(), "form_type"] = new_ft[need_fix & new_ft.notna()]
            # Also fix mismatched บช rows (constituency form mislabeled as party-list or vice-versa)
            mismatch = (df["form_type"] != new_ft) & new_ft.notna()
            if mismatch.any():
                df.loc[mismatch, "form_type"] = new_ft[mismatch]
        if before is not None:
            changed = (before != df["form_type"]).sum()
            if changed:
                logger.info(f"Reclassified form_type for {changed} rows")
        return df

    def _sanity_check_votes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop per-candidate vote values that exceed plausible ceilings.

        A candidate cannot receive more votes than `good_ballots` (or, failing
        that, `total_ballots`). Anything beyond that is OCR error and gets
        zeroed out so downstream sums remain valid.
        """
        cand_cols = [c for c in df.columns
                     if re.match(r"^candidate_\d+_votes$", c)
                     or re.match(r"^party_.+_votes$", c)]
        if not cand_cols:
            return df

        # Pick the tightest ceiling available per row
        if "good_ballots" in df.columns:
            ceiling = pd.to_numeric(df["good_ballots"], errors="coerce")
        elif "total_ballots" in df.columns:
            ceiling = pd.to_numeric(df["total_ballots"], errors="coerce")
        else:
            return df
        # If good_ballots itself looks corrupt (0 with non-zero total), fall back
        if "total_ballots" in df.columns:
            tot = pd.to_numeric(df["total_ballots"], errors="coerce")
            bad_ceiling = (ceiling.fillna(0) <= 0) & (tot.fillna(0) > 0)
            ceiling = ceiling.where(~bad_ceiling, tot)

        n_dropped = 0
        total_votes_dropped = 0
        for col in cand_cols:
            v = pd.to_numeric(df[col], errors="coerce")
            mask = v.notna() & ceiling.notna() & (v > ceiling) & (ceiling > 0)
            if mask.any():
                n_dropped += int(mask.sum())
                total_votes_dropped += float(v[mask].sum())
                df.loc[mask, col] = 0
        if n_dropped:
            logger.warning(
                f"Sanity check: zeroed {n_dropped} impossible vote cells "
                f"(removed {total_votes_dropped:,.0f} bogus votes; rule: votes > ballot ceiling)"
            )
        return df

    def _fix_numeric_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common OCR misreads in numeric columns."""
        vote_cols = [c for c in df.columns if "votes" in c or "ballots" in c]

        for col in vote_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self._correct_digits)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _correct_digits(self, value: str) -> str:
        """Apply character-level corrections for OCR digit misreads."""
        if pd.isna(value) or value == "nan":
            return "0"
        for wrong, right in self.OCR_CORRECTIONS.items():
            value = value.replace(wrong, right)
        # Remove any remaining non-digit characters
        return re.sub(r"[^\d.]", "", value)

    def _normalize_party_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use fuzzy matching to correct misspelled party names.

        Catches both legacy `party_name*` columns and per-candidate `candidate_<N>_party`
        columns produced by Stage B.
        """
        party_cols = [
            c for c in df.columns
            if "party_name" in c.lower() or c.endswith("_party")
        ]
        for col in party_cols:
            df[col] = df[col].apply(self._fuzzy_match_party)
        return df

    def _fuzzy_match_party(self, name: str) -> str:
        """Match OCR'd party name to official party list using fuzzy matching."""
        if pd.isna(name) or not name.strip():
            return name

        best_match, score = process.extractOne(
            name.strip(), self.party_name_list,
            scorer=fuzz.ratio
        )
        if score >= 70:  # Threshold for accepting fuzzy match
            return best_match
        return name  # Return original if no good match

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        # Numeric columns: fill with 0 (missing votes = 0 votes)
        vote_cols = [c for c in df.columns if "votes" in c or "ballots" in c]
        for col in vote_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Station ID: flag but don't drop
        if "station_id" in df.columns:
            missing_stations = df["station_id"].isna().sum()
            if missing_stations > 0:
                logger.warning(f"{missing_stations} records missing station_id")

        return df

    def _enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce correct data types on all columns."""
        int_cols = [c for c in df.columns
                    if any(k in c for k in ["votes", "ballots", "station_id",
                                            "constituency_number"])]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived columns for analysis."""
        if "good_ballots" in df.columns and "total_ballots" in df.columns:
            df["turnout_valid_ratio"] = (
                df["good_ballots"] / df["total_ballots"].replace(0, np.nan)
            ).round(4)

        if "bad_ballots" in df.columns and "total_ballots" in df.columns:
            df["invalid_ballot_ratio"] = (
                df["bad_ballots"] / df["total_ballots"].replace(0, np.nan)
            ).round(4)

        return df

    def _map_party_votes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create named party vote columns — only for party-list forms (5_*_party).

        Constituency forms (5_18, 5_16, 5_17) use LOCAL candidate numbers
        (e.g. candidate_1 = the first candidate running in this constituency),
        which do NOT map to the nationwide party list.
        """
        if "form_type" not in df.columns:
            return df
        party_mask = df["form_type"].astype(str).str.endswith("_party")
        if not party_mask.any():
            return df
        for num, name in self.party_map.items():
            candidate_col = f"candidate_{num}_votes"
            if candidate_col in df.columns:
                df.loc[party_mask, f"party_{name}_votes"] = df.loc[party_mask, candidate_col]
        return df

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_all()
    print(f"\nCleaned dataset shape: {cleaned_df.shape}")
