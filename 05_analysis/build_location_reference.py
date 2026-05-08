"""
Build a conservative polling-unit location reference.

This script does not trust OCR geography as official truth. It creates a
draft table from source PDF paths and OCR markdown headers, then leaves
explicit official/verified columns for human or official-reference review.

Usage:
    python 05_analysis/build_location_reference.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config import FIGURES_DIR, OCR_RAW_DIR, PROJECT_ROOT, REFERENCE_DIR

PDF_ROOT = PROJECT_ROOT / "3"
MARKDOWN_DIR = OCR_RAW_DIR / "markdown" / "election"
DRAFT_PATH = REFERENCE_DIR / "polling_unit_locations_draft.csv"
REFERENCE_PATH = REFERENCE_DIR / "polling_unit_locations.csv"
TAMBON_REFERENCE_PATH = REFERENCE_DIR / "tambon_reference.csv"
GEO_WINNERS_PATH = FIGURES_DIR / "geographic_winner_summary.csv"
TAMBON_PARTY_SUMMARY_PATH = FIGURES_DIR / "tambon_party_summary.csv"

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
EXPECTED_PROVINCE = "กำแพงเพชร"
EXPECTED_DISTRICT = "เมืองกำแพงเพชร"
MANUAL_LOCATION_COLUMNS = [
    "official_province",
    "official_district",
    "official_subdistrict",
    "official_municipality",
    "official_moo",
    "lat",
    "lon",
    "tambon_code",
    "amphoe_code",
    "province_code",
    "location_verified",
    "verification_source",
    "verification_note",
]


def safe_stem(path: Path) -> str:
    """Match 02_preprocess/convert_pdfs.py naming for polling-unit keys."""
    rel = path.relative_to(PDF_ROOT)
    full = "_".join(rel.with_suffix("").parts)
    full = re.sub(r"[^\wก-๙]+", "_", full, flags=re.UNICODE)
    full = re.sub(r"_+", "_", full).strip("_")
    return full[:200] if len(full) > 200 else full


def normalize_text(value: object) -> str:
    text = str(value or "").translate(THAI_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def loose_key(value: object) -> str:
    """Match OCR-safe filenames to full Thai source paths without guessing geography."""
    text = normalize_text(value).lower()
    text = re.sub(r"\.(pdf|md)$", "", text)
    text = re.sub(r"[\u0e31-\u0e3a\u0e40-\u0e4e]", "", text)
    text = re.sub(r"[^\wก-ฮ0-9]+", "", text, flags=re.UNICODE)
    text = text.replace("_", "")
    return text


def extract_number(text: object, patterns: list[str]) -> float:
    value = normalize_text(text)
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return np.nan


def parse_pdf_path(path: Path) -> dict:
    if path is None:
        return {
            "pdf_path": "",
            "path_folder": "",
            "path_local_government": "",
            "path_local_government_type": "",
            "path_local_government_name": "",
            "path_unit_number": np.nan,
        }
    parts = path.relative_to(PDF_ROOT).parts
    local_government = parts[-2] if len(parts) >= 2 else ""
    local_type = ""
    local_name = local_government
    if "." in local_government:
        local_type, local_name = local_government.split(".", 1)
    unit = extract_number(
        path.stem,
        [
            r"หน่วย(?:เลือกตั้ง)?ที่\s*(\d+)",
            r"หน่วยที่\s*(\d+)",
            r"หน่วย\s*(\d+)",
            r"_(\d+)(?:_\d+)?$",
        ],
    )
    return {
        "pdf_path": str(path),
        "path_folder": parts[0] if parts else "",
        "path_local_government": local_government,
        "path_local_government_type": local_type,
        "path_local_government_name": local_name,
        "path_unit_number": unit,
    }


def read_markdown(stem: str) -> str:
    path = MARKDOWN_DIR / f"{stem}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def field_between(text: str, start: str, end: str) -> str:
    pattern = rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}"
    match = re.search(pattern, text, flags=re.DOTALL)
    return normalize_text(match.group(1)) if match else ""


def parse_markdown_header(text: str) -> dict:
    compact = normalize_text(text)
    header = compact[:2500]
    unit = extract_number(header, [r"หน่วยเลือกตั้งที่\s*(\d+)", r"หน่วยที่\s*(\d+)"])
    moo = extract_number(header, [r"หมู่ที่\s*(\d+)"])
    locality = field_between(header, "ตำบล/แขวง/เทศบาล", "อำเภอ/เขต")
    district = field_between(header, "อำเภอ/เขต", "เขตเลือกตั้ง")
    constituency = extract_number(header, [r"เขตเลือกตั้งที่\s*(\d+)"])
    province = ""
    province_match = re.search(r"จังหวัด\s*([ก-๙A-Za-z0-9 ]{2,40})", header)
    if province_match:
        province = normalize_text(province_match.group(1))
        province = re.split(r"(?:เสร็จ|สร็จ|โสร็จ|ดังนั้น|จำนวน|รวม|หน้า|ผู้|ลำดับ)", province)[0].strip()
        if EXPECTED_PROVINCE in province:
            province = EXPECTED_PROVINCE
    return {
        "header_unit_number": unit,
        "header_moo": moo,
        "header_locality": locality,
        "header_district": district,
        "header_constituency_number": constituency,
        "header_province": province,
        "has_markdown": bool(text.strip()),
    }


def load_winners() -> pd.DataFrame:
    candidates = [
        FIGURES_DIR / "evidence_fingerprint_map.csv",
        OCR_RAW_DIR / "raw_all_forms_split.csv",
        OCR_RAW_DIR / "raw_election_split.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "polling_unit_id" in df.columns:
                logger.info(f"Loaded winner context from {path}")
                return df
    return pd.DataFrame()


def load_tambon_reference() -> pd.DataFrame:
    if not TAMBON_REFERENCE_PATH.exists():
        return pd.DataFrame()
    ref = pd.read_csv(TAMBON_REFERENCE_PATH)
    required = {"path_local_government", "official_province", "official_district", "official_subdistrict"}
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"{TAMBON_REFERENCE_PATH} is missing columns: {sorted(missing)}")
    return ref.drop_duplicates("path_local_government")


def load_polling_units(winners: pd.DataFrame) -> pd.DataFrame:
    source_candidates = [
        OCR_RAW_DIR / "raw_all_forms_split.csv",
        OCR_RAW_DIR / "raw_election_split.csv",
        OCR_RAW_DIR / "raw_election_split_checkpoint.csv",
    ]
    for path in source_candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"polling_unit_id", "source_file"}.issubset(df.columns):
                return df[["polling_unit_id", "source_file"]].drop_duplicates("polling_unit_id")
    if not winners.empty and "polling_unit_id" in winners.columns:
        out = winners[["polling_unit_id"]].drop_duplicates("polling_unit_id")
        out["source_file"] = out["polling_unit_id"].astype(str) + ".pdf"
        return out
    rows = []
    for pdf_path in sorted(PDF_ROOT.rglob("*.pdf")):
        stem = safe_stem(pdf_path)
        rows.append({"polling_unit_id": stem, "source_file": f"{stem}.pdf"})
    return pd.DataFrame(rows)


def build_pdf_lookup() -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for pdf_path in sorted(PDF_ROOT.rglob("*.pdf")):
        lookup[loose_key(safe_stem(pdf_path))] = pdf_path
    return lookup


def build_reference() -> pd.DataFrame:
    winners = load_winners()
    unit_ids = load_polling_units(winners)
    pdf_lookup = build_pdf_lookup()
    pdf_rows = []
    if not PDF_ROOT.exists():
        raise FileNotFoundError(f"PDF root not found: {PDF_ROOT}")
    for _, unit_row in unit_ids.iterrows():
        stem = str(unit_row["polling_unit_id"])
        pdf_path = pdf_lookup.get(loose_key(stem))
        row = {
            "polling_unit_id": stem,
            "source_file": unit_row.get("source_file", f"{stem}.pdf"),
            **parse_pdf_path(pdf_path),
        }
        row.update(parse_markdown_header(read_markdown(stem)))
        pdf_rows.append(row)

    locations = pd.DataFrame(pdf_rows)
    if locations.empty:
        raise ValueError("No PDF files found for location reference.")

    locations["draft_unit_number"] = locations["header_unit_number"].fillna(locations["path_unit_number"])
    locations["draft_moo"] = locations["header_moo"]
    locations["draft_municipality"] = locations["path_local_government"]
    locations["draft_subdistrict_or_municipality"] = locations["header_locality"].replace("", np.nan).infer_objects(copy=False).fillna(
        locations["path_local_government_name"]
    )
    locations["draft_district"] = locations["header_district"]
    locations["draft_province"] = locations["header_province"]

    unit_match = (
        locations["header_unit_number"].isna()
        | locations["path_unit_number"].isna()
        | locations["header_unit_number"].eq(locations["path_unit_number"])
    )
    province_ok = locations["header_province"].eq("") | locations["header_province"].str.contains(EXPECTED_PROVINCE, na=False)
    district_ok = locations["header_district"].eq("") | locations["header_district"].str.contains("เมือง", na=False)
    locality_ok = locations.apply(
        lambda r: str(r["path_local_government_name"]) in str(r["header_locality"])
        or str(r["header_locality"]) in str(r["path_local_government_name"])
        or not str(r["header_locality"]).strip(),
        axis=1,
    )
    locations["path_header_unit_match"] = unit_match
    locations["path_header_locality_match"] = locality_ok
    locations["province_expected_match"] = province_ok
    locations["district_expected_match"] = district_ok
    locations["missing_moo"] = locations["header_moo"].isna()

    def review_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not bool(row.get("has_markdown")):
            reasons.append("missing_markdown")
        if not bool(row.get("path_header_unit_match")):
            reasons.append("unit_number_mismatch")
        if not bool(row.get("path_header_locality_match")):
            reasons.append("locality_mismatch")
        if not bool(row.get("province_expected_match")):
            reasons.append("province_ocr_unreliable")
        if not bool(row.get("district_expected_match")):
            reasons.append("district_ocr_unreliable")
        if bool(row.get("missing_moo")):
            reasons.append("missing_moo")
        return ";".join(reasons)

    locations["location_review_reason"] = locations.apply(review_reason, axis=1)
    locations["needs_location_review"] = locations["location_review_reason"].ne("")
    locations["location_confidence"] = np.select(
        [
            ~locations["has_markdown"],
            locations["needs_location_review"],
            locations["header_moo"].isna(),
        ],
        [0.35, 0.55, 0.75],
        default=0.90,
    )

    official_cols = {
        "official_province": "",
        "official_district": "",
        "official_subdistrict": "",
        "official_municipality": "",
        "official_moo": "",
        "lat": "",
        "lon": "",
        "tambon_code": "",
        "amphoe_code": "",
        "province_code": "",
        "location_verified": False,
        "verification_source": "",
        "verification_note": "",
    }
    for col, default in official_cols.items():
        locations[col] = default

    tambon_ref = load_tambon_reference()
    if not tambon_ref.empty:
        merge_cols = [
            c
            for c in [
                "path_local_government",
                "official_province",
                "official_district",
                "official_subdistrict",
                "tambon_verified",
                "tambon_lat",
                "tambon_lon",
                "verification_source",
                "verification_note",
            ]
            if c in tambon_ref.columns
        ]
        locations = locations.merge(tambon_ref[merge_cols], on="path_local_government", how="left", suffixes=("", "_tambon"))
        for col in ["official_province", "official_district", "official_subdistrict", "verification_source", "verification_note"]:
            tambon_col = f"{col}_tambon"
            if tambon_col in locations.columns:
                locations[col] = locations[col].replace("", np.nan).fillna(locations[tambon_col]).fillna("")
                locations = locations.drop(columns=[tambon_col])
        if "tambon_verified" in locations.columns:
            locations["tambon_verified"] = locations["tambon_verified"].fillna(False).astype(bool)
        else:
            locations["tambon_verified"] = False
        for col in ["tambon_lat", "tambon_lon"]:
            if col not in locations.columns:
                locations[col] = ""

    if not winners.empty:
        winner_cols = [
            c
            for c in [
                "polling_unit_id",
                "ballot_kind",
                "winner_party",
                "winner_votes",
                "winner_share",
                "is_confirmed",
                "quality_label",
                "review_reason",
            ]
            if c in winners.columns
        ]
        if {"polling_unit_id", "ballot_kind"}.issubset(winner_cols):
            geo = winners[winner_cols].merge(locations, on="polling_unit_id", how="left")
            GEO_WINNERS_PATH.parent.mkdir(parents=True, exist_ok=True)
            geo.to_csv(GEO_WINNERS_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"Wrote geographic winner summary: {GEO_WINNERS_PATH}")

    build_tambon_party_summary(locations)

    ordered = [
        "polling_unit_id",
        "source_file",
        "draft_province",
        "draft_district",
        "draft_subdistrict_or_municipality",
        "draft_municipality",
        "draft_moo",
        "draft_unit_number",
        "official_province",
        "official_district",
        "official_subdistrict",
        "official_municipality",
        "official_moo",
        "lat",
        "lon",
        "tambon_code",
        "amphoe_code",
        "province_code",
        "location_verified",
        "tambon_verified",
        "tambon_lat",
        "tambon_lon",
        "verification_source",
        "verification_note",
        "needs_location_review",
        "location_review_reason",
        "location_confidence",
        "path_header_unit_match",
        "path_header_locality_match",
        "province_expected_match",
        "district_expected_match",
        "missing_moo",
        "header_unit_number",
        "header_moo",
        "header_locality",
        "header_district",
        "header_constituency_number",
        "header_province",
        "has_markdown",
        "path_local_government",
        "path_local_government_type",
        "path_local_government_name",
        "path_unit_number",
        "path_folder",
        "pdf_path",
    ]
    return locations[[c for c in ordered if c in locations.columns]]


def build_tambon_party_summary(locations: pd.DataFrame) -> None:
    try:
        from analysis import build_party_long, constituency_party_lookup, load_election_data
    except Exception as exc:
        logger.warning(f"Could not import analysis helpers for tambon summary: {exc}")
        return

    try:
        df, _ = load_election_data()
    except Exception as exc:
        logger.warning(f"Could not load election data for tambon summary: {exc}")
        return

    long = build_party_long(df, constituency_party_lookup(df))
    if long.empty:
        return

    loc_cols = [
        "polling_unit_id",
        "path_local_government",
        "official_province",
        "official_district",
        "official_subdistrict",
        "tambon_verified",
        "tambon_lat",
        "tambon_lon",
        "needs_location_review",
        "location_review_reason",
    ]
    joined = long.merge(locations[[c for c in loc_cols if c in locations.columns]], on="polling_unit_id", how="left")
    joined["tambon_label"] = joined["official_subdistrict"].replace("", np.nan).fillna(joined["path_local_government"])
    grouped = (
        joined.groupby(
            [
                "ballot_kind",
                "official_province",
                "official_district",
                "tambon_label",
                "party",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            votes=("votes", "sum"),
            polling_units=("polling_unit_id", "nunique"),
            confirmed_party_rows=("is_confirmed", "sum"),
            party_rows=("polling_unit_id", "size"),
            mean_unit_share=("vote_share", "mean"),
            tambon_verified=("tambon_verified", "all"),
            tambon_lat=("tambon_lat", "first"),
            tambon_lon=("tambon_lon", "first"),
        )
    )
    totals = grouped.groupby(["ballot_kind", "tambon_label"])["votes"].transform("sum").replace(0, np.nan)
    grouped["tambon_vote_share"] = grouped["votes"] / totals
    grouped["tambon_rank"] = grouped.groupby(["ballot_kind", "tambon_label"])["votes"].rank(
        method="dense", ascending=False
    )
    grouped["is_tambon_winner"] = grouped["tambon_rank"].eq(1)
    grouped = grouped.sort_values(["ballot_kind", "tambon_label", "tambon_rank", "party"])
    TAMBON_PARTY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(TAMBON_PARTY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote tambon party summary: {TAMBON_PARTY_SUMMARY_PATH}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    locations = build_reference()
    locations.to_csv(DRAFT_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote draft location reference: {DRAFT_PATH}")

    if REFERENCE_PATH.exists():
        existing = pd.read_csv(REFERENCE_PATH)
        keep_cols = [c for c in ["polling_unit_id", *MANUAL_LOCATION_COLUMNS] if c in existing.columns]
        if keep_cols:
            manual = existing[keep_cols].drop_duplicates("polling_unit_id")
            merged = locations.merge(manual, on="polling_unit_id", how="left", suffixes=("", "_manual"))
            for col in MANUAL_LOCATION_COLUMNS:
                manual_col = f"{col}_manual"
                if col not in merged.columns:
                    merged[col] = ""
                if manual_col in merged.columns:
                    manual_values = merged[manual_col]
                    if col == "location_verified":
                        merged[col] = manual_values.where(manual_values.notna(), merged[col]).fillna(False)
                    else:
                        manual_values = manual_values.replace("", np.nan)
                        merged[col] = manual_values.fillna(merged[col]).fillna("")
                    merged = merged.drop(columns=[manual_col])
            for col in ["location_verified"]:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(False)
            locations = merged
        locations.to_csv(REFERENCE_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"Updated editable location reference while preserving manual columns: {REFERENCE_PATH}")
    else:
        locations.to_csv(REFERENCE_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"Created editable location reference: {REFERENCE_PATH}")

    print("\n=== Location Reference Summary ===")
    print(f"polling_units: {len(locations)}")
    print(f"needs_location_review: {int(locations['needs_location_review'].sum())}")
    print(f"has_markdown: {int(locations['has_markdown'].sum())}")
    print("\nBy municipality:")
    summary = locations["path_local_government"].value_counts().to_string()
    print(summary.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
