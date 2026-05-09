"""
Split already-OCRed dual election PDFs into separate constituency and
party-list records without touching the original raw OCR outputs.

Inputs:
    data/ocr_raw/markdown/election/*.md
    data/images/election/*_pageN.png

Outputs:
    data/ocr_raw/raw_election_split.csv
    data/ocr_raw/raw_all_forms_split.csv
    data/ocr_raw/raw_election_split_checkpoint.csv
    data/ocr_raw/split_review_queue.csv

This script intentionally does not read or write the original checkpoint files.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    GEMINI_API_KEY,
    IMAGE_DIR,
    OCR_RAW_DIR,
    PARTY_NAMES,
    PROVINCE,
    CONSTITUENCY_NUMBER,
    CONSTITUENCY_CANDIDATE_PARTIES,
)
from ocr_pipeline import OCRPipeline


PAGE_RANGES = {
    "constituency": [1, 2],
    "party_list": [3, 4, 5],
}
SPLIT_CHECKPOINT = OCR_RAW_DIR / "raw_election_split_checkpoint.csv"


def _repair_thai_mojibake(text: str) -> str:
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text
    return f"{text}\n{repaired}"


def _detect_page_kind(page_text: str) -> str | None:
    text = _repair_thai_mojibake(page_text)

    party_score = 0
    if "(บช" in text or "บช)" in text:
        party_score += 6
    if "แบบบัญชีรายชื่อ" in text:
        party_score += 5
    if "หมายเลขของบัญชีรายชื่อ" in text:
        party_score += 5
    if "พรรคการเมืองแต่ละพรรค" in text:
        party_score += 3

    constituency_score = 0
    if "แบบแบ่งเขต" in text:
        constituency_score += 6
    if "ผู้สมัครรับเลือกตั้ง" in text:
        constituency_score += 5
    if "หมายเลขประจำตัว" in text:
        constituency_score += 5
    if "ผู้สมัคร" in text:
        constituency_score += 3

    if party_score > constituency_score and party_score >= 5:
        return "party_list"
    if constituency_score > party_score and constituency_score >= 5:
        return "constituency"
    return None


def _infer_page_ranges(markdown: str) -> dict[str, list[int]]:
    blocks = _parse_page_blocks(markdown)
    inferred = {"constituency": [], "party_list": []}
    default_by_page = {page_no: kind for kind, pages in PAGE_RANGES.items() for page_no in pages}
    detected_by_page = {page_no: _detect_page_kind(block) for page_no, block in blocks.items()}
    current_kind: str | None = None

    sorted_pages = sorted(blocks)
    for idx, page_no in enumerate(sorted_pages):
        detected_kind = detected_by_page[page_no]
        if detected_kind:
            current_kind = detected_kind
            inferred[current_kind].append(page_no)
            continue

        next_detected = None
        for next_page_no in sorted_pages[idx + 1:]:
            if detected_by_page[next_page_no]:
                next_detected = detected_by_page[next_page_no]
                break
        default_kind = default_by_page.get(page_no)
        assigned_kind = current_kind
        if default_kind and (current_kind is None or next_detected == default_kind):
            assigned_kind = default_kind
        if assigned_kind:
            inferred[assigned_kind].append(page_no)

    if inferred["constituency"] and inferred["party_list"]:
        return inferred
    return {kind: pages[:] for kind, pages in PAGE_RANGES.items()}


def _format_page_range(page_numbers: list[int]) -> str:
    if not page_numbers:
        return ""
    sorted_pages = sorted(page_numbers)
    if sorted_pages == list(range(sorted_pages[0], sorted_pages[-1] + 1)):
        return str(sorted_pages[0]) if len(sorted_pages) == 1 else f"{sorted_pages[0]}-{sorted_pages[-1]}"
    return ",".join(str(p) for p in sorted_pages)


def _parse_page_blocks(markdown: str) -> dict[int, str]:
    """Return markdown content keyed by source page number."""
    marker = re.compile(r"^--- Page\s+(\d+)\s+\(.*?\)\s+---\s*$", re.MULTILINE)
    matches = list(marker.finditer(markdown))
    if not matches:
        return {1: markdown}

    pages: dict[int, str] = {}
    for idx, match in enumerate(matches):
        page_no = int(match.group(1))
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        pages[page_no] = markdown[match.start():end].strip()
    return pages


def _select_markdown_pages(markdown: str, page_numbers: list[int]) -> str:
    blocks = _parse_page_blocks(markdown)
    selected = [blocks[p] for p in page_numbers if p in blocks]
    return "\n\n".join(selected).strip()


def _page_images(stem: str, page_numbers: list[int]) -> list[Path]:
    img_dir = IMAGE_DIR / "election"
    return [img_dir / f"{stem}_page{p}.png" for p in page_numbers if (img_dir / f"{stem}_page{p}.png").exists()]


def _record_id(source_file: str, ballot_kind: str) -> str:
    return f"{OCRPipeline._polling_unit_id(source_file)}__{ballot_kind}"


def _load_checkpoint() -> pd.DataFrame:
    if not SPLIT_CHECKPOINT.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SPLIT_CHECKPOINT)
    except Exception as exc:
        logger.warning(f"Could not read split checkpoint {SPLIT_CHECKPOINT}: {exc}")
        return pd.DataFrame()
    if "ballot_record_id" not in df.columns:
        logger.warning(f"Split checkpoint lacks ballot_record_id; ignoring {SPLIT_CHECKPOINT}")
        return pd.DataFrame()
    return df.drop_duplicates(subset=["ballot_record_id"], keep="last").reset_index(drop=True)


def _save_checkpoint(records: list[dict]) -> None:
    pd.DataFrame(records).to_csv(SPLIT_CHECKPOINT, index=False)


def _write_checkpoint_record(record: dict, records: list[dict], seen_ids: set[str]) -> None:
    record_id = str(record.get("ballot_record_id", "")).strip()
    if not record_id:
        raise ValueError("Cannot checkpoint split record without ballot_record_id")
    records[:] = [r for r in records if str(r.get("ballot_record_id", "")).strip() != record_id]
    records.append(record)
    seen_ids.add(record_id)
    _save_checkpoint(records)


def _is_review_record(row: pd.Series) -> bool:
    def is_false(value) -> bool:
        return pd.notna(value) and str(value).strip().lower() in {"false", "0"}

    def is_true(value) -> bool:
        return pd.notna(value) and str(value).strip().lower() in {"true", "1"}

    return (
        is_false(row.get("ocr_success", True))
        or is_true(row.get("needs_review", False))
        or is_false(row.get("vote_sum_match", True))
        or is_false(row.get("ballot_sum_match", True))
        or pd.isna(row.get("ocr_success", True))
        or pd.isna(row.get("vote_sum_match", True))
        or pd.isna(row.get("ballot_sum_match", True))
    )


def _retry_ids_from_checkpoint() -> set[str]:
    checkpoint_df = _load_checkpoint()
    if checkpoint_df.empty:
        return set()
    mask = checkpoint_df.apply(_is_review_record, axis=1)
    return set(checkpoint_df.loc[mask, "ballot_record_id"].astype(str))


def _retry_ids_from_review_queue() -> set[str]:
    review_path = OCR_RAW_DIR / "split_review_queue.csv"
    if not review_path.exists():
        return set()
    try:
        review_df = pd.read_csv(review_path)
    except Exception as exc:
        logger.warning(f"Could not read review queue {review_path}: {exc}")
        return set()
    if "ballot_record_id" not in review_df.columns:
        return set()
    return set(review_df["ballot_record_id"].dropna().astype(str))


def _retry_ids() -> set[str]:
    """Return current retry IDs, using checkpoint as source of truth.

    The review queue is a derived output and can become stale after a partial
    retry. The checkpoint is updated after each record, so it should decide
    what still needs another pass.
    """
    ids = _retry_ids_from_checkpoint()
    if ids:
        return ids
    return _retry_ids_from_review_queue()


def _expected_record_ids(md_files: list[Path]) -> set[str]:
    ids = set()
    for md_path in md_files:
        source_file = f"{md_path.stem}.pdf"
        for ballot_kind in PAGE_RANGES:
            ids.add(_record_id(source_file, ballot_kind))
    return ids


def _party_reference_prompt() -> str:
    return "\n".join(f"{num}: {name}" for num, name in sorted(PARTY_NAMES.items()))


def _constituency_party_reference_prompt() -> str:
    return "\n".join(
        f"{num}: {name or '-'}" for num, name in sorted(CONSTITUENCY_CANDIDATE_PARTIES.items())
    )


def _location_prompt() -> str:
    province = PROVINCE if PROVINCE is not None else "the province shown in the image"
    constituency = CONSTITUENCY_NUMBER if CONSTITUENCY_NUMBER is not None else "the constituency shown in the image"
    return (
        "Location context:\n"
        f"- Dataset scope: province={province}, constituency={constituency}.\n"
        "- Use this scope when OCR text is garbled or incomplete.\n"
        "- If the image clearly shows a different complete value, preserve it so validation can flag it."
    )


def _build_prompt(markdown: str, ballot_kind: str, source_file: str, page_range: str) -> str:
    form_code = "5_18_party" if ballot_kind == "party_list" else "5_18"
    target_label = "party-list" if ballot_kind == "party_list" else "constituency"
    party_rules = ""
    if ballot_kind == "party_list":
        party_rules = (
            "\nParty-list rules:\n"
            "- Extract party numbers 1-57 when visible.\n"
            "- Normalize party names against this official party-number reference:\n"
            f"{_party_reference_prompt()}\n"
            "- Keep row alignment strict: candidate_26 is party number 26, candidate_27 is party number 27, and so on.\n"
            "- If a party row appears blank or has 0 votes, still keep that row as 0; do NOT shift the next row upward.\n"
            "- Pay special attention around party numbers 26-34. Do not move Democrat/party-27 votes into party 26.\n"
        )
    elif CONSTITUENCY_CANDIDATE_PARTIES:
        party_rules = (
            "\nConstituency candidate-party rules:\n"
            "- Constituency candidate numbers are local candidate numbers, not national party-list numbers.\n"
            "- Use this official candidate-number to party mapping for the parties field:\n"
            f"{_constituency_party_reference_prompt()}\n"
        )

    return f"""You are extracting ONE Thai election tally form from a dual-form PDF.

Source PDF: {source_file}
Selected page range: {page_range}
Target ballot kind: {ballot_kind} ({target_label})

The selected pages already isolate the target form. Ignore any text that clearly belongs to another form.
Use the OCR markdown as a structural hint, but the attached image pages are the ground truth.

Return ONLY a JSON object with this schema:
{{
  "station_id": int,
  "constituency_number": int,
  "province": str,
  "good_ballots": int,
  "bad_ballots": int,
  "no_vote_ballots": int,
  "total_ballots": int,
  "form_code": str,
  "ballot_kind": str,
  "votes": {{ "candidate_<N>_votes": int, ... }},
  "parties": {{ "candidate_<N>_party": str, ... }},
  "total_votes_sum": int
}}

Hard requirements:
- Set "ballot_kind" exactly to "{ballot_kind}".
- Set "form_code" to "{form_code}" unless the page clearly shows a different official code.
- For constituency forms, extract candidate rows only.
- For party-list forms, extract party rows only.
- Extract the summary fields from the same selected form: good ballots, bad ballots, no-vote ballots, and total ballots used.
- good_ballots + bad_ballots + no_vote_ballots should equal total_ballots.
- Read candidate/party rows independently from the summary fields. Do NOT change row votes just to make a checksum pass.
- The handwritten "รวมคะแนนทั้งสิ้น" and "บัตรดี" summary can be misread, overwritten, or crossed out. Extract them as written, but the table rows remain the ground truth for per-candidate/per-party votes.
- Convert Thai digits to Arabic digits.
- Use 0 for missing/unreadable numeric values and "" for missing party names.
- Do not concatenate candidate/party numbers with vote counts.
- Read each row horizontally. Never take a vote from the row above/below, and never borrow the candidate/party number as a vote.
- If a digit is crossed out, ignored, or corrected, use the replacement value and the Thai words written after/near it. Do not use the crossed-out value.
- If digit and Thai-word vote disagree, use the handwritten Thai-word value when readable.
- If the table sum, "บัตรดี", and "รวมคะแนนทั้งสิ้น" do not agree, keep the independently read row votes and summary fields; validation will flag the mismatch.
{party_rules}
{_location_prompt()}

OCR markdown for selected pages:
---
{markdown}
---
"""


def _record_from_extraction(
    pipeline: OCRPipeline,
    source_file: str,
    ballot_kind: str,
    page_range: str,
    n_pages: int,
    extracted: dict | None,
    parent_kind: str = "",
) -> dict:
    record_id = _record_id(source_file, ballot_kind)
    if not extracted:
        return {
            "source_file": source_file,
            "polling_unit_id": pipeline._polling_unit_id(source_file),
            "ballot_record_id": record_id,
            "source_run": "markdown_split",
            "page_range": page_range,
            "parent_raw_row_ballot_kind": parent_kind,
            "form_type": "5_18_party" if ballot_kind == "party_list" else "5_18",
            "ballot_kind": ballot_kind,
            "ocr_confidence": 0.0,
            "ocr_success": False,
            "needs_review": True,
            "raw_text_preview": "Stage B failed during markdown split",
            "n_pages": n_pages,
        }

    extracted = pipeline._sanitize_thai_digits(extracted)
    extracted["ballot_kind"] = ballot_kind
    extracted = pipeline._apply_constituency_party_mapping(extracted, ballot_kind)
    quality = pipeline._quality_metrics(extracted)
    form_type = "5_18_party" if ballot_kind == "party_list" else "5_18"

    record = {
        "source_file": source_file,
        "polling_unit_id": pipeline._polling_unit_id(source_file),
        "ballot_record_id": record_id,
        "source_run": "markdown_split",
        "page_range": page_range,
        "parent_raw_row_ballot_kind": parent_kind,
        "form_type": form_type,
        "ballot_kind": ballot_kind,
        "form_code": extracted.get("form_code", ""),
        "station_id": extracted.get("station_id", 0),
        "constituency_number": extracted.get("constituency_number", 0),
        "province": extracted.get("province", ""),
        "good_ballots": extracted.get("good_ballots", 0),
        "bad_ballots": extracted.get("bad_ballots", 0),
        "no_vote_ballots": extracted.get("no_vote_ballots", 0),
        "total_ballots": extracted.get("total_ballots", 0),
        "total_votes_sum": extracted.get("total_votes_sum", 0),
        "ocr_confidence": quality["ocr_confidence"],
        "raw_text_preview": json.dumps(extracted.get("votes", {}), ensure_ascii=False)[:200],
        "n_pages": n_pages,
        "ocr_success": quality["ocr_confidence"] > 0,
        "needs_review": quality["needs_review"],
        "votes_sum": quality["votes_sum"],
        "vote_sum_match": quality["vote_sum_match"],
        "vote_sum_match_good_ballots": quality["vote_sum_match_good_ballots"],
        "vote_sum_match_total_votes": quality["vote_sum_match_total_votes"],
        "summary_votes_match": quality["summary_votes_match"],
        "total_votes_sum_match": quality["total_votes_sum_match"],
        "ballot_sum_match": quality["ballot_sum_match"],
        "has_summary_fields": quality["has_summary_fields"],
        "partial_page": quality["partial_page"],
    }
    votes = extracted.get("votes", {}) or {}
    if isinstance(votes, dict):
        record.update({k: v for k, v in votes.items() if isinstance(v, (int, float))})
    parties = extracted.get("parties", {}) or {}
    if isinstance(parties, dict):
        record.update({k: v for k, v in parties.items() if isinstance(v, str)})
    return pipeline._apply_manual_corrections(record)


def split_markdown_forms(
    limit: int | None = None,
    source_contains: str | None = None,
    reset_checkpoint: bool = False,
    retry_failed: bool = False,
) -> pd.DataFrame:
    if not GEMINI_API_KEY:
        raise ValueError("Please set GEMINI_API_KEY before running markdown split.")

    if reset_checkpoint and SPLIT_CHECKPOINT.exists():
        SPLIT_CHECKPOINT.unlink()
        logger.info(f"Removed split checkpoint {SPLIT_CHECKPOINT}")

    pipeline = OCRPipeline(engine="typhoon")
    gemini_client = pipeline.reader["gemini"]

    md_dir = OCR_RAW_DIR / "markdown" / "election"
    md_files = sorted(md_dir.glob("*.md"))
    if source_contains:
        md_files = [p for p in md_files if source_contains in p.name]
    if limit is not None:
        md_files = md_files[:limit]
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {md_dir}")
    expected_ids = _expected_record_ids(md_files)

    parent_kind_by_source: dict[str, str] = {}
    raw_path = OCR_RAW_DIR / "raw_all_forms.csv"
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        if {"source_file", "ballot_kind"}.issubset(raw_df.columns):
            parent_kind_by_source = dict(zip(raw_df["source_file"].astype(str), raw_df["ballot_kind"].astype(str)))

    from google.genai import types as genai_types

    checkpoint_df = _load_checkpoint()
    retry_ids: set[str] = set()
    if retry_failed:
        checkpoint_ids = set(checkpoint_df["ballot_record_id"].astype(str)) if not checkpoint_df.empty else set()
        missing_ids = expected_ids - checkpoint_ids
        retry_ids = _retry_ids_from_checkpoint() | missing_ids
        if not retry_ids:
            retry_ids = _retry_ids_from_review_queue()
        if retry_ids:
            logger.info(f"Retrying {len(retry_ids)} failed/review split records.")
            checkpoint_df = checkpoint_df[
                ~checkpoint_df["ballot_record_id"].astype(str).isin(retry_ids)
            ].copy()
            _save_checkpoint(checkpoint_df.to_dict("records"))
        else:
            logger.info("No failed/review split records found to retry.")

    records: list[dict] = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []
    seen_ids = set(checkpoint_df["ballot_record_id"].astype(str)) if not checkpoint_df.empty else set()
    if seen_ids:
        logger.info(f"Loaded {len(seen_ids)} split records from checkpoint.")

    for md_path in tqdm(md_files, desc="Split dual forms"):
        source_file = f"{md_path.stem}.pdf"
        markdown_all = md_path.read_text(encoding="utf-8")
        parent_kind = parent_kind_by_source.get(source_file, "")
        page_ranges = _infer_page_ranges(markdown_all)

        for ballot_kind in PAGE_RANGES:
            pages = page_ranges.get(ballot_kind) or PAGE_RANGES[ballot_kind]
            ballot_record_id = _record_id(source_file, ballot_kind)
            if retry_failed and retry_ids and ballot_record_id not in retry_ids:
                continue
            if ballot_record_id in seen_ids:
                continue
            page_range = _format_page_range(pages)
            selected_md = _select_markdown_pages(markdown_all, pages)
            images = _page_images(md_path.stem, pages)
            if not selected_md and not images:
                logger.warning(f"No selected pages for {source_file} {ballot_kind}; skipping")
                continue

            image_parts = []
            for img_path in images:
                try:
                    img_bytes = pipeline._load_image_bytes_for_gemini(img_path)
                    image_parts.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                    for crop_bytes in pipeline._load_focus_crop_bytes_for_gemini(img_path):
                        image_parts.append(genai_types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"))
                except Exception as exc:
                    logger.warning(f"Could not attach image {img_path.name}: {exc}")

            prompt = _build_prompt(selected_md, ballot_kind, source_file, page_range)
            extracted = pipeline._stage_b_call_with_verify(
                gemini_client,
                prompt,
                image_parts,
                model="gemini-flash-lite-latest",
            )
            record = _record_from_extraction(
                pipeline=pipeline,
                source_file=source_file,
                ballot_kind=ballot_kind,
                page_range=page_range,
                n_pages=len(images) or len(pages),
                extracted=extracted,
                parent_kind=parent_kind,
            )
            _write_checkpoint_record(record, records, seen_ids)
            time.sleep(2)

    return pd.DataFrame(records)


def write_outputs(df: pd.DataFrame, overwrite: bool = True) -> None:
    output_paths = [
        OCR_RAW_DIR / "raw_election_split.csv",
        OCR_RAW_DIR / "raw_all_forms_split.csv",
    ]
    for path in output_paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace split output.")
        df.to_csv(path, index=False)
        logger.info(f"Wrote {len(df)} split records to {path}")

    review = df[
        (df.get("needs_review", False) == True)
        | (df.get("vote_sum_match", True) == False)
        | (df.get("ballot_sum_match", True) == False)
        | (df.get("ocr_success", True) == False)
    ].copy()
    review_path = OCR_RAW_DIR / "split_review_queue.csv"
    review.to_csv(review_path, index=False)
    logger.info(f"Wrote {len(review)} split review rows to {review_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dual election-form markdown into two records per PDF.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N markdown files.")
    parser.add_argument("--source-contains", default=None, help="Process markdown files whose filename contains this text.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing *_split.csv outputs.")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete split checkpoint and start split extraction from scratch.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only split records that failed checksum or need review.")
    args = parser.parse_args()

    df = split_markdown_forms(
        limit=args.limit,
        source_contains=args.source_contains,
        reset_checkpoint=args.reset_checkpoint,
        retry_failed=args.retry_failed,
    )
    write_outputs(df, overwrite=not args.no_overwrite)

    print("\n=== Split OCR Summary ===")
    print(f"records: {len(df)}")
    if "ballot_kind" in df.columns:
        print(df["ballot_kind"].value_counts(dropna=False).to_string())
    if "needs_review" in df.columns:
        print("\nneeds_review:")
        print(df["needs_review"].value_counts(dropna=False).to_string())
    if "vote_sum_match" in df.columns:
        print("\nvote_sum_match:")
        print(df["vote_sum_match"].value_counts(dropna=False).to_string())
    if "ballot_sum_match" in df.columns:
        print("\nballot_sum_match:")
        print(df["ballot_sum_match"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
