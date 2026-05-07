"""
Phase 2: OCR pipeline — extract text and vote data from preprocessed images.
Supports multiple OCR engines: EasyOCR, Tesseract, PaddleOCR.

Usage:
    python 03_ocr/ocr_pipeline.py
"""
try:
    import easyocr
except ImportError:
    easyocr = None  # Not needed when using Gemini engine
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import cv2
import re
import sys
import io
import json
import base64
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Allow high-res scans

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    IMAGE_DIR, OCR_RAW_DIR, FORM_TYPES,
    OCR_ENGINE, OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD,
    REFERENCE_DIR,
    GEMINI_API_KEY, TYPHOON_API_KEY, PARTY_NAMES
)
from field_extractor import FieldExtractor


class OCRPipeline:
    """Main OCR pipeline for election document processing."""

    # Thai → Arabic digit translation table
    _THAI_DIGIT_TABLE = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

    def __init__(self, engine: str = OCR_ENGINE):
        self.engine = engine
        self.reader = self._init_engine()
        self.field_extractor = FieldExtractor()
        self.manual_corrections = self._load_manual_corrections()
        self.official_party_reference = self._build_party_reference_prompt()

    def _build_party_reference_prompt(self) -> str:
        """Compact party-number reference for Gemini party-name normalization."""
        return "\n".join(f"{num}: {name}" for num, name in sorted(PARTY_NAMES.items()))

    def _load_manual_corrections(self) -> dict[tuple[str, str], str]:
        """Load user-verified OCR corrections keyed by source_file + field."""
        path = REFERENCE_DIR / "ocr_corrections.csv"
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Could not load OCR corrections from {path}: {e}")
            return {}
        required = {"source_file", "field", "value"}
        if not required.issubset(df.columns):
            logger.warning(f"OCR corrections file missing required columns: {required}")
            return {}
        corrections = {}
        for _, row in df.iterrows():
            source = str(row["source_file"]).strip()
            field = str(row["field"]).strip()
            if source and field:
                corrections[(source, field)] = row["value"]
        logger.info(f"Loaded {len(corrections)} manual OCR corrections")
        return corrections

    def _apply_manual_corrections(self, record: dict) -> dict:
        """Apply user-verified field overrides after OCR extraction."""
        source_file = str(record.get("source_file", "")).strip()
        if not source_file or not self.manual_corrections:
            return record
        changed = []
        for (source, field), value in self.manual_corrections.items():
            if source != source_file:
                continue
            try:
                if str(field).endswith(("_votes", "_ballots")) or field in {
                    "station_id", "constituency_number", "total_ballots",
                    "good_ballots", "bad_ballots", "no_vote_ballots",
                }:
                    value = int(float(value))
            except (TypeError, ValueError):
                pass
            record[field] = value
            changed.append(field)
        if changed:
            record = self._refresh_record_quality(record)
            logger.info(f"Applied manual OCR corrections for {source_file}: {', '.join(changed)}")
        return record

    def _refresh_record_quality(self, record: dict) -> dict:
        """Refresh flat-record quality fields after manual corrections."""
        votes_sum = 0
        for key, value in record.items():
            if not re.match(r"^(candidate|party)_\d+_votes$", str(key)):
                continue
            try:
                votes_sum += int(float(value))
            except (TypeError, ValueError):
                pass
        record["votes_sum"] = votes_sum
        try:
            good = int(float(record.get("good_ballots") or 0))
        except (TypeError, ValueError):
            good = 0
        try:
            bad = int(float(record.get("bad_ballots") or 0))
        except (TypeError, ValueError):
            bad = 0
        try:
            no_vote = int(float(record.get("no_vote_ballots") or 0))
        except (TypeError, ValueError):
            no_vote = 0
        try:
            total = int(float(record.get("total_ballots") or 0))
        except (TypeError, ValueError):
            total = 0

        record["vote_sum_match"] = bool(good > 0 and votes_sum == good)
        record["ballot_sum_match"] = bool(total > 0 and (good + bad + no_vote) == total)
        record["has_summary_fields"] = bool(any(v > 0 for v in [good, bad, no_vote, total]))
        record["partial_page"] = bool(not record.get("station_id") or not record["has_summary_fields"])
        record["needs_review"] = bool(record["partial_page"] or not (record["vote_sum_match"] and record["ballot_sum_match"]))
        return record

    def _init_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine == "easyocr":
            logger.info("Initializing EasyOCR with languages: th, en")
            return easyocr.Reader(
                OCR_LANGUAGES,
                gpu=True  # Set False if no GPU
            )
        elif self.engine == "tesseract":
            import pytesseract
            return pytesseract
        elif self.engine == "gemini":
            from google import genai
            if not GEMINI_API_KEY:
                logger.error("Please set GEMINI_API_KEY in your .env file or environment.")
                raise ValueError("No API key provided")
            client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Initialized Google Gemini (google-genai SDK)")
            return client
        elif self.engine == "typhoon":
            from openai import OpenAI
            if not TYPHOON_API_KEY:
                logger.error("Please set TYPHOON_API_KEY in your .env file or environment.")
                raise ValueError("No Typhoon API key provided")
            if not GEMINI_API_KEY:
                logger.error("Stage-B extractor needs GEMINI_API_KEY in your .env file or environment.")
                raise ValueError("No Gemini API key for Stage B")
            typhoon_client = OpenAI(
                api_key=TYPHOON_API_KEY,
                base_url="https://api.opentyphoon.ai/v1"
            )
            from google import genai
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Initialized Typhoon OCR 1.5 (Stage A) + Gemini text (Stage B)")
            return {"typhoon": typhoon_client, "gemini": gemini_client}
        elif self.engine == "easyocr_gemini":
            # Stage A: local EasyOCR (unlimited, free, runs on CPU/GPU)
            # Stage B: Gemini text (only ~1 small text call per polling unit)
            if easyocr is None:
                raise ValueError("easyocr not installed; pip install easyocr")
            if not GEMINI_API_KEY:
                raise ValueError("Stage-B needs GEMINI_API_KEY in config.py")
            try:
                reader = easyocr.Reader(OCR_LANGUAGES, gpu=True)
            except Exception:
                logger.warning("GPU init failed; falling back to CPU EasyOCR.")
                reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
            from google import genai
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Initialized EasyOCR (local Stage A) + Gemini text (Stage B)")
            return {"easyocr": reader, "gemini": gemini_client}
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")

    def process_all_forms(self, form_types=None):
        """Run OCR on all form types and produce raw CSV outputs."""
        all_results = []
        
        target_forms = form_types if form_types is not None else FORM_TYPES

        for form_type in target_forms:
            img_dir = IMAGE_DIR / form_type
            if not img_dir.exists():
                logger.warning(f"No images for {form_type}, skipping")
                continue

            logger.info(f"Processing form type: {form_type}")
            form_results = self._process_form_type(form_type, img_dir)
            all_results.extend(form_results)

            # Save per-form-type CSV
            if form_results:
                df = pd.DataFrame(form_results)
                csv_path = OCR_RAW_DIR / f"raw_{form_type}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"Saved {len(df)} records to {csv_path}")

        # Save combined raw output
        if all_results:
            combined_df = pd.DataFrame(all_results)
            if form_types == ["sample"]:
                combined_path = OCR_RAW_DIR / "raw_sample_all.csv"
            else:
                combined_path = OCR_RAW_DIR / "raw_all_forms.csv"
            combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
            logger.info(f"Combined output: {len(combined_df)} total records")

        return all_results

    def _process_form_type(self, form_type: str, img_dir: Path) -> list[dict]:
        """Process all images for a single form type with auto-saving.

        For the typhoon two-stage engine, images sharing the same PDF stem
        (foo_page1.png, foo_page2.png, ...) are grouped and merged into one
        record per polling unit. Other engines stay one-row-per-image.
        """
        results = []
        images = sorted(img_dir.glob("*.png"))

        checkpoint_file = OCR_RAW_DIR / f"raw_{form_type}_checkpoint.csv"

        processed_keys = set()
        if checkpoint_file.exists():
            try:
                existing_df = pd.read_csv(checkpoint_file)
                if 'source_file' in existing_df.columns:
                    # Only skip files that were successfully processed
                    if 'ocr_success' in existing_df.columns:
                        success_df = existing_df[existing_df['ocr_success'] == True]
                    else:
                        success_df = existing_df
                    if 'ocr_confidence' in success_df.columns:
                        success_df = success_df[
                            pd.to_numeric(success_df['ocr_confidence'], errors='coerce').fillna(0) > 0
                        ]
                    if 'needs_review' in success_df.columns:
                        success_df = success_df[success_df['needs_review'] != True]
                    processed_keys = set(success_df['source_file'].dropna().tolist())
                    
                    # Keep all results so we don't lose data, we'll overwrite failed ones by appending
                    # and eventually deduplicating if necessary, or just rely on the new append
                    results = existing_df.to_dict('records')
                    # Filter out failed runs from results so they get replaced by the new successful run
                    results = [r for r in results if r.get('ocr_success', True)]
                    results = [r for r in results if r.get('needs_review', False) != True]
                    
                    logger.info(f"Loaded {len(processed_keys)} successfully processed entries from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        # Group images by PDF stem (strip trailing "_pageN")
        if self.engine in ("typhoon", "easyocr_gemini"):
            groups: dict[str, list[Path]] = {}
            for img in images:
                key = re.sub(r"_page\d+$", "", img.stem)
                groups.setdefault(key, []).append(img)
            iterable = sorted(groups.items())

            for key, pages in tqdm(iterable, desc=f"OCR {form_type}"):
                group_id = f"{key}.pdf"  # Stable key for checkpoint
                if group_id in processed_keys:
                    continue
                try:
                    pages_sorted = sorted(pages, key=lambda p: int(
                        (re.search(r"_page(\d+)$", p.stem) or re.match(r"(\d+)", "1")).group(1)
                    ) if re.search(r"_page\d+$", p.stem) else 1)
                    record = self._process_image_group(pages_sorted, form_type, group_id)
                    if record:
                        results.append(record)
                        pd.DataFrame(results).to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
                except Exception as e:
                    logger.error(f"OCR failed for group {group_id}: {e}")
                    results.append({
                        "source_file": group_id,
                        "form_type": form_type,
                        "error": str(e),
                        "ocr_success": False,
                    })
                    pd.DataFrame(results).to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
            return results

        # Non-typhoon engines: per-image
        for img_path in tqdm(images, desc=f"OCR {form_type}"):
            if img_path.name in processed_keys:
                continue
            try:
                record = self._process_single_image(img_path, form_type)
                if record:
                    results.append(record)
                    pd.DataFrame(results).to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
            except Exception as e:
                logger.error(f"OCR failed for {img_path.name}: {e}")
                results.append({
                    "source_file": img_path.name,
                    "form_type": form_type,
                    "error": str(e),
                    "ocr_success": False,
                })
                pd.DataFrame(results).to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
        return results

    def _process_image_group(self, pages: list[Path], form_type: str, group_id: str) -> dict:
        """Run Typhoon Stage A on each page, concatenate markdown, then ONE Stage B call.
        Produces a single merged record per PDF/polling unit.
        """
        gemini_client = self.reader["gemini"]
        use_easyocr = self.engine == "easyocr_gemini"
        typhoon_client = None if use_easyocr else self.reader["typhoon"]
        easyocr_reader = self.reader["easyocr"] if use_easyocr else None

        # Auto-detect actual form type from filename when running in "sample" mode
        # so the Stage B prompt says e.g. "Form election" instead of "Form sample"
        prompt_form_type = form_type
        if form_type == "sample":
            fname = group_id.lower()
            if "5_18" in fname or "5_17" in fname or "5_16" in fname:
                prompt_form_type = "election"
            else:
                prompt_form_type = "election"  # default to election for Thai election forms

        merged_md_parts = []
        for idx, img_path in enumerate(pages, 1):
            md = ""  # Reset for each page

            if use_easyocr:
                # Local Stage A — EasyOCR. No API, no rate limit.
                img_array = np.fromfile(str(img_path), dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_cv is None:
                    logger.error(f"Cannot read {img_path.name}; skipping page")
                    merged_md_parts.append(f"\n\n--- Page {idx} ({img_path.name}) ---\n[unreadable]")
                    continue
                try:
                    raw = easyocr_reader.readtext(
                        img_cv, detail=1, paragraph=False,
                        contrast_ths=0.2, adjust_contrast=0.6,
                        text_threshold=0.6, low_text=0.35, mag_ratio=1.5,
                    )
                    # Convert to a simple text dump (keeps reading order top→bottom)
                    raw_sorted = sorted(raw, key=lambda r: (round(r[0][0][1] / 30), r[0][0][0]))
                    md = "\n".join(t for _, t, conf in raw_sorted if conf > 0.3)
                except Exception as e:
                    logger.error(f"EasyOCR Stage A error on {img_path.name}: {e}")
            else:
                # Typhoon vision API
                pil_img = Image.open(str(img_path))
                max_dim = 4096
                if max(pil_img.size) > max_dim:
                    ratio = max_dim / max(pil_img.size)
                    pil_img = pil_img.resize(
                        (int(pil_img.width * ratio), int(pil_img.height * ratio)),
                        Image.LANCZOS,
                    )
                # Convert to RGB (JPEG doesn't support alpha) and compress
                if pil_img.mode in ("RGBA", "LA", "P"):
                    pil_img = pil_img.convert("RGB")
                buf = io.BytesIO(); pil_img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                logger.debug(f"Image {img_path.name}: {len(buf.getvalue())/1024:.0f} KB after JPEG compression")

                stage_a_prompt = (
                    "Transcribe this Thai election form faithfully. Preserve the table "
                    "structure (candidate / party number → vote count). "
                    "IMPORTANT: Convert ALL Thai numerals (๐๑๒๓๔๕๖๗๘๙) to Arabic digits "
                    "(0123456789) throughout the entire transcription. Output raw text only."
                )
                for attempt in range(5):
                    try:
                        resp = typhoon_client.chat.completions.create(
                            model="typhoon-ocr",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": stage_a_prompt},
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    }},
                                ],
                            }],
                            temperature=0,
                            max_tokens=4096,
                        )
                        md = resp.choices[0].message.content or ""
                        break
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "rate" in err.lower():
                            wait = min(30 * (2 ** attempt), 300)
                            logger.warning(f"Typhoon rate limit (attempt {attempt+1}/5). Waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            logger.error(f"Stage A error on {img_path.name}: {e}")
                            break

            # Sanity check: Typhoon sometimes echoes its own system prompt when it
            # can't process the image (e.g. file too large). Detect and discard.
            if "Extract all text from the image." in md or "Only return the clean Markdown." in md:
                logger.warning(f"Stage A echoed its own prompt for {img_path.name} — image may be too large or corrupt. Skipping.")
                md = ""

            merged_md_parts.append(f"\n\n--- Page {idx} ({img_path.name}) ---\n{md}")
            if not use_easyocr:
                time.sleep(1)  # Pace API only

        merged_md = "".join(merged_md_parts).strip()

        # Persist for debugging
        md_dir = OCR_RAW_DIR / "markdown" / form_type
        md_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / (Path(group_id).stem + ".md")).write_text(merged_md, encoding="utf-8")

        if not merged_md:
            return {
                "source_file": group_id,
                "form_type": form_type,
                "ocr_confidence": 0.0,
                "raw_text_preview": "Stage A produced no markdown",
                "ocr_success": False,
            }

        # Pre-process: convert ALL Thai digits to Arabic in the OCR text
        # so that Stage B (Gemini) never sees Thai numerals.
        # Typhoon Stage A transcribes party numbers as Thai digits (๑,๒,๓...)
        # while vote counts are already Arabic. Converting everything to Arabic
        # prevents Gemini from accidentally merging party numbers into vote counts.
        merged_md = merged_md.translate(self._THAI_DIGIT_TABLE)

        stage_b_prompt = f"""You receive (a) the OCR transcription markdown of a Thai election form (Form {prompt_form_type})
spanning {len(pages)} page(s) of the SAME polling unit, AND (b) the original page image(s) attached after this prompt.

Use the markdown as a structural HINT, but the IMAGE is the GROUND TRUTH. The Typhoon OCR that produced
the markdown is known to:
  • fabricate the Thai-word column from the digit it read (so the words "match the digits" even when the
    handwritten Thai word is different — DO NOT trust matching markdown words alone)
  • miss strike-throughs / corrections (a digit that has been crossed out and re-written should be ignored)
  • duplicate numbers in ballot-count lines
Whenever you have any doubt, look at the image and read what is actually handwritten. If the image shows
a digit struck through with a different value below/beside it, use the corrected value. Use the Thai
words written by hand (not the markdown's regenerated words) as your real cross-check on the digits.

Merge all pages into a single record. Return ONLY a JSON object — no markdown fences, no commentary.
Schema:
{{
  "station_id": int,
  "constituency_number": int,
  "province": str,
  "good_ballots": int,
  "bad_ballots": int,
  "no_vote_ballots": int,
  "total_ballots": int,
  "votes": {{ "candidate_<N>_votes": int, ... }},
  "parties": {{ "candidate_<N>_party": "<party name from สังกัดพรรคการเมือง column>", ... }},
  "total_votes_sum": int   // value handwritten on the "รวมคะแนนทั้งสิ้น" row (or 0 if absent)
}}
Rules:
- Combine candidate vote counts across all pages — each candidate appears once.
- For each candidate row, also extract the party affiliation (สังกัดพรรคการเมือง column) into "parties".
- Use empty string "" for empty/blank candidate rows.
- When the row is a party-list row, normalize party names against this official party-number reference:
{self.official_party_reference}
- If a value appears on multiple pages, prefer the most-complete number.
- Convert Thai digits ๐-๙ to Arabic.
- Use 0 for missing/unreadable numeric values.

CRITICAL VOTE-COUNT RULES (the form is designed so that votes are written TWICE — once as digits and once as Thai words — for cross-validation):
- Each candidate row contains: (1) candidate/party number, (2) name, (3) vote count in digits, (4) vote count in Thai words within parentheses, e.g. "(เจ็ดสิบหก)" = 76.
- WHENEVER the Thai-word form is readable, USE IT AS THE GROUND TRUTH and ignore the digits column. Examples:
    "76 (เจ็ดสิบหก)"        → 76
    "5,150 (ห้า)"           → 5   ← digits were misread; ห้า means 5
    "113 (หนึ่งร้อยยี่สิบสาม)" → 123 ← digits were misread; words say 123
- Common Thai number words: ศูนย์=0 หนึ่ง=1 สอง=2 สาม=3 สี่=4 ห้า=5 หก=6 เจ็ด=7 แปด=8 เก้า=9 สิบ=10 ยี่สิบ=20 ร้อย=100 พัน=1000 หมื่น=10000.
- HANDWRITING DISAMBIGUATION (these Thai numerals are easily confused):
    "สาม" (3) — three short parallel horizontal strokes; no hook
    "สี่"  (4) — has a distinctive curl/hook at the top of the leading consonant
    "หก"  (6) — H-shape with a tail
    "เก้า" (9) — has a leading "เ" followed by ก้า; tone mark above
  Zoom in mentally on the hooked vs. straight strokes before deciding.
- If the Thai-word column is empty/illegible, fall back to the digits.
- If the row's vote cell appears SPLIT into TWO sub-cells (e.g. "| 2 | ๘๐๐ |" or "| 76 | ๒๐๐ |"), DO NOT concatenate them. The right-hand value is usually a form serial/page-number printed in Thai numerals, NOT part of the vote. Use only the LEFT digit cell (or the Thai-word column when available).
- A single candidate's vote can never exceed the polling station's good_ballots. If your extracted number is larger than good_ballots, you have parsed wrong — re-check the Thai-word column.
- The markdown may be wrong even when the digit and Thai word agree with each other. Always re-read the attached IMAGE/CROP for every handwritten vote.
- Common severe mistakes to avoid: 22 misread as 52/62, 64 misread as 34, and 11 misread as 1. Count separate vertical strokes carefully.
- If the image/crop disagrees with the markdown, the image/crop wins.
- In party-list tables, the first column is the party number. NEVER borrow digits from it. A row with party number 27 and vote 11 is 11, NOT 17 or 271.
- Distinguish Thai words carefully: "สิบเอ็ด" = 11, "สิบเจ็ด" = 17. If the word has เอ็ด/อ, output 11; do not invent เจ็ด/จ.
- The first column (party/candidate หมายเลข, e.g. ๑, ๒, ๓) must NEVER be concatenated with the vote count. If party number is ๙ and vote count is 76, the result is 76, NOT 976.

CRITICAL BALLOT-COUNT RULES:
- Lines like "บัตรเลือกตั้งที่ใช้ -> 237 จำนวน 263 ใบ" sometimes contain TWO numbers because OCR caught both a printed default and the handwritten total.
- Choose the HANDWRITTEN value (typically appearing AFTER "จำนวน" and BEFORE the unit "ใบ"/"บัตร"/"คน").
- good_ballots + bad_ballots + no_vote_ballots should equal total_ballots (บัตรเลือกตั้งที่ใช้). Use this as a sanity check.
- For "บัตรดี" pick the number that, combined with bad_ballots and no_vote_ballots, is closest to total_ballots.

PROVINCE: This dataset is exclusively from "แพร่" (Phrae) province, constituency 3. If the OCR text has a garbled/short province like "เพ", "เพิ่ง", "เพอร์", "เพชร" — output "แพร่". Only output a different province if the markdown clearly shows a complete different province name.

OCR transcription:
---
{merged_md}
---
"""
        # Build multi-modal contents: each page image as a Part.
        # Letting Gemini SEE the original handwriting lets it override Typhoon
        # markdown errors (strikethroughs, fabricated Thai-word transcriptions).
        from google.genai import types as genai_types
        image_parts = []
        for img_path in pages:
            try:
                img_bytes = self._load_image_bytes_for_gemini(img_path)
                if img_bytes:
                    image_parts.append(
                        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    )
                for crop_bytes in self._load_focus_crop_bytes_for_gemini(img_path):
                    image_parts.append(
                        genai_types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg")
                    )
            except Exception as e:
                logger.warning(f"Could not attach image {img_path.name} to Stage B: {e}")

        # Single helper handles thinking config + sum-check verification + retry
        extracted = self._stage_b_call_with_verify(
            gemini_client, stage_b_prompt, image_parts,
            model="gemini-flash-lite-latest",
        )

        if not extracted:
            return {
                "source_file": group_id,
                "form_type": form_type,
                "ocr_confidence": 0.5,
                "raw_text_preview": "Stage B failed; markdown saved",
                "ocr_success": False,
            }

        # Post-process: sanitize any remaining Thai digits in numeric values
        extracted = self._sanitize_thai_digits(extracted)
        quality = self._quality_metrics(extracted)

        record = {
            "source_file": group_id,
            "form_type": form_type,
            "station_id": extracted.get("station_id", 0),
            "constituency_number": extracted.get("constituency_number", 0),
            "province": extracted.get("province", ""),
            "good_ballots": extracted.get("good_ballots", 0),
            "bad_ballots": extracted.get("bad_ballots", 0),
            "no_vote_ballots": extracted.get("no_vote_ballots", 0),
            "total_ballots": extracted.get("total_ballots", 0),
            "ocr_confidence": quality["ocr_confidence"],
            "raw_text_preview": json.dumps(extracted.get("votes", {}), ensure_ascii=False)[:200],
            "n_pages": len(pages),
            "ocr_success": quality["ocr_confidence"] > 0,
            "needs_review": quality["needs_review"],
            "votes_sum": quality["votes_sum"],
            "vote_sum_match": quality["vote_sum_match"],
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
        record = self._apply_manual_corrections(record)
        time.sleep(2)
        return record

    @staticmethod
    def _votes_sum(extracted: dict) -> int:
        v = extracted.get("votes") or {}
        s = 0
        for val in v.values():
            try:
                s += int(val)
            except (TypeError, ValueError):
                pass
        return s

    def _quality_metrics(self, extracted: dict) -> dict:
        """Compute deterministic quality flags for an extracted record."""
        votes_sum = self._votes_sum(extracted)
        try:
            good = int(extracted.get("good_ballots") or 0)
        except (TypeError, ValueError):
            good = 0
        try:
            bad = int(extracted.get("bad_ballots") or 0)
        except (TypeError, ValueError):
            bad = 0
        try:
            no_vote = int(extracted.get("no_vote_ballots") or 0)
        except (TypeError, ValueError):
            no_vote = 0
        try:
            total = int(extracted.get("total_ballots") or 0)
        except (TypeError, ValueError):
            total = 0
        try:
            total_votes_row = int(extracted.get("total_votes_sum") or 0)
        except (TypeError, ValueError):
            total_votes_row = 0

        vote_target = good if good > 0 else total_votes_row
        vote_sum_match = vote_target > 0 and votes_sum == vote_target
        ballot_sum_match = total > 0 and (good + bad + no_vote) == total
        has_station = bool(extracted.get("station_id"))
        has_votes = votes_sum > 0
        has_summary_fields = any(v > 0 for v in [good, bad, no_vote, total, total_votes_row])
        partial_page = not has_station or not has_summary_fields

        score = 0.35
        score += 0.25 if vote_sum_match else 0
        score += 0.20 if ballot_sum_match else 0
        score += 0.10 if has_station else 0
        score += 0.10 if has_votes else 0

        return {
            "votes_sum": votes_sum,
            "vote_sum_match": vote_sum_match,
            "ballot_sum_match": ballot_sum_match,
            "has_summary_fields": has_summary_fields,
            "partial_page": partial_page,
            "needs_review": partial_page or not (vote_sum_match and ballot_sum_match),
            "ocr_confidence": round(min(score, 0.99), 2),
        }

    @staticmethod
    def _stage_b_thinking_config():
        """Return GenerateContentConfig that enables thinking on Flash Lite.

        Flash Lite supports thinking but is OFF by default. A small thinking
        budget (a few thousand tokens) lets the model verify the sum-check
        and re-read ambiguous handwritten digits before answering.
        """
        from google.genai import types as genai_types
        try:
            return genai_types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=4096),
            )
        except Exception:
            # Older SDK without thinking_config — fall back to plain config
            try:
                return genai_types.GenerateContentConfig(temperature=0.0)
            except Exception:
                return None

    def _stage_b_call_with_verify(self, gemini_client, base_prompt: str,
                                   image_parts: list, model: str = "gemini-flash-lite-latest"):
        """Call Stage B and, if sum-of-votes != good_ballots, retry once with
        explicit feedback pointing out the discrepancy. Returns the extracted
        dict (or None on permanent failure).
        """
        import json as _json
        cfg = self._stage_b_thinking_config()

        def _call(prompt_text):
            contents = [prompt_text, *image_parts]
            for attempt in range(5):
                try:
                    kwargs = {"model": model, "contents": contents}
                    if cfg is not None:
                        kwargs["config"] = cfg
                    resp = gemini_client.models.generate_content(**kwargs)
                    txt = (resp.text or "").strip().replace("```json", "").replace("```", "").strip()
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    if not m:
                        raise ValueError(f"No JSON in Stage B output: {txt[:160]}")
                    return _json.loads(m.group(0))
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "Service Unavailable" in err:
                        wait = min(30 * (2 ** attempt), 300)
                        logger.warning(f"Gemini busy (Stage B, attempt {attempt+1}/5). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Stage B error: {e}")
                        return None
            return None

        # First pass
        extracted = _call(base_prompt)
        if not extracted:
            return None

        # Sum-check verification — use good_ballots when available, otherwise
        # fall back to the "รวมคะแนนทั้งสิ้น" row that Stage B is asked to extract.
        votes_sum = self._votes_sum(extracted)
        try:
            good = int(extracted.get("good_ballots") or 0)
        except (TypeError, ValueError):
            good = 0
        try:
            total_votes_row = int(extracted.get("total_votes_sum") or 0)
        except (TypeError, ValueError):
            total_votes_row = 0
        target = good if good > 0 else total_votes_row
        if target > 0 and votes_sum != target:
            # Reuse the existing branch logic; rename `good` for the feedback msg
            good = target
            diff = votes_sum - good
            votes_dict = extracted.get("votes") or {}
            current = ", ".join(f"{k}={v}" for k, v in votes_dict.items())
            feedback = (
                f"\n\n=== VERIFICATION FAILED — RE-DO THIS EXTRACTION ===\n"
                f"Your previous answer was: {current}\n"
                f"Good ballots = {good}, but the sum of your votes = {votes_sum} (off by {diff:+d}).\n"
                f"This means at least one candidate row was misread. Look at the IMAGE again.\n"
                f"For each candidate row, read the Thai-word column carefully (สอง=2, ห้า=5, "
                f"หก=6, สาม=3, เก้า=9, สี่=4) and use it to override the digits column.\n"
                f"Also re-read the รวมคะแนนทั้งสิ้น row at the bottom — it must equal good_ballots.\n"
                f"Pay extra attention to digits that look like 2/5, 6/3, 6/0, 4/9.\n"
                f"Now produce the corrected JSON — same schema, no commentary."
            )
            logger.warning(
                f"Sum-check failed (sum={votes_sum}, good={good}, off by {diff:+d}); retrying Stage B with feedback…"
            )
            second = _call(base_prompt + feedback)
            if second:
                new_sum = self._votes_sum(second)
                if abs(new_sum - good) < abs(votes_sum - good):
                    extracted = second
                    if new_sum == good:
                        logger.info(f"Stage B retry succeeded — sum now matches good_ballots ({good}).")
                    else:
                        logger.info(f"Stage B retry improved (sum {votes_sum}→{new_sum}, target {good}).")
        return extracted

    @staticmethod
    def _load_image_bytes_for_gemini(img_path: Path, max_dim: int = 3000,
                                     jpeg_quality: int = 95) -> bytes:
        """Load an image and return JPEG bytes suitable for Gemini Stage B.

        Resizes large images so total payload stays small enough for the
        Gemini quota (each image ~200-400 KB at these defaults). Converts
        any non-RGB modes to RGB before JPEG encoding.
        """
        pil_img = Image.open(str(img_path))
        if max(pil_img.size) > max_dim:
            ratio = max_dim / max(pil_img.size)
            pil_img = pil_img.resize(
                (int(pil_img.width * ratio), int(pil_img.height * ratio)),
                Image.LANCZOS,
            )
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()

    @staticmethod
    def _load_focus_crop_bytes_for_gemini(img_path: Path, jpeg_quality: int = 95) -> list[bytes]:
        """Return high-resolution crops of vote areas for Stage B verification."""
        pil_img = Image.open(str(img_path))
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        w, h = pil_img.size
        boxes = [
            (0, int(h * 0.52), w, h),                 # lower half: candidate table + totals
            (int(w * 0.55), int(h * 0.52), w, h),     # right side: vote digits + Thai words
        ]
        crops = []
        for box in boxes:
            crop = pil_img.crop(box)
            if crop.width < 80 or crop.height < 80:
                continue
            max_dim = 3000
            if max(crop.size) > max_dim:
                ratio = max_dim / max(crop.size)
                crop = crop.resize((int(crop.width * ratio), int(crop.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            crops.append(buf.getvalue())
        return crops

    @staticmethod
    def _sanitize_thai_digits(data: dict) -> dict:
        """Recursively convert any remaining Thai digits ๐-๙ to Arabic 0-9
        in numeric values returned by Stage B.  Also re-parse stringified
        numbers that may still contain Thai digits."""
        table = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

        def _clean(v):
            if isinstance(v, str):
                return v.translate(table)
            if isinstance(v, dict):
                return {k: _clean(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_clean(item) for item in v]
            return v

        cleaned = _clean(data)
        # Re-parse vote/ballot values that might now be pure digit strings
        for key in list(cleaned.keys()):
            val = cleaned[key]
            if isinstance(val, str) and key != "province":
                stripped = val.strip()
                if stripped.isdigit():
                    cleaned[key] = int(stripped)
        # Also handle nested "votes" and "parties" dicts
        for sub_key in ("votes", "parties"):
            sub = cleaned.get(sub_key)
            if isinstance(sub, dict):
                for k in list(sub.keys()):
                    v = sub[k]
                    if isinstance(v, str) and v.strip().isdigit():
                        sub[k] = int(v.strip())
        return cleaned

    def _process_single_image(self, img_path: Path, form_type: str) -> dict:
        """
        Run OCR on a single image and extract structured fields.

        Returns dict with:
            - source_file, form_type
            - station_id, constituency
            - party votes (party_1_votes, party_2_votes, ...)
            - total_votes, good_ballots, bad_ballots, no_vote_ballots
            - ocr_confidence (average)
        """
        # Use np.fromfile to handle Thai/unicode characters in Windows paths
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Run OCR
        if self.engine == "gemini":
            pil_img = Image.open(str(img_path))

            # Resize large images to avoid API limits & speed up
            max_dim = 4096
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                logger.debug(f"Resized image to {new_size}")

            prompt = f"""Analyze this Thailand election vote tally document (Form {form_type}).
Extract all the candidate/party vote counts and summary data.
Return ONLY a valid JSON object with no markdown formatting.
Keys to extract:
- "station_id": int (หมายเลขหน่วยเลือกตั้ง)
- "constituency_number": int (เขตเลือกตั้งที่)
- "province": string (จังหวัด)
- "good_ballots": int (บัตรดี)
- "bad_ballots": int (บัตรเสีย)
- "no_vote_ballots": int (ไม่ประสงค์ลงคะแนน)
- "total_ballots": int (บัตรทั้งหมด)
- "votes": dict mapping candidate numbers to vote counts
  e.g. {{"candidate_1_votes": 20, "candidate_2_votes": 100}}
- "parties": dict mapping candidate numbers to their party affiliation (สังกัดพรรคการเมือง)
  read from the column next to the candidate name. Example:
  {{"candidate_1_party": "ภูมิใจไทย", "candidate_2_party": "เพื่อไทย"}}
  Use empty string "" if a candidate row has no party listed (empty row).
"""
            # Use the new google.genai SDK — with retry for rate limits
            import time
            record = None
            for attempt in range(5):
                try:
                    response = self.reader.models.generate_content(
                        model="gemini-flash-lite-latest",
                        contents=[prompt, pil_img]
                    )
                    json_str = response.text.replace("```json", "").replace("```", "").strip()
                    extracted = json.loads(json_str)

                    record = {
                        "station_id": extracted.get("station_id", 0),
                        "constituency_number": extracted.get("constituency_number", 0),
                        "province": extracted.get("province", ""),
                        "good_ballots": extracted.get("good_ballots", 0),
                        "bad_ballots": extracted.get("bad_ballots", 0),
                        "no_vote_ballots": extracted.get("no_vote_ballots", 0),
                        "total_ballots": extracted.get("total_ballots", 0),
                        "ocr_confidence": 0.99,
                        "raw_text_preview": json.dumps(extracted.get("votes", {}), ensure_ascii=False)[:200],
                    }
                    if "votes" in extracted:
                        record.update(extracted["votes"])
                    if isinstance(extracted.get("parties"), dict):
                        record.update({k: v for k, v in extracted["parties"].items() if isinstance(v, str)})
                    break  # Success — exit retry loop

                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "Service Unavailable" in err_str:
                        wait = min(60 * (2 ** attempt), 300)  # 60s, 120s, 240s, 300s max
                        logger.warning(f"Gemini busy/limited (attempt {attempt+1}/5). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Gemini error: {e}")
                        record = {"ocr_confidence": 0.0, "raw_text_preview": str(e)[:200]}
                        break

            if record is None:
                record = {"ocr_confidence": 0.0, "raw_text_preview": "All retry attempts exhausted"}

            # Rate limit: wait between successful requests to stay within free tier
            time.sleep(5)

        elif self.engine == "typhoon":
            pil_img = Image.open(str(img_path))

            # Resize for optimal Typhoon performance (election forms need high res)
            max_dim = 4096
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                logger.debug(f"Resized image to {new_size}")

            # Convert to RGB (JPEG doesn't support alpha) and compress
            if pil_img.mode in ("RGBA", "LA", "P"):
                pil_img = pil_img.convert("RGB")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.debug(f"Image {img_path.name}: {len(buffer.getvalue())/1024:.0f} KB after JPEG compression")

            typhoon_client = self.reader["typhoon"]
            gemini_client = self.reader["gemini"]

            # ===== STAGE A: typhoon-ocr → layout-preserving markdown =====
            stage_a_prompt = (
                "Transcribe this Thai election form faithfully. "
                "Preserve the table structure (candidate number / party number → vote count rows). "
                "Output the raw text and tables only — no commentary."
            )
            markdown = None
            for attempt in range(5):
                try:
                    resp_a = typhoon_client.chat.completions.create(
                        model="typhoon-ocr",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": stage_a_prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}"
                                }}
                            ]
                        }],
                        temperature=0,
                        max_tokens=4096,
                    )
                    markdown = resp_a.choices[0].message.content or ""
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate" in err.lower():
                        wait = min(30 * (2 ** attempt), 300)
                        logger.warning(f"Typhoon rate limited (A, attempt {attempt+1}/5). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Typhoon Stage A error: {e}")
                        return {
                            "source_file": img_path.name,
                            "form_type": form_type,
                            "ocr_confidence": 0.0,
                            "raw_text_preview": f"Stage A failed: {str(e)[:160]}",
                            "ocr_success": False,
                        }

            # Sanity check: Typhoon sometimes echoes its own system prompt when it
            # can't process the image (e.g. file too large). Detect and discard.
            if "Extract all text from the image." in markdown or "Only return the clean Markdown." in markdown:
                logger.warning(f"Stage A echoed its own prompt for {img_path.name} — image may be too large or corrupt. Skipping.")
                return {
                    "source_file": img_path.name,
                    "form_type": form_type,
                    "ocr_confidence": 0.0,
                    "raw_text_preview": "Stage A hallucinated its system prompt (image too large)",
                    "ocr_success": False,
                }

            if not markdown:
                return {
                    "source_file": img_path.name,
                    "form_type": form_type,
                    "ocr_confidence": 0.0,
                    "raw_text_preview": "Stage A returned empty",
                    "ocr_success": False,
                }

            # Persist markdown for debugging / re-extraction
            md_dir = OCR_RAW_DIR / "markdown" / form_type
            md_dir.mkdir(parents=True, exist_ok=True)
            md_path = md_dir / (img_path.stem + ".md")
            md_path.write_text(markdown, encoding="utf-8")

            # Pre-process: convert Thai digits to Arabic before Stage B
            markdown = markdown.translate(self._THAI_DIGIT_TABLE)

            # ===== STAGE B: Gemini multi-modal (markdown + image) → strict JSON =====
            stage_b_prompt = f"""You are extracting structured data from a Thai election form (Form {form_type}).
You receive (a) the OCR transcription markdown produced by Typhoon, AND (b) the original page image attached after this prompt.

Use the markdown as a structural HINT, but the IMAGE is the GROUND TRUTH. The Typhoon OCR is known to:
  • fabricate the Thai-word column from the digit it read (so words "match" the digits even when the
    handwriting actually says something different — DO NOT trust the markdown words alone)
  • miss strike-throughs / corrections (a digit crossed out and replaced with another value)
  • duplicate numbers in ballot-count lines
Whenever in doubt, look at the image and read the actual handwriting.

Return ONLY a JSON object — no markdown fences, no commentary.
Schema:
{{
  "station_id": int,
  "constituency_number": int,
  "province": str,
  "good_ballots": int,
  "bad_ballots": int,
  "no_vote_ballots": int,
  "total_ballots": int,
  "votes": {{ "candidate_<N>_votes": int, ... }},
  "parties": {{ "candidate_<N>_party": "<party name from สังกัดพรรคการเมือง column>", ... }},
  "total_votes_sum": int   // value on the "รวมคะแนนทั้งสิ้น" row (or 0 if not on this page)
}}
Use 0 for any missing or unreadable number. Use "" for empty party rows. Convert Thai digits (๐-๙) to Arabic.
When the row is a party-list row, normalize party names against this official party-number reference:
{self.official_party_reference}

CRITICAL VOTE-COUNT RULES (the form is designed so that votes are written TWICE — once as digits and once as Thai words — for cross-validation):
- Each candidate row contains: (1) candidate/party number, (2) name, (3) vote count in digits, (4) vote count in Thai words within parentheses, e.g. "(เจ็ดสิบหก)" = 76.
- WHENEVER the Thai-word form is readable, USE IT AS THE GROUND TRUTH and ignore the digits column. Examples:
    "76 (เจ็ดสิบหก)"        → 76
    "5,150 (ห้า)"           → 5   ← digits were misread; ห้า means 5
    "113 (หนึ่งร้อยยี่สิบสาม)" → 123
- Common Thai number words: ศูนย์=0 หนึ่ง=1 สอง=2 สาม=3 สี่=4 ห้า=5 หก=6 เจ็ด=7 แปด=8 เก้า=9 สิบ=10 ยี่สิบ=20 ร้อย=100 พัน=1000.
- If the Thai-word column is empty, fall back to digits.
- If the row's vote cell is SPLIT into TWO sub-cells (e.g. "| 2 | ๘๐๐ |"), DO NOT concatenate. The right-hand value is usually a form serial number, not part of the vote. Use only the LEFT digit cell, or prefer the Thai-word column.
- A single candidate cannot exceed good_ballots. If your number exceeds good_ballots, re-parse using the Thai-word column.
- The markdown may be wrong even when the digit and Thai word agree with each other. Always re-read the attached IMAGE/CROP for every handwritten vote.
- Common severe mistakes to avoid: 22 misread as 52/62, 64 misread as 34, and 11 misread as 1. Count separate vertical strokes carefully.
- If the image/crop disagrees with the markdown, the image/crop wins.
- In party-list tables, the first column is the party number. NEVER borrow digits from it. A row with party number 27 and vote 11 is 11, NOT 17 or 271.
- Distinguish Thai words carefully: "สิบเอ็ด" = 11, "สิบเจ็ด" = 17. If the word has เอ็ด/อ, output 11; do not invent เจ็ด/จ.

MANDATORY SUM-ROW CROSS-CHECK (do this BEFORE returning):
- The form has a row "รวมคะแนนทั้งสิ้น" (TOTAL VOTES) at the bottom of the candidate table — usually a handwritten number plus Thai words. Read it.
- Sum your extracted candidate votes. The sum MUST equal both (a) good_ballots and (b) the รวมคะแนนทั้งสิ้น row.
- If your sum doesn't match, you have misread one or more candidate rows. Look again at the image — pay extra attention to digits that look ambiguous and re-read the Thai-word column for those rows.
- Coincidental matches are NOT acceptable: if the sum equals the total but you suspect an individual digit, recheck that digit before finalizing.

CONFUSING HANDWRITTEN DIGITS (Thai forms commonly mix these up — look carefully):
- "2" with a curled top can look like "5" or "9" — count strokes; "2" has a single curve ending in a flat baseline.
- "6" written quickly can look like "3" — "6" is a closed loop at the bottom; "3" is two open curves.
- "0" vs "6" — "0" is fully closed; "6" has a tail.
- "4" vs "9" — "4" has an open top angle; "9" has a closed loop on top.
- "1" vs "7" — "1" is a single vertical stroke (with a small flag); "7" has a horizontal top.
- "11" vs "17" — count both digits! Both can look like vertical strokes when written hastily.

CONFUSING THAI NUMBER WORDS (read the entire word, do not skim):
- "สิบเอ็ด" (11) vs "สิบเจ็ด" (17) — the second syllable is different: เอ็ด has the อ vowel, เจ็ด has จ.
- "สอง" (2) vs "เก้า" (9) vs "ห้า" (5) — completely different shapes; if unsure, the Thai word ALWAYS wins over a misread digit.
- "หก" (6) vs "สาม" (3) — หก = two characters; สาม = three characters with vowel.
- When uncertain, read the Thai-word column FIRST (ศูนย์=0 หนึ่ง=1 สอง=2 สาม=3 สี่=4 ห้า=5 หก=6 เจ็ด=7 แปด=8 เก้า=9 สิบ=10 สิบเอ็ด=11 สิบเจ็ด=17) and use it to disambiguate the digit.
- NEVER concatenate the party หมายเลข (๑, ๒, ๓...) with the vote count.

CRITICAL BALLOT-COUNT RULES:
- Lines like "บัตรดี -> 11 จำนวน 15 ใบ" may contain TWO numbers; pick the HANDWRITTEN value (after "จำนวน", before "ใบ"/"บัตร").
- good_ballots + bad_ballots + no_vote_ballots ≈ total_ballots — use this to validate.

PROVINCE: This dataset is from "แพร่" (Phrae) constituency 3. If OCR shows garbled short forms like "เพ", "เพิ่ง", "เพอร์", "เพชร" — output "แพร่". Only output a different province if the markdown clearly shows a complete name.

OCR transcription:
---
{markdown}
---
"""
            # Build multi-modal contents: prompt + page image
            from google.genai import types as genai_types
            image_parts = []
            try:
                img_bytes = self._load_image_bytes_for_gemini(img_path)
                if img_bytes:
                    image_parts.append(
                        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    )
                for crop_bytes in self._load_focus_crop_bytes_for_gemini(img_path):
                    image_parts.append(
                        genai_types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg")
                    )
            except Exception as e:
                logger.warning(f"Could not attach image {img_path.name} to Stage B: {e}")

            extracted = self._stage_b_call_with_verify(
                gemini_client, stage_b_prompt, image_parts,
                model="gemini-flash-lite-latest",
            )

            if not extracted:
                return {
                    "source_file": img_path.name,
                    "form_type": form_type,
                    "ocr_confidence": 0.5,
                    "raw_text_preview": "Stage B failed; markdown saved",
                    "ocr_success": False,
                }

            # Post-process: sanitize any remaining Thai digits in numeric values
            extracted = self._sanitize_thai_digits(extracted)
            quality = self._quality_metrics(extracted)

            record = {
                "station_id": extracted.get("station_id", 0),
                "constituency_number": extracted.get("constituency_number", 0),
                "province": extracted.get("province", ""),
                "good_ballots": extracted.get("good_ballots", 0),
                "bad_ballots": extracted.get("bad_ballots", 0),
                "no_vote_ballots": extracted.get("no_vote_ballots", 0),
                "total_ballots": extracted.get("total_ballots", 0),
                "ocr_confidence": quality["ocr_confidence"],
                "raw_text_preview": json.dumps(extracted.get("votes", {}), ensure_ascii=False)[:200],
                "needs_review": quality["needs_review"],
                "votes_sum": quality["votes_sum"],
                "vote_sum_match": quality["vote_sum_match"],
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
            record = self._apply_manual_corrections(record)

            time.sleep(2)  # Pace between images

        elif self.engine == "easyocr":
            raw_results = self.reader.readtext(
                img,
                detail=1,
                paragraph=False,
                contrast_ths=0.2,   
                adjust_contrast=0.6, 
                text_threshold=0.6,
                low_text=0.35,      
                mag_ratio=1.5       
            )
            # Extract structured fields from raw OCR output
            record = self.field_extractor.extract(
                raw_results,
                form_type=form_type,
                engine=self.engine
            )
        elif self.engine == "tesseract":
            raw_results = self.reader.image_to_data(
                img, lang="tha+eng", output_type=self.reader.Output.DICT
            )
            record = self.field_extractor.extract(
                raw_results,
                form_type=form_type,
                engine=self.engine
            )

        # Add metadata
        record["source_file"] = img_path.name
        record["form_type"] = form_type
        record = self._apply_manual_corrections(record)
        record["ocr_success"] = (record.get("ocr_confidence", 0) > 0)

        return record

    def generate_confidence_report(self):
        """Generate a report of OCR confidence scores for quality review."""
        report_rows = []
        for csv_file in OCR_RAW_DIR.glob("raw_*.csv"):
            if csv_file.stem.endswith("_checkpoint"):
                continue
            df = pd.read_csv(csv_file)
            if "ocr_confidence" in df.columns:
                low_conf = df[df["ocr_confidence"] < OCR_CONFIDENCE_THRESHOLD]
                report_rows.append({
                    "form_type": csv_file.stem.replace("raw_", ""),
                    "total_records": len(df),
                    "low_confidence_count": len(low_conf),
                    "avg_confidence": df["ocr_confidence"].mean(),
                    "min_confidence": df["ocr_confidence"].min(),
                })

        if report_rows:
            report_df = pd.DataFrame(report_rows)
            report_path = OCR_RAW_DIR / "confidence_report.csv"
            report_df.to_csv(report_path, index=False)
            logger.info(f"Confidence report saved to {report_path}")
            print("\n=== OCR Confidence Report ===")
            print(report_df.to_string(index=False))


ENGINE_CHOICES = {
    "gemini": "Google Gemini Vision — strong quality, free tier 10 RPM / 250 RPD",
    "typhoon": "Typhoon OCR (Stage A vision) + Gemini text (Stage B). Best quality. Subject to BOTH APIs' rate limits.",
    "easyocr_gemini": "Local EasyOCR (Stage A, no rate limit) + Gemini text (Stage B). Recommended for bulk runs.",
    "easyocr": "EasyOCR only. Fully offline. Lower extraction quality (no LLM post-processing).",
    "tesseract": "Tesseract only. Fully offline. Lowest quality without tuning.",
}


def _select_engine_interactive() -> str:
    print("\nAvailable OCR engines:\n")
    items = list(ENGINE_CHOICES.items())
    for i, (k, desc) in enumerate(items, 1):
        marker = " (default)" if k == OCR_ENGINE else ""
        print(f"  [{i}] {k}{marker}\n      {desc}")
    print()
    raw = input(f"Choose engine [1-{len(items)}] or name (Enter = {OCR_ENGINE}): ").strip()
    if not raw:
        return OCR_ENGINE
    if raw.isdigit() and 1 <= int(raw) <= len(items):
        return items[int(raw) - 1][0]
    if raw in ENGINE_CHOICES:
        return raw
    print(f"Unknown choice '{raw}', using default {OCR_ENGINE}")
    return OCR_ENGINE


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR pipeline for Thai election forms.")
    parser.add_argument(
        "--engine", "-e",
        choices=list(ENGINE_CHOICES.keys()),
        help=f"OCR engine to use. Default: {OCR_ENGINE}.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Choose OCR engine from an interactive prompt.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run OCR only on the 'sample' directory without overwriting main checkpoints.",
    )
    args = parser.parse_args()

    engine = _select_engine_interactive() if args.interactive else (args.engine or OCR_ENGINE)
    logger.info(f"Selected engine: {engine}")

    pipeline = OCRPipeline(engine=engine)
    
    if args.sample:
        pipeline.process_all_forms(form_types=["sample"])
    else:
        pipeline.process_all_forms()
        
    pipeline.generate_confidence_report()
