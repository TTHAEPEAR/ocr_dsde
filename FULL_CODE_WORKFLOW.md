# Data Science for Thailand Election 2026 — Full Code Workflow

**Course:** 2110446 Data Science and Data Engineering (2025/2)  
**Objective:** Convert unstructured ECT election PDFs → structured data → analysis → BI dashboard  
**Requirement:** OCR the ENTIRE selected constituency (≥250 polling stations, all 6 form types)

---

## Project Directory Structure

```
election-2026-project/
│
├── config.py                  # Global settings & paths
├── requirements.txt           # Python dependencies
│
├── 01_download/
│   └── download_pdfs.py       # Phase 1: Scrape & download PDFs from ECT
│
├── 02_preprocess/
│   └── preprocess_images.py   # Phase 1C: PDF → image conversion + enhancement
│
├── 03_ocr/
│   ├── ocr_pipeline.py        # Phase 2: Main OCR engine
│   ├── field_extractor.py     # Phase 2B: Parse OCR output → structured fields
│   └── ocr_utils.py           # Shared OCR helpers
│
├── 04_clean/
│   ├── clean_data.py          # Phase 3A: Fix OCR errors, normalize
│   └── validate_data.py       # Phase 3B: Cross-validation checks
│
├── 05_analysis/
│   ├── analysis.py            # Phase 4: Statistical analysis
│   └── notebooks/
│       ├── EDA.ipynb           # Exploratory data analysis
│       └── comparative.ipynb   # 2026 vs 2023 comparison
│
├── 06_dashboard/
│   └── app.py                 # Phase 5: Streamlit BI dashboard
│
├── data/
│   ├── raw_pdfs/              # Downloaded PDFs organized by form type
│   ├── images/                # Converted & preprocessed images
│   ├── ocr_raw/               # Raw OCR output CSVs
│   ├── cleaned/               # Validated & cleaned datasets
│   └── reference/             # P'PanJ reference data + 2023 data
│
├── outputs/
│   ├── figures/               # Generated plots
│   └── reports/               # Analysis reports
│
└── presentation/
    ├── slides.pptx
    └── slides.pdf
```

---

## Phase 0: Configuration & Setup

### `config.py`
```python
"""
Global configuration for the Election 2026 Data Science Project.
Update CONSTITUENCY_NAME and PROVINCE before running.
"""
from pathlib import Path

# ===== PROJECT SETTINGS =====
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
IMAGE_DIR = DATA_DIR / "images"
OCR_RAW_DIR = DATA_DIR / "ocr_raw"
CLEANED_DIR = DATA_DIR / "cleaned"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ===== CONSTITUENCY SELECTION =====
# Choose ONE constituency with ≥250 polling stations
PROVINCE = "ปราจีนบุรี"               # Thai province name
CONSTITUENCY_NUMBER = 3               # Constituency number
CONSTITUENCY_NAME = f"{PROVINCE}เขต{CONSTITUENCY_NUMBER}"

# ===== ECT DATA SOURCE =====
ECT_BASE_URL = "https://www.ect.go.th/ect_th/th/election-2026"

# ===== FORM TYPES (all 6 required) =====
FORM_TYPES = {
    "5_18":        "Election day - Constituency vote",
    "5_18_party":  "Election day - Party-list vote",
    "5_16":        "Advance voting (in-district) - Constituency vote",
    "5_16_party":  "Advance voting (in-district) - Party-list vote",
    "5_17":        "Advance voting (out-of-district) - Constituency vote",
    "5_17_party":  "Advance voting (out-of-district) - Party-list vote",
}

# ===== OCR SETTINGS =====
OCR_ENGINE = "easyocr"  # Options: "tesseract", "easyocr", "paddleocr"
OCR_LANGUAGES = ["th", "en"]
OCR_CONFIDENCE_THRESHOLD = 0.6

# ===== PARTY NAME MAPPING =====
# Official party list for the 2026 election (update as needed)
PARTY_NAMES = {
    1: "เศรษฐกิจ",
    2: "เสรีรวมไทย",
    3: "รวมไทย",
    4: "รวมพลังประชาชน",
    5: "ท้องที่ไทย",
    6: "อนาคตไทย",
    7: "พลังเพื่อไทย",
    8: "ไทยศรีวิไลย์",
    9: "พลังสังคมใหม่",
    10: "สังคมประชาธิปไตยไทย",
    # ... add all parties from the ballot
}

# Create directories
for d in [RAW_PDF_DIR, IMAGE_DIR, OCR_RAW_DIR, CLEANED_DIR,
          REFERENCE_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

### `requirements.txt`
```
# Data acquisition
requests>=2.31
beautifulsoup4>=4.12
selenium>=4.15

# PDF & Image processing
pdf2image>=1.16
Pillow>=10.0
opencv-python>=4.8
pypdf>=3.17

# OCR engines (choose one or more)
easyocr>=1.7
pytesseract>=0.3.10
# paddleocr>=2.7  # optional, heavier

# Data processing
pandas>=2.1
numpy>=1.25
openpyxl>=3.1
pandera>=0.17

# Analysis & visualization
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18
scipy>=1.11
scikit-learn>=1.3
statsmodels>=0.14

# Dashboard
streamlit>=1.28

# Utilities
tqdm>=4.66
fuzzywuzzy>=0.18
python-Levenshtein>=0.23
loguru>=0.7
```

---

## Phase 1: Data Acquisition

### `01_download/download_pdfs.py`
```python
"""
Phase 1: Download all election PDF documents from ECT website.
Downloads all 6 form types for the selected constituency.

Usage:
    python 01_download/download_pdfs.py
"""
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    ECT_BASE_URL, RAW_PDF_DIR, PROVINCE,
    CONSTITUENCY_NUMBER, FORM_TYPES
)

# Configure logging
logger.add("download.log", rotation="10 MB")


class ECTPDFDownloader:
    """Scrapes and downloads election PDFs from the ECT website."""

    def __init__(self, base_url: str, province: str, constituency: int):
        self.base_url = base_url
        self.province = province
        self.constituency = constituency
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Election Research Project)"
        })

    def discover_pdf_links(self) -> dict[str, list[str]]:
        """
        Crawl the ECT site to find all PDF download links
        for the selected province and constituency.

        Returns:
            dict mapping form_type -> list of PDF URLs
        """
        logger.info(f"Discovering PDFs for {self.province} เขต {self.constituency}")
        pdf_links = {ft: [] for ft in FORM_TYPES}

        # Step 1: Navigate to province page
        # NOTE: Actual URL structure depends on ECT site layout.
        # You may need Selenium if the site uses JavaScript rendering.
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Step 2: Find links matching our province/constituency
            # This is a template — adjust selectors based on actual ECT HTML
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.endswith(".pdf") and self.province in link.text:
                    # Classify by form type based on filename or link text
                    form_type = self._classify_form_type(href, link.text)
                    if form_type:
                        pdf_links[form_type].append(href)

        except requests.RequestException as e:
            logger.error(f"Failed to crawl ECT site: {e}")
            raise

        total = sum(len(v) for v in pdf_links.values())
        logger.info(f"Found {total} PDF links across {len(FORM_TYPES)} form types")
        return pdf_links

    def _classify_form_type(self, url: str, text: str) -> str | None:
        """Classify a PDF link into one of the 6 form types."""
        url_lower = url.lower()
        text_lower = text.lower()

        if "5-18" in url_lower or "5/18" in text_lower:
            if "บช" in text or "party" in url_lower:
                return "5_18_party"
            return "5_18"
        elif "5-16" in url_lower or "5/16" in text_lower:
            if "บช" in text or "party" in url_lower:
                return "5_16_party"
            return "5_16"
        elif "5-17" in url_lower or "5/17" in text_lower:
            if "บช" in text or "party" in url_lower:
                return "5_17_party"
            return "5_17"
        return None

    def download_all(self, pdf_links: dict[str, list[str]]):
        """Download all discovered PDFs, organized by form type."""
        for form_type, urls in pdf_links.items():
            form_dir = RAW_PDF_DIR / form_type
            form_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {len(urls)} PDFs for form {form_type}")
            for i, url in enumerate(tqdm(urls, desc=f"Form {form_type}")):
                self._download_single(url, form_dir, i + 1)
                time.sleep(0.5)  # Be polite to the server

    def _download_single(self, url: str, save_dir: Path, index: int):
        """Download a single PDF file."""
        try:
            # Make URL absolute if needed
            if not url.startswith("http"):
                url = f"https://www.ect.go.th{url}"

            response = self.session.get(url, timeout=60)
            response.raise_for_status()

            # Extract filename or generate one
            filename = url.split("/")[-1]
            if not filename.endswith(".pdf"):
                filename = f"station_{index:04d}.pdf"

            save_path = save_dir / filename
            save_path.write_bytes(response.content)
            logger.debug(f"Saved: {save_path}")

        except requests.RequestException as e:
            logger.warning(f"Failed to download {url}: {e}")

    def verify_completeness(self):
        """Verify we have PDFs for all form types and enough stations."""
        logger.info("=== Download Verification ===")
        for form_type in FORM_TYPES:
            form_dir = RAW_PDF_DIR / form_type
            count = len(list(form_dir.glob("*.pdf"))) if form_dir.exists() else 0
            status = "OK" if count > 0 else "MISSING"
            logger.info(f"  {form_type}: {count} files [{status}]")

        # Check 5/18 has ≥250 stations
        station_dir = RAW_PDF_DIR / "5_18"
        if station_dir.exists():
            n_stations = len(list(station_dir.glob("*.pdf")))
            if n_stations < 250:
                logger.warning(
                    f"Only {n_stations} station PDFs found. "
                    f"Need ≥250 for this constituency."
                )


if __name__ == "__main__":
    downloader = ECTPDFDownloader(ECT_BASE_URL, PROVINCE, CONSTITUENCY_NUMBER)

    # Option A: Automatic discovery (if ECT site is crawlable)
    # pdf_links = downloader.discover_pdf_links()
    # downloader.download_all(pdf_links)

    # Option B: Manual download — place PDFs in data/raw_pdfs/{form_type}/
    # Then just run verification:
    downloader.verify_completeness()
```

---

## Phase 1C: PDF Pre-processing

### `02_preprocess/preprocess_images.py`
```python
"""
Phase 1C: Convert PDF documents to preprocessed images for OCR.
Applies deskewing, denoising, and binarization for better OCR accuracy.

Usage:
    python 02_preprocess/preprocess_images.py
"""
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_PDF_DIR, IMAGE_DIR, FORM_TYPES


class PDFPreprocessor:
    """Converts PDFs to clean images optimized for OCR."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def convert_all_pdfs(self):
        """Convert all PDFs across all form types to images."""
        for form_type in FORM_TYPES:
            pdf_dir = RAW_PDF_DIR / form_type
            if not pdf_dir.exists():
                logger.warning(f"No PDF directory for {form_type}, skipping")
                continue

            img_dir = IMAGE_DIR / form_type
            img_dir.mkdir(parents=True, exist_ok=True)

            pdfs = list(pdf_dir.glob("*.pdf"))
            logger.info(f"Converting {len(pdfs)} PDFs for form {form_type}")

            for pdf_path in tqdm(pdfs, desc=f"Form {form_type}"):
                self._convert_single_pdf(pdf_path, img_dir)

    def _convert_single_pdf(self, pdf_path: Path, output_dir: Path):
        """Convert a single PDF to preprocessed image(s)."""
        try:
            # PDF → PIL Images
            pages = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt="png"
            )

            for page_num, page_img in enumerate(pages):
                # Convert PIL → OpenCV format
                img_array = np.array(page_img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Apply preprocessing pipeline
                processed = self._preprocess_image(img_bgr)

                # Save
                stem = pdf_path.stem
                filename = f"{stem}_page{page_num + 1}.png"
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), processed)

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path.name}: {e}")

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for scanned election documents.

        Steps:
        1. Convert to grayscale
        2. Deskew (correct rotation)
        3. Denoise
        4. Adaptive thresholding (binarize)
        5. Remove borders/noise
        """
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Deskew
        gray = self._deskew(gray)

        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 4. Adaptive threshold for binarization
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8
        )

        # 5. Morphological cleanup — remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return cleaned

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew/rotation in scanned document."""
        coords = np.column_stack(np.where(image < 128))
        if len(coords) < 100:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:  # Skip tiny corrections
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated


if __name__ == "__main__":
    preprocessor = PDFPreprocessor(dpi=300)
    preprocessor.convert_all_pdfs()
    logger.info("Preprocessing complete!")
```

---

## Phase 2: OCR Extraction

### `03_ocr/ocr_pipeline.py`
```python
"""
Phase 2: OCR pipeline — extract text and vote data from preprocessed images.
Supports multiple OCR engines: EasyOCR, Tesseract, PaddleOCR.

Usage:
    python 03_ocr/ocr_pipeline.py
"""
import easyocr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import cv2
import re
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    IMAGE_DIR, OCR_RAW_DIR, FORM_TYPES,
    OCR_ENGINE, OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD
)
from field_extractor import FieldExtractor


class OCRPipeline:
    """Main OCR pipeline for election document processing."""

    def __init__(self, engine: str = OCR_ENGINE):
        self.engine = engine
        self.reader = self._init_engine()
        self.field_extractor = FieldExtractor()

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
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")

    def process_all_forms(self):
        """Run OCR on all form types and produce raw CSV outputs."""
        all_results = []

        for form_type in FORM_TYPES:
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
            combined_path = OCR_RAW_DIR / "raw_all_forms.csv"
            combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
            logger.info(f"Combined output: {len(combined_df)} total records")

        return all_results

    def _process_form_type(self, form_type: str, img_dir: Path) -> list[dict]:
        """Process all images for a single form type."""
        results = []
        images = sorted(img_dir.glob("*.png"))

        for img_path in tqdm(images, desc=f"OCR {form_type}"):
            try:
                record = self._process_single_image(img_path, form_type)
                if record:
                    results.append(record)
            except Exception as e:
                logger.error(f"OCR failed for {img_path.name}: {e}")
                results.append({
                    "source_file": img_path.name,
                    "form_type": form_type,
                    "error": str(e),
                    "ocr_success": False
                })
        return results

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
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Run OCR
        if self.engine == "easyocr":
            raw_results = self.reader.readtext(img)
            # raw_results = [(bbox, text, confidence), ...]
        elif self.engine == "tesseract":
            raw_results = self.reader.image_to_data(
                img, lang="tha+eng", output_type=self.reader.Output.DICT
            )

        # Extract structured fields from raw OCR output
        record = self.field_extractor.extract(
            raw_results,
            form_type=form_type,
            engine=self.engine
        )

        # Add metadata
        record["source_file"] = img_path.name
        record["form_type"] = form_type
        record["ocr_success"] = True

        return record

    def generate_confidence_report(self):
        """Generate a report of OCR confidence scores for quality review."""
        report_rows = []
        for csv_file in OCR_RAW_DIR.glob("raw_*.csv"):
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


if __name__ == "__main__":
    pipeline = OCRPipeline()
    pipeline.process_all_forms()
    pipeline.generate_confidence_report()
```

### `03_ocr/field_extractor.py`
```python
"""
Phase 2B: Extract structured fields from raw OCR output.
Handles both constituency vote forms and party-list vote forms.
"""
import re
from loguru import logger


class FieldExtractor:
    """Parse raw OCR text into structured election data fields."""

    # Thai digit mapping (handwritten OCR often produces Thai numerals)
    THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

    def extract(self, raw_results, form_type: str, engine: str) -> dict:
        """
        Main extraction entry point.

        Args:
            raw_results: Raw OCR output (format depends on engine)
            form_type: One of the 6 form types
            engine: "easyocr" or "tesseract"

        Returns:
            dict with extracted fields
        """
        if engine == "easyocr":
            texts, confidences, bboxes = self._parse_easyocr(raw_results)
        elif engine == "tesseract":
            texts, confidences, bboxes = self._parse_tesseract(raw_results)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        # Join all text for full-document parsing
        full_text = " ".join(texts)

        # Extract common fields
        record = {
            "ocr_confidence": sum(confidences) / max(len(confidences), 1),
            "raw_text_preview": full_text[:200],
        }

        # Extract station/constituency identifiers
        record.update(self._extract_identifiers(full_text))

        # Route to form-specific extractor
        if "party" in form_type:
            record.update(self._extract_party_list_votes(texts, bboxes))
        else:
            record.update(self._extract_constituency_votes(texts, bboxes))

        # Extract ballot summary (good/bad/no-vote)
        record.update(self._extract_ballot_summary(full_text))

        return record

    def _parse_easyocr(self, results):
        """Parse EasyOCR output format: [(bbox, text, conf), ...]"""
        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        bboxes = [r[0] for r in results]
        return texts, confidences, bboxes

    def _parse_tesseract(self, results):
        """Parse Tesseract output dict format."""
        texts, confidences, bboxes = [], [], []
        for i in range(len(results["text"])):
            text = results["text"][i].strip()
            conf = int(results["conf"][i])
            if text and conf > 0:
                texts.append(text)
                confidences.append(conf / 100.0)
                bboxes.append((
                    results["left"][i], results["top"][i],
                    results["width"][i], results["height"][i]
                ))
        return texts, confidences, bboxes

    def _extract_identifiers(self, text: str) -> dict:
        """Extract station ID, constituency, province from text."""
        result = {}

        # Look for constituency number pattern: "เขตเลือกตั้งที่ X"
        match = re.search(r"เขตเลือกตั้งที่\s*(\d+)", text)
        if match:
            result["constituency_number"] = int(match.group(1))

        # Look for station number: "หน่วยเลือกตั้งที่ X" or just numbers
        match = re.search(r"หน่วย(?:เลือกตั้ง)?ที่\s*(\d+)", text)
        if match:
            result["station_id"] = int(match.group(1))

        # Province name
        match = re.search(r"จังหวัด\s*(\S+)", text)
        if match:
            result["province"] = match.group(1)

        return result

    def _extract_constituency_votes(self, texts: list, bboxes: list) -> dict:
        """
        Extract candidate votes from constituency vote forms (5/16, 5/17, 5/18).
        The form has a table: [number] [party name] [vote count in handwriting]
        """
        votes = {}
        for i, text in enumerate(texts):
            # Normalize Thai digits
            normalized = text.translate(self.THAI_DIGITS)

            # Try to find pattern: party_number followed by vote count
            # The layout is typically: row number | party name | vote count
            numbers = re.findall(r"\d+", normalized)
            if numbers:
                # Heuristic: use spatial position (bbox) to determine
                # if this number is a party index or a vote count
                # Numbers on the right side of the page = vote counts
                if bboxes and len(bboxes) > i:
                    bbox = bboxes[i]
                    x_position = bbox[0][0] if isinstance(bbox[0], list) else bbox[0]

                    # If x > 60% of page width, likely a vote count
                    if x_position > 500:  # Adjust based on your image dimensions
                        vote_count = int(numbers[0])
                        votes[f"candidate_{len(votes)+1}_votes"] = vote_count

        return votes

    def _extract_party_list_votes(self, texts: list, bboxes: list) -> dict:
        """
        Extract party-list votes from party-list forms (5/16บช, 5/17บช, 5/18บช).
        Similar table structure but for party-list ballot.
        """
        votes = {}
        for i, text in enumerate(texts):
            normalized = text.translate(self.THAI_DIGITS)
            numbers = re.findall(r"\d+", normalized)

            if numbers and bboxes and len(bboxes) > i:
                bbox = bboxes[i]
                x_position = bbox[0][0] if isinstance(bbox[0], list) else bbox[0]

                if x_position > 500:
                    vote_count = int(numbers[0])
                    votes[f"party_{len(votes)+1}_votes"] = vote_count

        return votes

    def _extract_ballot_summary(self, text: str) -> dict:
        """Extract ballot summary: good ballots, bad ballots, no-vote ballots."""
        result = {}
        normalized = text.translate(self.THAI_DIGITS)

        # Total ballots: "บัตรดี X บัตร"
        match = re.search(r"บัตรดี\s*(\d+)", normalized)
        if match:
            result["good_ballots"] = int(match.group(1))

        # Bad ballots: "บัตรเสีย X บัตร"
        match = re.search(r"บัตรเสีย\s*(\d+)", normalized)
        if match:
            result["bad_ballots"] = int(match.group(1))

        # No-vote ballots: "ไม่เลือกผู้สมัคร X บัตร"
        match = re.search(r"ไม่เลือก(?:ผู้สมัคร)?\s*(\d+)", normalized)
        if match:
            result["no_vote_ballots"] = int(match.group(1))

        # Total received ballots
        match = re.search(r"(?:รวม|ทั้งหมด|ได้รับบัตร)\s*(\d+)", normalized)
        if match:
            result["total_ballots"] = int(match.group(1))

        return result
```

---

## Phase 3: Data Cleaning & Validation

### `04_clean/clean_data.py`
```python
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
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import OCR_RAW_DIR, CLEANED_DIR, PARTY_NAMES


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
        self.party_name_list = list(PARTY_NAMES.values())

    def clean_all(self) -> pd.DataFrame:
        """Run full cleaning pipeline on all raw OCR data."""
        # Load raw combined data
        raw_path = OCR_RAW_DIR / "raw_all_forms.csv"
        if not raw_path.exists():
            logger.error("No raw data found. Run OCR pipeline first.")
            return pd.DataFrame()

        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} raw records")

        # Remove failed OCR records
        df = df[df.get("ocr_success", True) == True].copy()

        # Step 1: Fix numeric OCR errors
        df = self._fix_numeric_errors(df)

        # Step 2: Normalize party names
        df = self._normalize_party_names(df)

        # Step 3: Handle missing values
        df = self._handle_missing(df)

        # Step 4: Enforce data types
        df = self._enforce_types(df)

        # Step 5: Derive computed columns
        df = self._compute_derived(df)

        # Save cleaned data
        cleaned_path = CLEANED_DIR / "election_results_cleaned.csv"
        df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        logger.info(f"Cleaned data saved: {len(df)} records → {cleaned_path}")

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
        """Use fuzzy matching to correct misspelled party names."""
        party_cols = [c for c in df.columns if "party_name" in c.lower()]

        for col in party_cols:
            if col in df.columns:
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


# Need this import for _correct_digits
import re

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_all()
    print(f"\nCleaned dataset shape: {cleaned_df.shape}")
    print(cleaned_df.head())
```

### `04_clean/validate_data.py`
```python
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
        vote_cols = [c for c in df.columns if c.endswith("_votes")]
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
        if "station_id" in df.columns and "form_type" in df.columns:
            dupes = df.duplicated(subset=["station_id", "form_type"], keep=False)
            if dupes.sum() > 0:
                self.warnings.append(
                    f"Duplicate station+form entries: {dupes.sum()} records"
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


if __name__ == "__main__":
    cleaned_path = CLEANED_DIR / "election_results_cleaned.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        validator = DataValidator()
        passed = validator.validate_all(df)
        print(f"\nValidation {'PASSED' if passed else 'FAILED'}")
    else:
        print("No cleaned data found. Run clean_data.py first.")
```

---

## Phase 4: Data Science Analysis

### `05_analysis/analysis.py`
```python
"""
Phase 4: Data Science Analysis — explore patterns, trends, and insights.

Usage:
    python 05_analysis/analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, FIGURES_DIR, REFERENCE_DIR

# Thai font support
plt.rcParams["font.family"] = "Tahoma"  # or "TH Sarabun New" if available


class ElectionAnalyzer:
    """Comprehensive analysis of Thailand Election 2026 data."""

    def __init__(self):
        self.df = self._load_data()
        self.results = {}

    def _load_data(self) -> pd.DataFrame:
        path = CLEANED_DIR / "election_results_cleaned.csv"
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records for analysis")
        return df

    def run_all_analyses(self):
        """Execute the full analysis suite."""
        self.descriptive_statistics()
        self.party_performance_analysis()
        self.turnout_analysis()
        self.advance_vs_election_day()
        self.anomaly_detection()
        self.station_clustering()
        self.comparative_2023()
        logger.info("All analyses complete. Figures saved to outputs/figures/")

    # ── 4A. Descriptive Statistics ──

    def descriptive_statistics(self):
        """Basic descriptive stats on vote distributions."""
        logger.info("Running descriptive statistics...")

        vote_cols = [c for c in self.df.columns if c.endswith("_votes")]
        summary = self.df[vote_cols].describe()
        summary.to_csv(FIGURES_DIR / "descriptive_stats.csv")

        # Vote distribution histogram per party
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Vote Distribution by Candidate/Party", fontsize=14)

        for idx, col in enumerate(vote_cols[:6]):
            ax = axes[idx // 3, idx % 3]
            self.df[col].hist(bins=30, ax=ax, color="#6366f1", alpha=0.7)
            ax.set_title(col.replace("_votes", ""))
            ax.set_xlabel("Votes")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "vote_distribution.png", dpi=150)
        plt.close()

    # ── 4B. Party Performance ──

    def party_performance_analysis(self):
        """Analyze party performance across all stations."""
        logger.info("Analyzing party performance...")

        vote_cols = [c for c in self.df.columns if c.endswith("_votes")]
        if not vote_cols:
            return

        # Total votes per party
        party_totals = self.df[vote_cols].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        party_totals.plot(kind="barh", ax=ax, color="#6366f1")
        ax.set_title("Total Votes by Candidate/Party")
        ax.set_xlabel("Total Votes")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "party_totals.png", dpi=150)
        plt.close()

        # Vote share percentage
        total = party_totals.sum()
        vote_share = (party_totals / total * 100).round(2)
        vote_share.to_csv(FIGURES_DIR / "vote_share.csv")

        self.results["party_totals"] = party_totals
        self.results["vote_share"] = vote_share

    # ── 4B. Turnout Analysis ──

    def turnout_analysis(self):
        """Analyze voter turnout patterns across polling stations."""
        logger.info("Analyzing turnout...")

        if "total_ballots" not in self.df.columns:
            return

        # Turnout distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of total ballots per station
        self.df["total_ballots"].hist(bins=40, ax=axes[0], color="#10b981")
        axes[0].set_title("Ballots per Station Distribution")
        axes[0].set_xlabel("Total Ballots")

        # Invalid ballot ratio
        if "invalid_ballot_ratio" in self.df.columns:
            self.df["invalid_ballot_ratio"].hist(bins=40, ax=axes[1], color="#f59e0b")
            axes[1].set_title("Invalid Ballot Ratio Distribution")
            axes[1].set_xlabel("Ratio")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "turnout_analysis.png", dpi=150)
        plt.close()

    # ── 4B. Advance vs Election Day Comparison ──

    def advance_vs_election_day(self):
        """Compare advance voting patterns with election day voting."""
        logger.info("Comparing advance vs election day voting...")

        election_day = self.df[self.df["form_type"].isin(["5_18", "5_18_party"])]
        advance = self.df[self.df["form_type"].isin([
            "5_16", "5_16_party", "5_17", "5_17_party"
        ])]

        vote_cols = [c for c in self.df.columns if c.endswith("_votes")]
        if not vote_cols:
            return

        comparison = pd.DataFrame({
            "Election Day": election_day[vote_cols].sum(),
            "Advance Voting": advance[vote_cols].sum(),
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        comparison.plot(kind="bar", ax=ax, color=["#6366f1", "#f59e0b"])
        ax.set_title("Advance Voting vs Election Day")
        ax.set_ylabel("Total Votes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "advance_vs_election_day.png", dpi=150)
        plt.close()

    # ── 4C. Anomaly Detection ──

    def anomaly_detection(self):
        """Detect anomalous polling stations using statistical methods."""
        logger.info("Running anomaly detection...")

        vote_cols = [c for c in self.df.columns if c.endswith("_votes")]
        if not vote_cols or "station_id" not in self.df.columns:
            return

        # Z-score based anomaly detection on total votes
        if "total_ballots" in self.df.columns:
            z_scores = np.abs(stats.zscore(
                self.df["total_ballots"].dropna()
            ))
            anomalies = self.df.loc[z_scores > 2.5]

            if len(anomalies) > 0:
                logger.info(f"Found {len(anomalies)} anomalous stations (|z| > 2.5)")
                anomalies.to_csv(FIGURES_DIR / "anomalous_stations.csv", index=False)

        # IQR method on invalid ballot ratio
        if "invalid_ballot_ratio" in self.df.columns:
            Q1 = self.df["invalid_ballot_ratio"].quantile(0.25)
            Q3 = self.df["invalid_ballot_ratio"].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df["invalid_ballot_ratio"] < Q1 - 1.5 * IQR) |
                (self.df["invalid_ballot_ratio"] > Q3 + 1.5 * IQR)
            ]
            if len(outliers) > 0:
                logger.info(f"Found {len(outliers)} outlier stations by invalid ballot ratio")

    # ── 4C. Station Clustering ──

    def station_clustering(self):
        """Cluster polling stations by voting patterns using K-Means."""
        logger.info("Clustering polling stations...")

        vote_cols = [c for c in self.df.columns if c.endswith("_votes")]
        if len(vote_cols) < 2:
            return

        # Prepare features
        features = self.df[vote_cols].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        # K-Means with k=4
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df["cluster"] = kmeans.fit_predict(scaled)

        # Visualize clusters (PCA for 2D projection)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(scaled)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=self.df["cluster"], cmap="viridis",
            alpha=0.6, s=30
        )
        ax.set_title("Polling Station Clusters (PCA Projection)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "station_clusters.png", dpi=150)
        plt.close()

    # ── 4B. Comparative Analysis: 2026 vs 2023 ──

    def comparative_2023(self):
        """Compare 2026 results with 2023 election data."""
        logger.info("Running 2023 comparison...")

        ref_path = REFERENCE_DIR / "election_2023.csv"
        if not ref_path.exists():
            logger.warning("No 2023 reference data found, skipping comparison")
            return

        df_2023 = pd.read_csv(ref_path)

        # Compare total vote share shifts
        vote_cols = [c for c in self.df.columns if c.endswith("_votes")]
        current_totals = self.df[vote_cols].sum()

        vote_cols_2023 = [c for c in df_2023.columns if c.endswith("_votes")]
        previous_totals = df_2023[vote_cols_2023].sum()

        # Create shift visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(min(len(current_totals), len(previous_totals)))
        width = 0.35

        ax.bar([i - width/2 for i in x],
               previous_totals.values[:len(x)],
               width, label="2023", color="#94a3b8")
        ax.bar([i + width/2 for i in x],
               current_totals.values[:len(x)],
               width, label="2026", color="#6366f1")

        ax.set_title("Election Results: 2023 vs 2026")
        ax.set_ylabel("Total Votes")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "comparison_2023_2026.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    analyzer = ElectionAnalyzer()
    analyzer.run_all_analyses()
```

---

## Phase 5: BI Dashboard

### `06_dashboard/app.py`
```python
"""
Phase 5: Interactive BI Dashboard using Streamlit.

Usage:
    streamlit run 06_dashboard/app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DIR, FIGURES_DIR, CONSTITUENCY_NAME


# ===== Page Config =====
st.set_page_config(
    page_title=f"Election 2026 — {CONSTITUENCY_NAME}",
    page_icon="🗳️",
    layout="wide"
)


@st.cache_data
def load_data():
    path = CLEANED_DIR / "election_results_cleaned.csv"
    return pd.read_csv(path)


df = load_data()

# ===== Sidebar Filters =====
st.sidebar.title("Filters")
form_types = st.sidebar.multiselect(
    "Form Type",
    options=df["form_type"].unique().tolist(),
    default=df["form_type"].unique().tolist()
)

filtered_df = df[df["form_type"].isin(form_types)]

# ===== Tab Layout =====
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Station Detail", "Comparison", "Insights"
])


# ── TAB 1: Overview ──
with tab1:
    st.header(f"Election Results Overview — {CONSTITUENCY_NAME}")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    vote_cols = [c for c in filtered_df.columns if c.endswith("_votes")]
    total_votes = filtered_df[vote_cols].sum().sum() if vote_cols else 0

    col1.metric("Total Votes Cast", f"{total_votes:,.0f}")
    col2.metric("Polling Stations", filtered_df["station_id"].nunique()
                if "station_id" in filtered_df.columns else "N/A")
    col3.metric("Form Types Loaded", len(form_types))

    if "invalid_ballot_ratio" in filtered_df.columns:
        avg_invalid = filtered_df["invalid_ballot_ratio"].mean() * 100
        col4.metric("Avg Invalid Ballot %", f"{avg_invalid:.1f}%")

    # Party vote totals bar chart
    if vote_cols:
        party_totals = filtered_df[vote_cols].sum().sort_values(ascending=True)
        fig = px.bar(
            x=party_totals.values,
            y=party_totals.index,
            orientation="h",
            title="Total Votes by Candidate/Party",
            labels={"x": "Votes", "y": "Candidate/Party"},
            color_discrete_sequence=["#6366f1"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Vote share pie chart
    if vote_cols:
        fig_pie = px.pie(
            names=party_totals.index,
            values=party_totals.values,
            title="Vote Share Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ── TAB 2: Station Detail ──
with tab2:
    st.header("Per-Station Drill-down")

    if "station_id" in filtered_df.columns:
        station_list = sorted(filtered_df["station_id"].dropna().unique())
        selected_station = st.selectbox("Select Station", station_list)

        station_data = filtered_df[filtered_df["station_id"] == selected_station]
        st.dataframe(station_data, use_container_width=True)

        # Station-level vote breakdown
        if vote_cols:
            station_votes = station_data[vote_cols].sum()
            fig = px.bar(
                x=station_votes.index,
                y=station_votes.values,
                title=f"Station {selected_station} Vote Breakdown",
                color_discrete_sequence=["#10b981"]
            )
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap: stations × parties
    if vote_cols and "station_id" in filtered_df.columns:
        st.subheader("Vote Heatmap (Stations × Parties)")
        election_day = filtered_df[filtered_df["form_type"] == "5_18"]
        if len(election_day) > 0:
            pivot = election_day.set_index("station_id")[vote_cols].fillna(0)
            fig_heat = px.imshow(
                pivot.head(50),  # Show first 50 stations
                title="Vote Heatmap (first 50 stations)",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ── TAB 3: Comparison ──
with tab3:
    st.header("Advance Voting vs Election Day")

    election_day = filtered_df[filtered_df["form_type"].isin(["5_18", "5_18_party"])]
    advance = filtered_df[filtered_df["form_type"].isin([
        "5_16", "5_16_party", "5_17", "5_17_party"
    ])]

    if vote_cols:
        comparison = pd.DataFrame({
            "Election Day": election_day[vote_cols].sum(),
            "Advance Voting": advance[vote_cols].sum(),
        }).reset_index()
        comparison.columns = ["Party", "Election Day", "Advance Voting"]

        fig = px.bar(
            comparison, x="Party",
            y=["Election Day", "Advance Voting"],
            barmode="group",
            title="Advance vs Election Day Voting",
            color_discrete_sequence=["#6366f1", "#f59e0b"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2023 vs 2026 comparison (if data exists)
    ref_path = CLEANED_DIR.parent / "reference" / "election_2023.csv"
    if ref_path.exists():
        st.subheader("2023 vs 2026 Comparison")
        df_2023 = pd.read_csv(ref_path)
        st.info("Historical comparison loaded from 2023 data")


# ── TAB 4: Insights ──
with tab4:
    st.header("Key Insights & Findings")

    st.markdown("""
    ### Findings from the Analysis

    *(Replace with your actual findings after running the analysis)*

    1. **Winning Party**: Identify which party/candidate won the constituency
    2. **Voter Turnout**: Overall turnout rate and how it varies by station
    3. **Invalid Ballots**: Stations with unusually high invalid ballot rates
    4. **Advance vs Day-of**: Whether advance voting patterns differ significantly
    5. **Anomalies**: Any stations flagged for unusual voting patterns
    6. **Historical Shift**: How results compare to the 2023 election
    """)

    # Display saved analysis figures
    for fig_path in sorted(FIGURES_DIR.glob("*.png")):
        st.image(str(fig_path), caption=fig_path.stem.replace("_", " ").title())


# ===== Footer =====
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Course:** 2110446 Data Science & Data Engineering\n\n"
    f"**Constituency:** {CONSTITUENCY_NAME}\n\n"
    f"**Data Source:** ECT (Election Commission of Thailand)"
)
```

---

## Execution Order (Run Commands)

```bash
# Step 0: Install dependencies
pip install -r requirements.txt

# Step 1: Download PDFs (or place manually in data/raw_pdfs/)
python 01_download/download_pdfs.py

# Step 2: Convert PDFs to preprocessed images
python 02_preprocess/preprocess_images.py

# Step 3: Run OCR pipeline
python 03_ocr/ocr_pipeline.py

# Step 4: Clean and validate data
python 04_clean/clean_data.py
python 04_clean/validate_data.py

# Step 5: Run analysis
python 05_analysis/analysis.py

# Step 6: Launch dashboard
streamlit run 06_dashboard/app.py
```

---

## Key Technical Notes

1. **OCR Engine Choice**: EasyOCR handles Thai handwritten digits better than Tesseract out of the box. PaddleOCR is the most accurate but heavier to install. Consider using multiple engines and taking the consensus result.

2. **Thai Font in Plots**: Install `TH Sarabun New` or use `Tahoma` font for proper Thai character rendering in matplotlib.

3. **Validation is Critical**: The cross-validation step (Phase 3B) is where you catch OCR errors. Compare summed station votes against constituency totals, and cross-reference with P'PanJ's cleaned data on GitHub.

4. **All 6 Forms Required**: You must OCR every form type (5/16, 5/16 party-list, 5/17, 5/17 party-list, 5/18, 5/18 party-list) for the entire constituency. Missing any form type will result in point deductions.

5. **Reference Data Sources**:
   - ECT: https://www.ect.go.th/ect_th/th/election-2026
   - Election areas: https://github.com/killernay/election-area-69
   - Election stations: https://github.com/killernay/election-station-69
   - OCR results reference: https://github.com/killernay/election-69-OCR-result
   - 2023 data: Google Drive link in project spec
