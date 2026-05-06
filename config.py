"""
Global configuration for the Election OCR/Data Science project.

Secrets are loaded from environment variables or a local .env file.
Never commit real API keys to GitHub.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
PROVINCE = None               # Thai province name (None = do not filter)
CONSTITUENCY_NUMBER = 3               # Constituency number
CONSTITUENCY_NAME = f"{PROVINCE}เขต{CONSTITUENCY_NUMBER}"

# ===== ECT DATA SOURCE =====
ECT_BASE_URL = "https://www.ect.go.th/ect_th/th/election-2026"

# ===== FORM TYPES (all 6 required) =====
FORM_TYPES = ["election"]  # All PDFs from D:\ocr\ocr\3, converted to PNGs

# ===== OCR SETTINGS =====
OCR_ENGINE = "typhoon"  # Options: "tesseract", "easyocr", "gemini", "typhoon"
OCR_LANGUAGES = ["th", "en"]
OCR_CONFIDENCE_THRESHOLD = 0.6
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "")

# ===== PARTY NAME MAPPING =====
# Official party list for the 2026 election (update as needed)
PARTY_NAMES = {
    7: "พลวัต",
    8: "ประชาธิปไตยใหม่",
    9: "เพื่อไทย",
    10: "ทางเลือกใหม่",
    11: "เศรษฐกิจ",
    12: "เสรีรวมไทย",
    13: "รวมพลังประชาชน",
    14: "ท้องที่ไทย",
    16: "พลังเพื่อไทย",
    17: "ไทยชนะ",
    27: "ประชาธิปัตย์",
    28: "ไทยก้าวหน้า",
    29: "ไทยภักดี",
    30: "แรงงานสร้างชาติ",
    31: "ประชากรไทย",
    33: "ประชาชาติ",
    35: "รักชาติ",
    36: "ไทยพร้อม",
    37: "ภูมิใจไทย",
    40: "ไทยธรรม",
    41: "แผ่นดินธรรม",
    46: "ประชาชน",
}

# Create directories
for d in [RAW_PDF_DIR, IMAGE_DIR, OCR_RAW_DIR, CLEANED_DIR,
          REFERENCE_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
