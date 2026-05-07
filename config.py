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
# Official party-list numbers from data/reference/party_reference.csv.
PARTY_NAMES = {
    1: "ไทยทรัพย์ทวี",
    2: "เพื่อชาติไทย",
    3: "ใหม่",
    4: "มิติใหม่",
    5: "รวมใจไทย",
    6: "รวมไทยสร้างชาติ",
    7: "พลวัต",
    8: "ประชาธิปไตยใหม่",
    9: "เพื่อไทย",
    10: "ทางเลือกใหม่",
    11: "เศรษฐกิจ",
    12: "เสรีรวมไทย",
    13: "รวมพลังประชาชน",
    14: "ท้องที่ไทย",
    15: "อนาคตไทย",
    16: "พลังเพื่อไทย",
    17: "ไทยชนะ",
    18: "พลังสังคมใหม่",
    19: "สังคมประชาธิปไตยไทย",
    20: "ฟิวชัน",
    21: "ไทรวมพลัง",
    22: "ก้าวอิสระ",
    23: "ปวงชนไทย",
    24: "วิชชั่นใหม่",
    25: "เพื่อชีวิตใหม่",
    26: "คลองไทย",
    27: "ประชาธิปัตย์",
    28: "ไทยก้าวหน้า",
    29: "ไทยภักดี",
    30: "แรงงานสร้างชาติ",
    31: "ประชากรไทย",
    32: "ครูไทยเพื่อประชาชน",
    33: "ประชาชาติ",
    34: "สร้างอนาคตไทย",
    35: "รักชาติ",
    36: "ไทยพร้อม",
    37: "ภูมิใจไทย",
    38: "พลังธรรมใหม่",
    39: "กรีน",
    40: "ไทยธรรม",
    41: "แผ่นดินธรรม",
    42: "กล้าธรรม",
    43: "พลังประชารัฐ",
    44: "โอกาสใหม่",
    45: "เป็นธรรม",
    46: "ประชาชน",
    47: "ประชาไทย",
    48: "ไทยสร้างไทย",
    49: "ไทยก้าวใหม่",
    50: "ประชาอาสาชาติ",
    51: "พร้อม",
    52: "เครือข่ายชาวนาแห่งประเทศไทย",
    53: "ไทยพิทักษ์ธรรม",
    54: "ความหวังใหม่",
    55: "ไทยรวมไทย",
    56: "เพื่อบ้านเมือง",
    57: "พลังไทยรักชาติ",
}

# Create directories
for d in [RAW_PDF_DIR, IMAGE_DIR, OCR_RAW_DIR, CLEANED_DIR,
          REFERENCE_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
