"""
Phase 1C: Convert PDF documents to preprocessed images for OCR.
Applies deskewing, denoising, and binarization for better OCR accuracy.

Usage:
    python 02_preprocess/preprocess_images.py
"""
import cv2
import numpy as np
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image
from tqdm import tqdm
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_PDF_DIR, IMAGE_DIR, FORM_TYPES


class PDFPreprocessor:
    """Converts PDFs to clean images optimized for OCR."""

    def __init__(self, dpi: int = 400):
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
            logger.info(f"Converting {len(pdfs)} PDFs for form {form_type} at {self.dpi} DPI")

            for pdf_path in tqdm(pdfs, desc=f"Form {form_type}"):
                self._convert_single_pdf(pdf_path, img_dir)

    def _convert_single_pdf(self, pdf_path: Path, output_dir: Path):
        """Convert a single PDF to preprocessed image(s)."""
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))

            for page_num in range(len(pdf)):
                page = pdf[page_num]
                # Render with high quality
                bitmap = page.render(scale=self.dpi / 72.0)
                page_img = bitmap.to_pil()

                # Convert PIL → OpenCV format
                img_array = np.array(page_img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # For Gemini, we want high-quality original images, not binarized ones
                # We only do basic grayscale + deskew to keep it clean
                from config import OCR_ENGINE
                if OCR_ENGINE == "gemini":
                    # Just grayscale and deskew, don't binarize!
                    processed = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    processed = self._deskew(processed)
                else:
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
        Full preprocessing pipeline for traditional OCR (Tesseract/EasyOCR).
        """
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Deskew
        gray = self._deskew(gray)

        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=5) # Reduced h to keep details

        # 4. Adaptive threshold (Keep it but make it less aggressive)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, # Larger block size for smoother text
            C=10
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
