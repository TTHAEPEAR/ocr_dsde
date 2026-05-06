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
    pdf_links = downloader.discover_pdf_links()
    downloader.download_all(pdf_links)

    # Option B: Manual download — place PDFs in data/raw_pdfs/{form_type}/
    # Then just run verification:
    downloader.verify_completeness()
