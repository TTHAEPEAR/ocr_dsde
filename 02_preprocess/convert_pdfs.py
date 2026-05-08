r"""
Convert election PDFs from D:\ocr\ocr\3 to PNG images.
Skips the LAST page of each PDF (signature page -- no vote data).

Output goes to data/images/election/ with one PNG per page.
"""
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import sys, re, hashlib

sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGE_DIR, RAW_PDF_DIR

# ===== CONFIG =====
PDF_ROOT = RAW_PDF_DIR
OUT_DIR = IMAGE_DIR / "election"
DPI = 300  # Good balance of quality vs file size
SKIP_LAST_PAGE = True  # Skip signature page

OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_stem(path: Path) -> str:
    """Create a unique, safe filename from the full relative path.
    
    Uses the FULL folder hierarchy to avoid collisions.
    Example: อำเภอลอง/.../ต.ต้าผามอก/หน่วยที่ 1/ส.ส.5_18.pdf
          -> amphoe_long_t_taphamok_unit1_ss5_18
    """
    rel = path.relative_to(PDF_ROOT)
    # Join all path parts (folders + filename without extension)
    full = "_".join(rel.with_suffix("").parts)
    # Replace non-alphanumeric (keeping Thai chars) with underscore
    safe = re.sub(r"[^\w]", "_", full, flags=re.UNICODE)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe).strip("_")
    # Truncate if too long but keep unique by appending short hash
    if len(safe) > 200:
        h = hashlib.md5(safe.encode()).hexdigest()[:8]
        safe = safe[:190] + "_" + h
    return safe


def convert_all():
    pdfs = sorted(PDF_ROOT.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {PDF_ROOT}")

    total_pages = 0
    skipped_pages = 0
    converted = 0
    already_done = 0

    for pdf_path in tqdm(pdfs, desc="Converting PDFs"):
        stem = safe_stem(pdf_path)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"  ERROR opening {pdf_path.name}: {e}")
            continue

        n_pages = doc.page_count
        pages_to_render = n_pages - 1 if (SKIP_LAST_PAGE and n_pages > 1) else n_pages

        for page_idx in range(pages_to_render):
            out_name = f"{stem}_page{page_idx + 1}.png"
            out_path = OUT_DIR / out_name

            if out_path.exists():
                already_done += 1
                total_pages += 1
                continue

            page = doc[page_idx]
            mat = fitz.Matrix(DPI / 72, DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            pix.save(str(out_path))
            converted += 1
            total_pages += 1

        skipped_pages += (n_pages - pages_to_render)
        doc.close()

    print(f"\n=== Conversion Complete ===")
    print(f"  PDFs processed:     {len(pdfs)}")
    print(f"  Pages converted:    {converted}")
    print(f"  Already existed:    {already_done}")
    print(f"  Signature pages skipped: {skipped_pages}")
    print(f"  Total useful pages: {total_pages}")
    print(f"  Output directory:   {OUT_DIR}")


if __name__ == "__main__":
    convert_all()
