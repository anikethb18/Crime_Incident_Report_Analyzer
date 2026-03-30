"""
PDF Processor Module
Student 1 — Responsibility: Extract structured incident data from PDF reports.

Handles BOTH:
  - Text-based PDFs  → pdfplumber (fast, accurate)
  - Scanned/image PDFs → OCR pipeline (pdf2image + Tesseract pytesseract)

Full pipeline:
  1. Try pdfplumber text extraction first
  2. If text yield is low (< 50 chars/page), classify as scanned PDF
  3. For scanned PDFs: convert each page to image → run Tesseract OCR
  4. Send combined text to GPT-4o for structured field extraction
  5. Return normalized incident record
"""

import os
import json
import uuid
import pdfplumber
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Threshold: if average chars per page < this, treat PDF as scanned
SCANNED_THRESHOLD_CHARS_PER_PAGE = 50

EXTRACTION_PROMPT = """
You are an expert police incident report analyst.
Given the raw text extracted from a PDF incident report (may be OCR output from a scanned document),
extract the following fields and return ONLY valid JSON.

Fields to extract:
- date (YYYY-MM-DD format, or null if not found)
- time (HH:MM 24h format, or null if not found)
- location (full address or description, or null)
- incident_type (e.g. Robbery, Assault, Burglary, Vandalism, etc.)
- description (brief 1-2 sentence summary)
- suspects (list of names/descriptions, empty list if none)
- victims (list of names/descriptions, empty list if none)
- evidence (list of evidence items mentioned, empty list if none)
- officer (reporting officer name, or null)
- status (open | closed | under_investigation)
- confidence_score (float 0.0-1.0 — lower if text looks like noisy OCR output)

Return ONLY the JSON object. No explanation, no markdown, no preamble.

PDF Text:
{text}
"""


# ── Step 1: Text-based extraction (pdfplumber) ────────────────────────────────

def extract_text_pdfplumber(pdf_path: str) -> tuple[str, int]:
    """
    Extract text from a PDF using pdfplumber.
    Returns (extracted_text, page_count).
    """
    full_text = []
    page_count = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    full_text.append(f"[Page {page_num}]\n{text.strip()}")
                # Extract embedded tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join([cell or "" for cell in row if cell])
                        if row_text.strip():
                            full_text.append(row_text)
    except Exception as e:
        print(f"[PDF] pdfplumber error: {e}")
    return "\n".join(full_text), page_count


def is_scanned_pdf(text: str, page_count: int) -> bool:
    """
    Heuristic: if average extracted characters per page is below threshold,
    the PDF is likely a scanned image document with no embedded text layer.
    """
    if page_count == 0:
        return True
    avg_chars = len(text.strip()) / page_count
    if avg_chars < SCANNED_THRESHOLD_CHARS_PER_PAGE:
        print(f"[PDF] Scanned PDF detected (avg {avg_chars:.0f} chars/page < threshold {SCANNED_THRESHOLD_CHARS_PER_PAGE})")
        return True
    return False


# ── Step 2: OCR pipeline for scanned PDFs ─────────────────────────────────────

def ocr_pdf_with_tesseract(pdf_path: str) -> str:
    """
    OCR pipeline for scanned/image-based PDFs:
      1. Convert each PDF page to a PIL Image using pdf2image (wraps pdftoppm/Poppler)
      2. Pre-process image with Pillow (grayscale + threshold) for better OCR accuracy
      3. Run Tesseract OCR via pytesseract on each page image
      4. Concatenate all page texts

    Dependencies (must be installed):
      pip install pdf2image pytesseract Pillow
      System: sudo apt-get install tesseract-ocr poppler-utils   (Linux)
              brew install tesseract poppler                       (macOS)
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as e:
        print(f"[PDF-OCR] Missing dependency: {e}")
        print("[PDF-OCR] Install with: pip install pdf2image pytesseract Pillow")
        print("[PDF-OCR] System: sudo apt-get install tesseract-ocr poppler-utils")
        return ""

    print(f"[PDF-OCR] Converting PDF pages to images...")
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"[PDF-OCR] pdf2image error: {e}")
        print("[PDF-OCR] Ensure poppler-utils is installed (sudo apt-get install poppler-utils)")
        return ""

    print(f"[PDF-OCR] Running Tesseract OCR on {len(pages)} page(s)...")
    all_text = []

    for page_num, page_img in enumerate(pages, 1):
        try:
            # Pre-processing for better OCR accuracy:
            # 1. Convert to greyscale
            grey = page_img.convert("L")
            # 2. Increase contrast / apply threshold (binarize)
            #    Pixels > 180 → white (255), else → black (0)
            threshold = grey.point(lambda px: 255 if px > 180 else 0, "L")
            # 3. Slight sharpening to improve character edges
            sharpened = threshold.filter(ImageFilter.SHARPEN)

            # Run Tesseract with English language + page segmentation mode 6
            # PSM 6 = "Assume a single uniform block of text"
            custom_config = r"--oem 3 --psm 6 -l eng"
            page_text = pytesseract.image_to_string(sharpened, config=custom_config)

            if page_text.strip():
                all_text.append(f"[Page {page_num} — OCR]\n{page_text.strip()}")
                print(f"[PDF-OCR] Page {page_num}: {len(page_text)} chars extracted")
            else:
                print(f"[PDF-OCR] Page {page_num}: no text detected")

        except Exception as e:
            print(f"[PDF-OCR] OCR error on page {page_num}: {e}")

    combined = "\n\n".join(all_text)
    print(f"[PDF-OCR] Total OCR output: {len(combined)} characters")
    return combined


# ── Step 3: Smart extraction router ───────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> tuple[str, str]:
    """
    Smart router: tries text extraction first, falls back to OCR for scanned PDFs.

    Returns:
        (extracted_text, method_used)
        method_used: "text" | "ocr" | "failed"
    """
    # Try text-based extraction
    text, page_count = extract_text_pdfplumber(pdf_path)

    if text.strip() and not is_scanned_pdf(text, page_count):
        print(f"[PDF] Text extraction succeeded: {len(text)} chars from {page_count} page(s)")
        return text, "text"

    # Fall back to OCR
    print(f"[PDF] Falling back to Tesseract OCR...")
    ocr_text = ocr_pdf_with_tesseract(pdf_path)

    if ocr_text.strip():
        return ocr_text, "ocr"

    # Both failed
    print(f"[PDF] Both text extraction and OCR failed for: {pdf_path}")
    return "", "failed"


# ── Step 4: LLM structured extraction ────────────────────────────────────────

def parse_with_llm(raw_text: str) -> dict:
    """Send extracted text to GPT-4o to parse structured incident fields."""
    if not raw_text.strip():
        return {}

    prompt = EXTRACTION_PROMPT.format(text=raw_text[:4000])

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[PDF] JSON parse error: {e}")
        return {}
    except Exception as e:
        print(f"[PDF] LLM error: {e}")
        return {}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str) -> dict:
    """
    Full pipeline: PDF → smart text extraction (text or OCR) → GPT-4o → structured record.

    Automatically detects scanned PDFs and applies Tesseract OCR when needed.

    Args:
        pdf_path: Path to PDF file (text-based or scanned/image)

    Returns:
        Structured incident dict conforming to shared schema
    """
    print(f"[PDF] Processing: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"[PDF] File not found: {pdf_path}")
        return _empty_record(pdf_path)

    # Extract text (with OCR fallback)
    raw_text, method = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        print("[PDF] No text could be extracted (text or OCR)")
        record = _empty_record(pdf_path)
        record["extraction_method"] = "failed"
        return record

    print(f"[PDF] Extraction method: {method.upper()} — {len(raw_text)} chars")

    # Parse with LLM
    parsed = parse_with_llm(raw_text)

    # Build final record
    record = {
        "incident_id":       f"INC-PDF-{str(uuid.uuid4())[:8].upper()}",
        "source_modality":   "pdf",
        "source_file":       os.path.basename(pdf_path),
        "processed_at":      datetime.now().isoformat(),
        "extraction_method": method,   # "text" | "ocr"
        "date":              parsed.get("date"),
        "time":              parsed.get("time"),
        "location":          parsed.get("location"),
        "incident_type":     parsed.get("incident_type", "Unknown"),
        "description":       parsed.get("description", ""),
        "suspects":          parsed.get("suspects", []),
        "victims":           parsed.get("victims", []),
        "evidence":          parsed.get("evidence", []),
        "officer":           parsed.get("officer"),
        "status":            parsed.get("status", "open"),
        "confidence_score":  parsed.get("confidence_score", 0.0),
    }

    # Slightly reduce confidence for OCR output (noisier text)
    if method == "ocr":
        record["confidence_score"] = round(min(record["confidence_score"], 0.80), 3)
        print(f"[PDF] OCR confidence capped at 0.80 due to potential noise")

    print(f"[PDF] Done — method: {method}, incident_type: {record['incident_type']}, "
          f"confidence: {record['confidence_score']}")
    return record


def _empty_record(pdf_path: str = "") -> dict:
    return {
        "incident_id":       f"INC-PDF-{str(uuid.uuid4())[:8].upper()}",
        "source_modality":   "pdf",
        "source_file":       os.path.basename(pdf_path) if pdf_path else "",
        "processed_at":      datetime.now().isoformat(),
        "extraction_method": "failed",
        "date":              None,
        "time":              None,
        "location":          None,
        "incident_type":     "Unknown",
        "description":       "Extraction failed",
        "suspects":          [],
        "victims":           [],
        "evidence":          [],
        "officer":           None,
        "status":            "open",
        "confidence_score":  0.0,
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python processor.py <path_to_pdf>")
        print("Works with both text-based and scanned/image PDFs.")
        sys.exit(1)
    result = process_pdf(sys.argv[1])
    print(json.dumps(result, indent=2))
