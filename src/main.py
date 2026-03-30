from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from pdf_ingestor import pdf_to_images
from ocr_engine import extract_text_with_boxes, reader
from layout_analyser import analyse_layout
from schema_builder import build_document_result

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Doc-Intel API",
    description = "Layout-aware document intelligence — turns PDFs into structured JSON",
    version     = "1.0.0"
)


# ─────────────────────────────────────────────
# STARTUP — pre-load OCR model once
# so first request isn't slow
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("Doc-Intel API starting up...")
    print("OCR engine ready.")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status"  : "running",
        "service" : "Doc-Intel API",
        "version" : "1.0.0",
        "docs"    : "/docs"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status"      : "healthy",
        "ocr_engine"  : "ready",
        "layout_model": "ready"
    }


@app.post("/parse")
async def parse_document(
    file        : UploadFile = File(..., description="PDF file to parse"),
    pages_limit : int        = 3,   # limit pages for speed in demo
    dpi         : int        = 150  # lower DPI = faster, still readable
):
    """
    Parse a PDF and return structured JSON.

    - **file**: PDF file upload
    - **pages_limit**: max pages to process (default 3)
    - **dpi**: rendering resolution (default 150)
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail      = "Only PDF files are supported"
        )

    start_time = time.time()

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ── Step 1: PDF → images ──
        print(f"Processing: {file.filename}")
        pages = pdf_to_images(tmp_path, dpi=dpi)

        # Limit pages for performance
        pages = pages[:pages_limit]

        # ── Step 2 & 3: OCR + Layout per page ──
        all_labelled_blocks = []

        for page in pages:
            ocr_results = extract_text_with_boxes(page["image"])
            labelled    = analyse_layout(page["image"], ocr_results)
            all_labelled_blocks.append(labelled)

        # ── Step 4: Build structured JSON ──
        result = build_document_result(
            filename            = file.filename,
            pages_data          = pages,
            all_labelled_blocks = all_labelled_blocks
        )

        elapsed = round(time.time() - start_time, 2)

        # Return as JSON with processing metadata
        response_data = {
            "metadata": {
                "filename"        : file.filename,
                "pages_processed" : len(pages),
                "processing_time" : f"{elapsed}s",
                "dpi_used"        : dpi
            },
            "result": result.model_dump()
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Processing failed: {str(e)}"
        )

    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)


@app.post("/parse/summary")
async def parse_summary(
    file: UploadFile = File(..., description="PDF file to parse")
):
    """
    Lightweight endpoint — returns just a summary, not full JSON.
    Useful for quickly checking what's in a document.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pages   = pdf_to_images(tmp_path, dpi=150)
        pages   = pages[:2]  # only first 2 pages for summary

        summary_pages = []

        for page in pages:
            ocr_results = extract_text_with_boxes(page["image"])
            labelled    = analyse_layout(page["image"], ocr_results)

            headers    = [b["text"] for b in labelled if b["label"] == "HEADER"]
            paragraphs = [b["text"] for b in labelled if b["label"] == "PARAGRAPH"]
            cells      = [b for b in labelled if b["label"] == "TABLE_CELL"]

            summary_pages.append({
                "page"            : page["page_number"],
                "headers"         : headers,
                "paragraph_count" : len(paragraphs),
                "table_cells"     : len(cells),
                "first_paragraph" : paragraphs[0] if paragraphs else None
            })

        return JSONResponse(content={
            "filename"    : file.filename,
            "total_pages" : len(pages),
            "pages"       : summary_pages
        })

    finally:
        os.unlink(tmp_path)