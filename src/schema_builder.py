from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
import json, sys, os

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS — define the exact shape
# of our output JSON
# ─────────────────────────────────────────────

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


class TextBlock(BaseModel):
    text: str
    label: str                          # HEADER, PARAGRAPH, TABLE_CELL, etc.
    confidence: float = Field(ge=0.0, le=1.0)  # must be between 0 and 1
    bbox: BoundingBox


class TableRow(BaseModel):
    cells: list[str]


class Table(BaseModel):
    rows: list[TableRow]
    bbox: BoundingBox                   # bounding box of the whole table


class PageResult(BaseModel):
    page_number: int
    width: int
    height: int
    headers: list[TextBlock]            # all HEADER blocks
    paragraphs: list[TextBlock]         # all PARAGRAPH blocks
    tables: list[Table]                 # reconstructed tables
    other_blocks: list[TextBlock]       # everything else
    total_text_blocks: int


class DocumentResult(BaseModel):
    filename: str
    total_pages: int
    pages: list[PageResult]


# ─────────────────────────────────────────────
# TABLE RECONSTRUCTION
# Group TABLE_CELL blocks into rows and tables
# using their vertical (y) positions
# ─────────────────────────────────────────────

def reconstruct_tables(
    table_cells: list[dict],
    row_tolerance: int = 20
) -> list[Table]:
    """
    Group individual TABLE_CELL blocks into rows and tables.

    Strategy:
    - Cells with similar y_min (within tolerance) = same row
    - Sort rows by y position
    - Group nearby rows into one table
    """
    if not table_cells:
        return []

    # Sort cells top to bottom, then left to right
    sorted_cells = sorted(
        table_cells,
        key=lambda c: (c["bbox"]["y_min"], c["bbox"]["x_min"])
    )

    # Group cells into rows based on y proximity
    rows = []
    current_row = [sorted_cells[0]]

    for cell in sorted_cells[1:]:
        last_y = current_row[-1]["bbox"]["y_min"]
        curr_y = cell["bbox"]["y_min"]

        if abs(curr_y - last_y) <= row_tolerance:
            # Same row
            current_row.append(cell)
        else:
            # New row
            rows.append(sorted(current_row, key=lambda c: c["bbox"]["x_min"]))
            current_row = [cell]

    rows.append(sorted(current_row, key=lambda c: c["bbox"]["x_min"]))

    # Build Table object from rows
    if not rows:
        return []

    # Calculate bounding box of the whole table
    all_x_min = min(c["bbox"]["x_min"] for row in rows for c in row)
    all_y_min = min(c["bbox"]["y_min"] for row in rows for c in row)
    all_x_max = max(c["bbox"]["x_max"] for row in rows for c in row)
    all_y_max = max(c["bbox"]["y_max"] for row in rows for c in row)

    table = Table(
        rows=[TableRow(cells=[c["text"] for c in row]) for row in rows],
        bbox=BoundingBox(
            x_min=all_x_min,
            y_min=all_y_min,
            x_max=all_x_max,
            y_max=all_y_max
        )
    )

    return [table]


# ─────────────────────────────────────────────
# MAIN BUILDER FUNCTION
# Takes labelled blocks → returns DocumentResult
# ─────────────────────────────────────────────

def build_page_result(
    page_data: dict,
    labelled_blocks: list[dict]
) -> PageResult:
    """
    Convert labelled OCR blocks into a structured PageResult.

    Args:
        page_data: dict with page_number, width, height
        labelled_blocks: output from layout_analyser.analyse_layout()
    """
    headers      = []
    paragraphs   = []
    table_cells  = []
    other_blocks = []

    for block in labelled_blocks:
        # Build a clean TextBlock
        text_block = TextBlock(
            text       = block["text"],
            label      = block["label"],
            confidence = block["confidence"],
            bbox       = BoundingBox(**block["bbox"])
        )

        if block["label"] == "HEADER":
            headers.append(text_block)
        elif block["label"] == "PARAGRAPH":
            paragraphs.append(text_block)
        elif block["label"] == "TABLE_CELL":
            table_cells.append(block)   # keep raw for table reconstruction
        else:
            other_blocks.append(text_block)

    # Reconstruct tables from individual cells
    tables = reconstruct_tables(table_cells)

    return PageResult(
        page_number       = page_data["page_number"],
        width             = page_data["width"],
        height            = page_data["height"],
        headers           = headers,
        paragraphs        = paragraphs,
        tables            = tables,
        other_blocks      = other_blocks,
        total_text_blocks = len(labelled_blocks)
    )


def build_document_result(
    filename: str,
    pages_data: list[dict],
    all_labelled_blocks: list[list[dict]]
) -> DocumentResult:
    """
    Build the final DocumentResult for the whole PDF.

    Args:
        filename: name of the PDF file
        pages_data: list of page dicts from pdf_ingestor
        all_labelled_blocks: list of labelled blocks per page
    """
    page_results = []

    for page_data, labelled_blocks in zip(pages_data, all_labelled_blocks):
        page_result = build_page_result(page_data, labelled_blocks)
        page_results.append(page_result)

    return DocumentResult(
        filename    = filename,
        total_pages = len(page_results),
        pages       = page_results
    )


if __name__ == "__main__":
    from pdf_ingestor import pdf_to_images
    from ocr_engine import extract_text_with_boxes
    from layout_analyser import analyse_layout

    pdf_path = "sample_docs/AI_Engineer_Roadmap.pdf"

    print("Running full pipeline on page 1 only (for speed)...\n")

    # Step 1: PDF to images
    pages = pdf_to_images(pdf_path, dpi=200)

    # Step 2 & 3: OCR + Layout on first page only
    page        = pages[0]
    ocr_results = extract_text_with_boxes(page["image"])
    labelled    = analyse_layout(page["image"], ocr_results)

    # Step 4: Build JSON
    result = build_document_result(
        filename            = "AI_Engineer_Roadmap.pdf",
        pages_data          = [page],
        all_labelled_blocks = [labelled]
    )

    # Save to JSON file
    output_path = "sample_docs/output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    print(f"✅ JSON saved to {output_path}\n")
    print(f"Pages processed : {result.total_pages}")
    print(f"Headers found   : {len(result.pages[0].headers)}")
    print(f"Paragraphs found: {len(result.pages[0].paragraphs)}")
    print(f"Tables found    : {len(result.pages[0].tables)}")

    # Preview the JSON
    print(f"\nFirst header:")
    if result.pages[0].headers:
        print(f"  {result.pages[0].headers[0].model_dump()}")

    print(f"\nFirst paragraph:")
    if result.pages[0].paragraphs:
        print(f"  {result.pages[0].paragraphs[0].model_dump()}")