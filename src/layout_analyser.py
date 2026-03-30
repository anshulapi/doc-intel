from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def analyse_layout(image: Image.Image, ocr_results: list[dict]) -> list[dict]:
    """
    Classify each text block by its layout role using bounding box geometry.
    Labels: HEADER, PARAGRAPH, TABLE_CELL, CAPTION, OTHER

    Strategy:
    - Text height (bbox height) estimates font size
    - Position on page (y ratio) tells us where it is
    - Width ratio tells us if it spans the full page (paragraph) or is narrow (cell/label)
    - Confidence threshold filters noise
    """
    if not ocr_results:
        return []

    page_width, page_height = image.size

    # Filter out low confidence noise first
    clean_results = [r for r in ocr_results if r["confidence"] > 0.4]

    # Calculate heights of all text blocks to find relative sizes
    heights = []
    for item in clean_results:
        h = item["bbox"]["y_max"] - item["bbox"]["y_min"]
        heights.append(h)

    if not heights:
        return ocr_results

    avg_height   = sum(heights) / len(heights)
    max_height   = max(heights)

    labelled = []

    for item in clean_results:
        bbox       = item["bbox"]
        text       = item["text"].strip()

        # Geometric features
        box_height = bbox["y_max"] - bbox["y_min"]
        box_width  = bbox["x_max"] - bbox["x_min"]
        y_position = bbox["y_min"] / page_height   # 0 = top, 1 = bottom
        width_ratio = box_width / page_width        # 0 = narrow, 1 = full width

        label = classify_block(
            text        = text,
            box_height  = box_height,
            box_width   = box_width,
            y_position  = y_position,
            width_ratio = width_ratio,
            avg_height  = avg_height,
            max_height  = max_height,
            page_width  = page_width,
        )

        labelled.append({
            **item,
            "label": label
        })

    return labelled


def classify_block(
    text, box_height, box_width,
    y_position, width_ratio,
    avg_height, max_height, page_width
) -> str:
    """
    Rule-based classification of a single text block.
    Returns one of: HEADER, PARAGRAPH, TABLE_CELL, CAPTION, OTHER
    """

    # Skip empty or single character text
    if not text or len(text) < 2:
        return "OTHER"

    # --- HEADER detection ---
    # Large text (bigger than average) near the top of the page
    is_large_text  = box_height > avg_height * 1.4
    is_upper_half  = y_position < 0.35
    is_all_caps    = text.isupper() and len(text) > 3
    is_short_text  = len(text.split()) <= 8

    if is_large_text and is_short_text:
        return "HEADER"
    if is_all_caps and is_upper_half and is_short_text:
        return "HEADER"

    # --- TABLE CELL detection ---
    # Narrow blocks — not spanning much of the page width
    # and text is short (single word or number)
    is_narrow      = width_ratio < 0.25
    is_short_word  = len(text.split()) <= 3
    looks_like_num = any(c.isdigit() for c in text)

    if is_narrow and is_short_word:
        return "TABLE_CELL"
    if looks_like_num and is_narrow:
        return "TABLE_CELL"

    # --- CAPTION detection ---
    # Small text below average height, often below figures
    is_small_text  = box_height < avg_height * 0.85
    is_lower_half  = y_position > 0.65
    starts_fig     = text.lower().startswith(("fig", "figure", "table", "note", "source"))

    if starts_fig:
        return "CAPTION"
    if is_small_text and is_lower_half and is_short_text:
        return "CAPTION"

    # --- PARAGRAPH detection ---
    # Wide blocks with multiple words
    is_wide        = width_ratio > 0.4
    has_many_words = len(text.split()) > 5

    if is_wide and has_many_words:
        return "PARAGRAPH"

    return "OTHER"


if __name__ == "__main__":
    from pdf_ingestor import pdf_to_images
    from ocr_engine import extract_text_with_boxes
    from collections import Counter

    pdf_path = "sample_docs/AI_Engineer_Roadmap.pdf"

    print("Step 1: Converting PDF to images...")
    pages = pdf_to_images(pdf_path, dpi=200)

    print("Step 2: Running OCR on page 1...")
    ocr_results = extract_text_with_boxes(pages[0]["image"])
    print(f"Found {len(ocr_results)} text blocks\n")

    print("Step 3: Analysing layout (heuristic)...")
    labelled = analyse_layout(pages[0]["image"], ocr_results)

    print(f"\nLayout analysis results:\n")
    for item in labelled:
        print(f"  [{item['label']:12}] '{item['text']}'")

    # Summary
    labels = [item["label"] for item in labelled]
    counts = Counter(labels)
    print(f"\nSummary:")
    for label, count in counts.items():
        print(f"  {label:12}: {count} blocks")