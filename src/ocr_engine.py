import easyocr
import numpy as np
from PIL import Image

# Initialize the OCR reader once (loading it is slow, so we do it once)
# ['en'] means English. You can add more languages like ['en', 'hi'] for Hindi
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_with_boxes(image: Image.Image) -> list[dict]:
    """
    Run OCR on a PIL Image and return every word with its position.

    Args:
        image: PIL Image object (one page from pdf_ingestor)

    Returns:
        List of dicts, one per detected text block:
        {
            "text": "Invoice",
            "confidence": 0.98,
            "bbox": {
                "x_min": 100, "y_min": 50,
                "x_max": 200, "y_max": 80
            }
        }
    """
    # EasyOCR needs a numpy array, not a PIL Image
    img_array = np.array(image)

    # Run OCR — this is the core call
    # detail=1 means return bounding boxes too (not just text)
    results = reader.readtext(img_array, detail=1)

    extracted = []

    for (bbox_points, text, confidence) in results:
        # EasyOCR returns 4 corner points of the text box
        # bbox_points looks like: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        # We convert to simple x_min, y_min, x_max, y_max format
        x_coords = [pt[0] for pt in bbox_points]
        y_coords = [pt[1] for pt in bbox_points]

        extracted.append({
            "text": text.strip(),
            "confidence": round(float(confidence), 4),
            "bbox": {
                "x_min": int(min(x_coords)),
                "y_min": int(min(y_coords)),
                "x_max": int(max(x_coords)),
                "y_max": int(max(y_coords))
            }
        })

    # Sort top to bottom, left to right (reading order)
    extracted.sort(key=lambda x: (x["bbox"]["y_min"], x["bbox"]["x_min"]))

    return extracted


if __name__ == "__main__":
    # Test using the preview image we saved in Module 1
    from pathlib import Path

    preview_path = "sample_docs/page_1_preview.png"

    if not Path(preview_path).exists():
        print("Run pdf_ingestor.py first to generate page_1_preview.png")
    else:
        print("Loading image...")
        image = Image.open(preview_path).convert("RGB")

        print("Running OCR (first run downloads model ~100MB, be patient)...")
        results = extract_text_with_boxes(image)

        print(f"\nFound {len(results)} text blocks:\n")
        for item in results[:10]:  # Show first 10 results
            print(f"  '{item['text']}'")
            print(f"    confidence: {item['confidence']}")
            print(f"    position: {item['bbox']}")
            print()