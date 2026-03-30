import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
import io

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Try to import groq
try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    raise ImportError("Run: pip install groq")

sys.path.insert(0, os.path.dirname(__file__))

# Model to use — Llama 4 vision hosted on Groq
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ─────────────────────────────────────────────
# HELPER — convert PIL Image to base64
# (Groq needs images as base64 strings)
# ─────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 encoded string."""
    # Resize to max 1024px on longest side to save API tokens
    max_size = 1024
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# CORE FUNCTION — Vision analysis with Qwen2-VL
# ─────────────────────────────────────────────

def analyse_with_vision(image: Image.Image) -> list[dict]:
    """
    Send a page image to Qwen2-VL and get back structured layout data.

    Args:
        image: PIL Image of one document page

    Returns:
        List of dicts with text, label, confidence, bbox
    """
    img_b64 = image_to_base64(image)

    # This is the prompt — we ask Qwen2-VL to extract layout
    # and return strict JSON so we can parse it deterministically
    prompt = """Analyze this document page image and extract all text elements.

For each text element found, return a JSON array with this exact structure:
[
  {
    "text": "the actual text content",
    "label": "HEADER or PARAGRAPH or TABLE_CELL or CAPTION or OTHER",
    "confidence": 0.95,
    "bbox": {
      "x_min": 100,
      "y_min": 50,
      "x_max": 400,
      "y_max": 80
    }
  }
]

Labeling rules:
- HEADER: titles, headings, section names (large or bold text)
- PARAGRAPH: body text, descriptions, multi-sentence blocks
- TABLE_CELL: individual cells in tables, short data values
- CAPTION: figure captions, footnotes, small descriptive text
- OTHER: page numbers, watermarks, decorative elements

Return ONLY the JSON array. No explanation, no markdown, no extra text."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.1  # low temperature = more deterministic output
        )

        raw_response = response.choices[0].message.content.strip()

        # Parse the JSON response
        results = parse_vision_response(raw_response)
        return results

    except Exception as e:
        print(f"Vision API error: {e}")
        return []


# ─────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────

def parse_vision_response(raw: str) -> list[dict]:
    """
    Safely parse the JSON response from Qwen2-VL.
    Handles cases where model adds extra text around the JSON.
    """
    # Strip markdown code blocks if model added them
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON array in response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                print("Could not parse vision response as JSON")
                return []
        else:
            return []

    # Validate and clean each item
    valid_labels = {"HEADER", "PARAGRAPH", "TABLE_CELL", "CAPTION", "OTHER"}
    cleaned = []

    for item in data:
        if not isinstance(item, dict):
            continue
        if "text" not in item or "bbox" not in item:
            continue

        # Ensure label is valid
        label = item.get("label", "OTHER").upper()
        if label not in valid_labels:
            label = "OTHER"

        # Ensure bbox has all fields
        bbox = item.get("bbox", {})
        if not all(k in bbox for k in ["x_min", "y_min", "x_max", "y_max"]):
            continue

        cleaned.append({
            "text": str(item["text"]).strip(),
            "label": label,
            "confidence": float(item.get("confidence", 0.9)),
            "bbox": {
                "x_min": int(bbox["x_min"]),
                "y_min": int(bbox["y_min"]),
                "x_max": int(bbox["x_max"]),
                "y_max": int(bbox["y_max"])
            }
        })

    return cleaned


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pdf_ingestor import pdf_to_images

    pdf_path = "sample_docs/AI_Engineer_Roadmap.pdf"

    print("Step 1: Converting PDF to image...")
    pages = pdf_to_images(pdf_path, dpi=150)

    print("Step 2: Sending page 1 to Qwen2-VL via Together.ai...")
    print("(This makes an API call — takes 10-30 seconds)\n")

    results = analyse_with_vision(pages[0]["image"])

    print(f"Extracted {len(results)} elements:\n")
    for item in results[:10]:
        print(f"  [{item['label']:12}] '{item['text'][:60]}'")
        print(f"    confidence: {item['confidence']}")
        print()