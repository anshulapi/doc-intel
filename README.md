# Doc-Intel — Layout-Aware Document Intelligence API

> Turns any PDF into deterministic, machine-readable JSON using computer vision, OCR, and layout-aware AI.

Built as part of my ML engineering portfolio, inspired by Microsoft's **LayoutLMv3** paper (2022).

---

## What it does

Most tools extract raw text from PDFs. Doc-Intel understands **layout** — it knows the difference between a heading, a paragraph, and a table cell, and returns everything as structured, validated JSON.
```
PDF → page images → OCR (text + coordinates) → layout classification → structured JSON
```

---

## Live API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed status |
| POST | `/parse` | Full document parsing |
| POST | `/parse/summary` | Lightweight summary |

---

## Example output
```json
{
  "metadata": {
    "filename": "invoice.pdf",
    "pages_processed": 3,
    "processing_time": "47s"
  },
  "result": {
    "pages": [{
      "page_number": 1,
      "headers": [
        {
          "text": "AI Engineer Roadmap",
          "label": "HEADER",
          "confidence": 0.97,
          "bbox": { "x_min": 451, "y_min": 189, "x_max": 1207, "y_max": 257 }
        }
      ],
      "tables": [{
        "rows": [
          { "cells": ["Python", "PyTorch", "HuggingFace"] }
        ]
      }]
    }]
  }
}
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| PDF ingestion | PyMuPDF |
| OCR | EasyOCR |
| Layout analysis | Geometric heuristics (LayoutLMv3-inspired) |
| Schema validation | Pydantic v2 |
| API | FastAPI + Uvicorn |
| Language | Python 3.10 |
| Layout analysis | Llama 4 Vision via Groq (multimodal) |

---

## Research foundation

This project implements concepts from:
- **LayoutLMv3** (Microsoft, 2022) — multimodal document understanding combining text, layout, and image
- **FUNSD dataset** — form understanding benchmark used for layout classification evaluation

---

## Run locally
```bash
# Clone
git clone https://github.com/anshulapi/doc-intel.git
cd doc-intel

# Setup
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run API
uvicorn src.main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## Project structure
```
doc-intel/
  src/
    pdf_ingestor.py     # PDF → page images
    ocr_engine.py       # images → text + bounding boxes
    layout_analyser.py  # text blocks → HEADER/PARAGRAPH/TABLE_CELL
    schema_builder.py   # everything → validated Pydantic JSON
    main.py             # FastAPI REST API
  sample_docs/          # test PDFs
  README.md
```
## Benchmark — Heuristic vs Vision Model

Benchmarked on a 2-page document (AI Engineer Roadmap PDF):

| Metric | Heuristic | Llama 4 Vision |
|--------|-----------|----------------|
| Total processing time | 34.5s | 11.49s |
| Average confidence | 0.836 | 0.95 |
| Complex layout accuracy | Medium | High |
| Cost | Free (local) | API credits |
| Explainability | Full | Black box |
| Requires OCR step | Yes | No |

> Vision model is **3x faster** and **13.5% more confident** than the heuristic approach on this document. Heuristic remains useful as a free offline fallback.

**Our pipeline uses Vision first, Heuristic as fallback — best of both worlds.**
## Roadmap

- [ ] Qwen2-VL multimodal model integration (in progress)
- [ ] AMD ROCm GPU deployment
- [ ] Multi-language OCR support
- [ ] Table structure reconstruction improvements
- [ ] Docker containerization

---

*Built with Python 3.10 · FastAPI · EasyOCR · PyMuPDF · Pydantic*