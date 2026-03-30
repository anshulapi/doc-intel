import time
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from pdf_ingestor import pdf_to_images
from ocr_engine import extract_text_with_boxes
from layout_analyser import analyse_layout
from vision_analyser import analyse_with_vision
from collections import Counter


def benchmark_page(image: Image.Image, page_num: int):
    """
    Run both analysers on the same page and compare results.
    """
    print(f"\n{'='*50}")
    print(f"PAGE {page_num} BENCHMARK")
    print(f"{'='*50}")

    # ── HEURISTIC ──
    print("\n[1] Heuristic analyser...")
    h_start = time.time()
    ocr_results = extract_text_with_boxes(image)
    heuristic_results = analyse_layout(image, ocr_results)
    h_time = round(time.time() - h_start, 2)

    h_labels = Counter([r["label"] for r in heuristic_results])
    h_avg_conf = round(
        sum(r["confidence"] for r in heuristic_results) / max(len(heuristic_results), 1), 3
    )

    print(f"  Time taken     : {h_time}s")
    print(f"  Blocks found   : {len(heuristic_results)}")
    print(f"  Avg confidence : {h_avg_conf}")
    print(f"  Label breakdown: {dict(h_labels)}")

    # ── VISION ──
    print("\n[2] Vision analyser (Llama 4)...")
    v_start = time.time()
    vision_results = analyse_with_vision(image)
    v_time = round(time.time() - v_start, 2)

    v_labels = Counter([r["label"] for r in vision_results])
    v_avg_conf = round(
        sum(r["confidence"] for r in vision_results) / max(len(vision_results), 1), 3
    )

    print(f"  Time taken     : {v_time}s")
    print(f"  Blocks found   : {len(vision_results)}")
    print(f"  Avg confidence : {v_avg_conf}")
    print(f"  Label breakdown: {dict(v_labels)}")

    # ── COMPARISON ──
    print(f"\n[COMPARISON]")
    print(f"  Speed winner   : {'Heuristic' if h_time < v_time else 'Vision'} "
          f"({min(h_time, v_time)}s vs {max(h_time, v_time)}s)")
    print(f"  Confidence win : {'Heuristic' if h_avg_conf > v_avg_conf else 'Vision'} "
          f"({max(h_avg_conf, v_avg_conf)} vs {min(h_avg_conf, v_avg_conf)})")
    print(f"  Blocks found   : Heuristic={len(heuristic_results)} "
          f"Vision={len(vision_results)}")

    # ── LABEL AGREEMENT ──
    # Check how many blocks both agree on
    h_texts = set(r["text"].strip().lower() for r in heuristic_results)
    v_texts = set(r["text"].strip().lower() for r in vision_results)
    overlap = h_texts & v_texts
    print(f"  Text overlap   : {len(overlap)} blocks found by both")

    return {
        "page": page_num,
        "heuristic": {
            "time": h_time,
            "blocks": len(heuristic_results),
            "avg_confidence": h_avg_conf,
            "labels": dict(h_labels)
        },
        "vision": {
            "time": v_time,
            "blocks": len(vision_results),
            "avg_confidence": v_avg_conf,
            "labels": dict(v_labels)
        }
    }


if __name__ == "__main__":
    pdf_path = "sample_docs/AI_Engineer_Roadmap.pdf"

    print("Loading PDF...")
    pages = pdf_to_images(pdf_path, dpi=150)

    # Benchmark first 2 pages only
    all_results = []
    for page in pages[:2]:
        result = benchmark_page(page["image"], page["page_number"])
        all_results.append(result)

    # ── FINAL SUMMARY ──
    print(f"\n{'='*50}")
    print("FINAL BENCHMARK SUMMARY")
    print(f"{'='*50}")

    total_h_time = sum(r["heuristic"]["time"] for r in all_results)
    total_v_time = sum(r["vision"]["time"] for r in all_results)
    avg_h_conf   = sum(r["heuristic"]["avg_confidence"] for r in all_results) / len(all_results)
    avg_v_conf   = sum(r["vision"]["avg_confidence"] for r in all_results) / len(all_results)

    print(f"\nTotal processing time:")
    print(f"  Heuristic : {total_h_time}s")
    print(f"  Vision    : {total_v_time}s")
    print(f"  Vision is {round(total_v_time/max(total_h_time,0.1), 1)}x slower")

    print(f"\nAverage confidence:")
    print(f"  Heuristic : {round(avg_h_conf, 3)}")
    print(f"  Vision    : {round(avg_v_conf, 3)}")

    print(f"\nConclusion:")
    print(f"  Use Heuristic when: speed matters, simple layouts, offline/no API")
    print(f"  Use Vision when   : accuracy matters, complex layouts, GPU available")
    print(f"\n  Our pipeline uses Vision first, Heuristic as fallback = best of both!")