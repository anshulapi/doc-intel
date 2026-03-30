import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[dict]:
    """
    Convert each page of a PDF into an image.
    
    Args:
        pdf_path: path to the PDF file
        dpi: resolution (higher = clearer but slower). 200 is sweet spot.
    
    Returns:
        List of dicts, one per page:
        {
            "page_number": 1,
            "image": PIL Image object,
            "width": 1240,
            "height": 1754
        }
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")
    
    # Open the PDF
    doc = fitz.open(str(pdf_path))
    pages = []
    
    print(f"Processing: {pdf_path.name} ({len(doc)} pages)")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Matrix controls resolution. 1 point = 1/72 inch by default.
        # dpi/72 gives us the scale factor.
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        
        # Render page to a pixel map
        pixmap = page.get_pixmap(matrix=matrix)
        
        # Convert to PIL Image via bytes
        img_bytes = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pages.append({
            "page_number": page_num + 1,
            "image": image,
            "width": image.width,
            "height": image.height
        })
        
        print(f"  Page {page_num + 1}: {image.width}x{image.height}px")
    
    doc.close()
    print(f"Done! Extracted {len(pages)} pages.\n")
    return pages


if __name__ == "__main__":
    # Quick test — we'll use a sample PDF
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingestor.py <path_to_pdf>")
        print("Example: python pdf_ingestor.py sample_docs/test.pdf")
    else:
        pages = pdf_to_images(sys.argv[1])
        
        # Save first page as preview image
        if pages:
            output_path = "sample_docs/page_1_preview.png"
            pages[0]["image"].save(output_path)
            print(f"Saved preview: {output_path}")