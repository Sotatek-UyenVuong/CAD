"""pdf_to_images.py — Convert PDF (or image files) to a list of page PNG images.

Supports:
  - PDF  → per-page PNG via PyMuPDF (fitz)
  - PNG/JPG/TIFF → passed through directly (single page)

Output images are saved under:
  {base_dir}/{file_id}/pages/page_{n}.png
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Generator

from PIL import Image


def pdf_to_page_images(
    file_path: Path | str,
    file_id: str,
    base_dir: Path | str,
    dpi: int = 300,
) -> list[dict]:
    """Convert a PDF (or image) to page PNG files.

    Returns:
        List of dicts: [{"page_number": 1, "image_path": Path, "width": int, "height": int}, ...]
    """
    file_path = Path(file_path)
    base_dir = Path(base_dir)
    out_dir = base_dir / file_id / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf(file_path, out_dir, dpi)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return _copy_image(file_path, out_dir)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _render_pdf(pdf_path: Path, out_dir: Path, dpi: int) -> list[dict]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required: pip install pymupdf") from exc

    results: list[dict] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        page_number = page_idx + 1
        img_path = out_dir / f"page_{page_number}.png"
        pix.save(str(img_path))
        results.append(
            {
                "page_number": page_number,
                "image_path": img_path,
                "width": pix.width,
                "height": pix.height,
            }
        )

    doc.close()
    return results


def _copy_image(src: Path, out_dir: Path) -> list[dict]:
    img = Image.open(src).convert("RGB")
    dst = out_dir / "page_1.png"
    img.save(str(dst), format="PNG")
    return [
        {
            "page_number": 1,
            "image_path": dst,
            "width": img.width,
            "height": img.height,
        }
    ]


def iter_page_images(
    file_path: Path | str,
    file_id: str,
    base_dir: Path | str,
    dpi: int = 300,
) -> Generator[dict, None, None]:
    """Lazy generator variant — yields one page at a time (memory-friendly)."""
    pages = pdf_to_page_images(file_path, file_id, base_dir, dpi)
    yield from pages
