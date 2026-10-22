from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List


class PageVisualRenderer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def render_pdf_pages(self, pdf_path: str, paper_id: str, force: bool = False, max_pages: int | None = None) -> List[Dict[str, Any]]:
        path = Path(pdf_path)
        output_dir = self.root_dir / paper_id / "pages"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for real page visual rendering") from exc
        doc = fitz.open(path)
        pages = []
        for page_index, page in enumerate(doc):
            if max_pages is not None and page_index >= max_pages:
                break
            page_number = page_index + 1
            output_path = output_dir / f"page_{page_number:04d}.png"
            if force or not output_path.exists():
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                pix.save(str(output_path))
            data = output_path.read_bytes()
            pages.append({
                "paper_id": paper_id,
                "page_number": page_number,
                "page_image_path": str(output_path),
                "page_hash": hashlib.sha256(data).hexdigest(),
                "width": int(getattr(pix, "width", 0)) if "pix" in locals() else None,
                "height": int(getattr(pix, "height", 0)) if "pix" in locals() else None,
            })
        return pages
