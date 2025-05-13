"""
A component for viewing PDF files inside the Gradio UI.
"""
import os
import base64
from typing import Optional

import fitz
from config import logger

def get_pdf_page_as_base64(pdf_path: str, page_number: int = 0) -> Optional[str]:
    """Returns a specific page of the PDF as a base64 encoded PNG image."""
    try:
        abs_path = os.path.abspath(pdf_path)
        logger.debug(f"Loading PDF from absolute path: {abs_path}")

        if not os.path.exists(abs_path):
            logger.error(f"PDF file not found at {abs_path}")
            return None

        doc = fitz.open(abs_path)
        if page_number < 0 or page_number >= len(doc):
            logger.warning(f"Page {page_number+1} out of bounds (total pages: {len(doc)})")
            page_number = 0

        page = doc[page_number]
        # Increase zoom for better quality
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        return f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Error rendering PDF page {page_number} from {pdf_path}: {str(e)}")
        return None

def create_pdf_preview_html(pdf_path: str, page_number: int) -> str:
    """Creates HTML to display a PDF preview."""
    if not pdf_path or not os.path.exists(pdf_path):
        return "<div class='pdf-preview-error'>PDF file not found</div>"

    base64_img = get_pdf_page_as_base64(pdf_path, page_number)
    if not base64_img:
        return "<div class='pdf-preview-error'>Failed to render PDF page</div>"

    return f"""
    <div class="pdf-preview-container">
        <img src="{base64_img}" style="width:100%; max-width:800px; border:1px solid #ddd;"/>
    </div>
    """

def create_side_by_side_view(markdown_content: str, pdf_html: str) -> str:
    """Creates HTML for side-by-side view of PDF and Markdown."""
    return f"""
    <div style="display:flex; flex-direction:row; gap:20px; min-height:700px;">
        <div style="flex:1; border:1px solid #eee; border-radius:8px; padding:16px; overflow:auto; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <div style="border-bottom:1px solid #eee; padding-bottom:8px; margin-bottom:12px;">
                <h3 style="margin:0; color:#2c3e50;">PDF Preview</h3>
            </div>
            {pdf_html}
        </div>
        <div style="flex:1; border:1px solid #eee; border-radius:8px; padding:16px; overflow:auto; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <div style="border-bottom:1px solid #eee; padding-bottom:8px; margin-bottom:12px;">
                <h3 style="margin:0; color:#2c3e50;">Extracted Markdown</h3>
            </div>
            <pre style="white-space:pre-wrap; font-family:monospace; margin:0; padding:8px; border-radius:4px;">{markdown_content}</pre>
        </div>
    </div>
    """
