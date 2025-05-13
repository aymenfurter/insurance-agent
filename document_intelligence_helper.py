"""
Helper functions for working with Azure Document Intelligence results.
"""
import re
from typing import Dict, List, Any

from azure.ai.documentintelligence.models import AnalyzeResult

from config import logger


def extract_page_markdown(result: AnalyzeResult, page_number: int) -> str:
    """
    Extracts markdown content for a specific page from Document Intelligence result.
    Tries multiple methods to ensure content extraction.
    """
    if not result or not result.pages or not (0 <= page_number < len(result.pages)):
        logger.warning(f"Invalid page_number {page_number} or missing pages in DI result.")
        return ""

    page_obj = result.pages[page_number]
    page_markdown = ""

    if hasattr(page_obj, 'spans') and page_obj.spans:
        for span in page_obj.spans:
            page_markdown += result.content[span.offset : span.offset + span.length]
    page_markdown = page_markdown.strip()

    if not page_markdown and result.content and "<!-- Page" in result.content:
        page_marker = f"<!-- Page {page_number + 1} -->"
        next_page_marker = f"<!-- Page {page_number + 2} -->"
        content_parts = result.content.split(page_marker, 1)

        if len(content_parts) > 1:
            content_after_marker = content_parts[1]
            if next_page_marker in content_after_marker:
                page_markdown = content_after_marker.split(next_page_marker, 1)[0].strip()
            else:
                page_markdown = content_after_marker.strip()
        elif page_number == 0 and page_marker not in content_parts[0] and "<!-- Page" not in content_parts[0]:
             page_markdown = content_parts[0].strip()


    if not page_markdown and result.paragraphs:
        page_paragraphs_content = []
        for paragraph in result.paragraphs:
            if paragraph.bounding_regions:
                for region in paragraph.bounding_regions:
                    if region.page_number == page_number + 1:
                        page_paragraphs_content.append(paragraph.content)
                        break
        if page_paragraphs_content:
            page_markdown = "\n\n".join(page_paragraphs_content).strip()

    return page_markdown


def extract_tables_from_page(result: AnalyzeResult, page_number: int) -> List[str]:
    """
    Extracts tables from a specific page and converts them to markdown format.
    """
    table_markdowns: List[str] = []
    if not result or not result.tables:
        return table_markdowns

    for table_idx, table in enumerate(result.tables):
        table_on_page = False
        if table.bounding_regions:
            for region in table.bounding_regions:
                if region.page_number == page_number + 1:
                    table_on_page = True
                    break
        if not table_on_page:
            continue

        table_md_rows: List[str] = []
        header_generated = False


        rows_data: Dict[int, Dict[int, str]] = {}
        for cell in table.cells:
            if cell.row_index not in rows_data:
                rows_data[cell.row_index] = {}
            rows_data[cell.row_index][cell.column_index] = rows_data[cell.row_index].get(cell.column_index, "") + cell.content

        for r_idx in sorted(rows_data.keys()):
            current_row_cells = []
            for c_idx in range(table.column_count):
                cell_content = rows_data[r_idx].get(c_idx, "")
                current_row_cells.append(cell_content.replace("|", "\\|"))
            table_md_rows.append(f"| {' | '.join(current_row_cells)} |")

            if not header_generated and table_md_rows:
                separator = "| " + " | ".join(["---"] * table.column_count) + " |"
                table_md_rows.insert(1, separator)
                header_generated = True

        if table_md_rows:
            table_markdowns.append(f"\n\n**Table {table_idx + 1}**\n" + "\n".join(table_md_rows))

    return table_markdowns


def extract_selection_marks(result: AnalyzeResult, page_number: int) -> List[str]:
    """
    Extracts selection marks (e.g., checkboxes) from a specific page.
    """
    selection_mark_texts: List[str] = []
    if not result or not result.pages or not (0 <= page_number < len(result.pages)):
        return selection_mark_texts

    page = result.pages[page_number]
    if hasattr(page, 'selection_marks') and page.selection_marks:
        for idx, mark in enumerate(page.selection_marks):
            state = mark.state if hasattr(mark, 'state') else "unknown"
            confidence = mark.confidence if hasattr(mark, 'confidence') else 0.0
            selection_mark_texts.append(
                f"Selection Mark {idx + 1}: {state} (Confidence: {confidence:.2f})"
            )
    return selection_mark_texts


def get_document_structure(result: AnalyzeResult) -> Dict[str, Any]:
    """
    Analyzes and returns basic structural information from Document Intelligence result.
    """
    structure: Dict[str, Any] = {
        "total_pages": len(result.pages) if result.pages else 0,
        "total_paragraphs": len(result.paragraphs) if hasattr(result, 'paragraphs') and result.paragraphs else 0,
        "total_tables": len(result.tables) if hasattr(result, 'tables') and result.tables else 0,
        "has_selection_marks": False
    }

    if result.pages:
        for page in result.pages:
            if hasattr(page, 'selection_marks') and page.selection_marks:
                structure["has_selection_marks"] = True
                break
    return structure


def split_content_by_pages(content: str) -> List[str]:
    """
    Splits a single markdown string (potentially containing page markers) into a list of strings,
    one per page. This is a fallback if page-specific extraction fails.
    """
    if not content:
        return []
    if "<!-- Page" not in content:
        return [content]

    pages: List[str] = []
    pattern = r"(.*?)(?=<!-- Page \d+ -->|$)"

    marker_pattern = r"<!-- Page \d+ -->"

    last_end = 0
    parts = re.split(f"({marker_pattern})", content)

    current_page_content = ""
    for part in parts:
        if re.match(marker_pattern, part):
            if current_page_content.strip():
                 pages.append(current_page_content.strip())
            current_page_content = part
        else:
            current_page_content += part

    if current_page_content.strip():
        pages.append(current_page_content.strip())

    if not pages and content.strip():
        return [content.strip()]

    return [p for p in pages if p.strip()]
