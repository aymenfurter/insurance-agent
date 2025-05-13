"""
Handles document processing including PDF analysis and Markdown conversion.
"""
import os
import time
from typing import List, Optional, Tuple

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (AzureError, ServiceRequestTimeoutError,
                                   ServiceResponseTimeoutError)

from config import (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                   AZURE_DOCUMENT_INTELLIGENCE_KEY, logger)
from document_intelligence_helper import (extract_page_markdown,
                                          extract_selection_marks,
                                          extract_tables_from_page,
                                          get_document_structure)
from local_storage import save_markdown_page
from utils import clean_filename, format_error_message

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def pdf_to_markdown_pages_for_doc(
    pdf_path: str,
    product_name: str,
    doc_name: str
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Converts a single PDF document to a list of markdown strings, one per page,
    and saves each page's markdown to a local file.

    Args:
        pdf_path: Path to the local PDF file.
        product_name: Name of the insurance product.
        doc_name: Name of the specific document (e.g., "general_terms.pdf").

    Returns:
        A tuple containing:
        - List of absolute paths to the saved markdown pages, or None if a critical error occurred.
        - List of markdown content for each page, or None if a critical error occurred.
          Returns ([], []) if DI processes but yields no content.
    """
    client = DocumentIntelligenceClient(
        endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
    )
    if not client:
        logger.error("Document Intelligence client is not available for PDF processing.")
        return None, None

    cleaned_doc_name_for_saving = clean_filename(os.path.splitext(doc_name)[0])
    logger.info(f"Starting Document Intelligence analysis for {pdf_path} "
                f"(Product: {product_name}, Doc: {doc_name}).")

    try:
        with open(pdf_path, "rb") as f_pdf:
            document_bytes = f_pdf.read()

        poller = None
        for attempt in range(MAX_RETRIES):
            try:
                poller = client.begin_analyze_document(
                    "prebuilt-layout",
                    body=document_bytes,
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
                break
            except (ServiceResponseTimeoutError, ServiceRequestTimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Timeout error during DI analysis (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error("Max retries reached. Could not process document.")
                    return None, None
            except AzureError as e:
                logger.error(f"AzureError during DI analysis for {doc_name} (Attempt {attempt + 1}): {format_error_message(e)}")
                return None, None

        if not poller:
            logger.error(f"DI poller not initialized for {doc_name} after retries.")
            return None, None

        result = poller.result()
        logger.info(f"Document Intelligence analysis completed for {pdf_path}.")

        if not result or not result.content:
            logger.warning(f"No content returned from Document Intelligence for {pdf_path}.")
            return [], []

        doc_structure = get_document_structure(result)
        logger.info(f"Document {doc_name}: {doc_structure['total_pages']} pages, "
                    f"{doc_structure['total_paragraphs']} paragraphs, {doc_structure['total_tables']} tables.")

        markdown_page_contents: List[str] = []
        markdown_page_paths: List[str] = []

        if result.pages:
            for page_idx, _ in enumerate(result.pages):
                page_md = extract_page_markdown(result, page_idx)
                tables_md = extract_tables_from_page(result, page_idx)
                selection_marks_md = extract_selection_marks(result, page_idx)

                full_page_content = page_md
                if tables_md:
                    full_page_content += "\n\n" + "\n\n".join(tables_md)
                if selection_marks_md:
                    full_page_content += "\n\n" + "\n\n".join(selection_marks_md)

                if full_page_content.strip():
                    markdown_page_contents.append(full_page_content)
                    saved_path = save_markdown_page(
                        product_name, cleaned_doc_name_for_saving, page_idx, full_page_content
                    )
                    if saved_path:
                        markdown_page_paths.append(saved_path)
                    else:
                        logger.error(f"Failed to save markdown page {page_idx} for {doc_name}.")

            if markdown_page_paths:
                logger.info(f"Extracted and saved {len(markdown_page_paths)} markdown pages for {doc_name}.")
                return markdown_page_paths, markdown_page_contents

        if result.content.strip() and not markdown_page_paths:
            logger.info(f"Saving entire document content as one page for {doc_name} (fallback).")
            markdown_page_contents = [result.content]
            saved_path = save_markdown_page(product_name, cleaned_doc_name_for_saving, 0, result.content)
            if saved_path:
                markdown_page_paths = [saved_path]
                return markdown_page_paths, markdown_page_contents

        logger.warning(f"No processable markdown content extracted from {pdf_path}.")
        return [], []

    except AzureError as e:
        logger.error(f"Azure Document Intelligence Error for {pdf_path}: {format_error_message(e)}")
        return None, None
    except FileNotFoundError:
        logger.error(f"PDF file not found at {pdf_path}.")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during PDF to Markdown conversion for {pdf_path}: {format_error_message(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

    if all_markdown_document_infos:
        product_data_dir = get_product_data_dir(product_name)
        for doc_info in all_markdown_document_infos:
            doc_info['original_pdf_path'] = os.path.abspath(
                os.path.join(product_data_dir, doc_info['original_pdf_path'])
            )
            doc_info['markdown_pages_paths'] = [
                os.path.abspath(os.path.join(product_data_dir, p))
                for p in doc_info['markdown_pages_paths']
            ]

        save_product_config(product_name, pdf_urls, all_original_pdf_paths, all_markdown_document_infos)