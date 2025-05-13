"""
Common utility functions for the Insurance Comparison Assistant.
"""
import os
import re
from typing import Any, Optional

import requests

from config import logger


def clean_filename(filename: str) -> str:
    """Cleans a filename to be safe for filesystem operations."""
    if not isinstance(filename, str):
        filename = str(filename)
    filename = re.sub(r'[^\w\s-]', '', filename).strip()
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename


def download_pdf(url: str, product_name: str, doc_name: str) -> Optional[str]:
    """Downloads a PDF from a URL and saves it locally."""
    from local_storage import get_product_data_dir

    product_dir = get_product_data_dir(product_name)
    os.makedirs(product_dir, exist_ok=True)

    safe_doc_name = clean_filename(doc_name)
    if not safe_doc_name.lower().endswith(".pdf"):
        safe_doc_name += ".pdf"

    filepath = os.path.join(product_dir, safe_doc_name)

    try:
        logger.info(f"Downloading PDF from {url} to {filepath}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
    except IOError as e:
        logger.error(f"Error saving PDF to {filepath}: {e}")
    return None


def get_document_name_from_url(url: str) -> str:
    """Extracts a document name from a URL, ensuring it ends with .pdf."""
    try:
        filename = os.path.basename(url.split('?')[0].split('#')[0])
        if not filename: # If URL ends with / or is just domain
            filename = "document"

        name_part, ext_part = os.path.splitext(filename)
        if not ext_part or ext_part.lower() != ".pdf":
            filename = name_part + ".pdf"

        return filename if filename != ".pdf" else "document.pdf"
    except Exception as e:
        logger.warning(f"Could not parse document name from URL '{url}': {e}. Using fallback.")
        return "document.pdf"


def format_error_message(error_details: Any) -> str:
    """Formats an error message for display, attempting to be informative."""
    if isinstance(error_details, str):
        return error_details
    if isinstance(error_details, Exception):
        if hasattr(error_details, 'response') and error_details.response is not None:
            try:
                err_json = error_details.response.json()
                if 'error' in err_json and 'message' in err_json['error']:
                    return f"API Error: {err_json['error']['message']} (Status: {error_details.response.status_code})"
            except ValueError:
                return f"API Error: {error_details.response.text} (Status: {error_details.response.status_code})"
        return f"An error occurred: {str(error_details)}"
    return f"An unknown error occurred: {str(error_details)}"
