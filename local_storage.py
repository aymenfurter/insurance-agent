"""
Handles local file storage for product configurations, markdown content,
question configurations, extracted answers, settings, and Excel export.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                    AZURE_DOCUMENT_INTELLIGENCE_KEY, AZURE_OPENAI_API_KEY,
                    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT,
                    EXTRACTED_DATA_DIR, PRODUCTS_DIR, QUESTIONS_CONFIG_PATH,
                    SETTINGS_PATH, logger)


def get_product_data_dir(product_name: str) -> str:
    """Gets the directory path for a specific product, using a cleaned name."""
    from utils import clean_filename
    safe_product_name = clean_filename(product_name)
    product_dir = os.path.join(PRODUCTS_DIR, safe_product_name)
    logger.debug(f"Product directory for '{product_name}' is '{product_dir}'")
    return product_dir


def save_product_config(
    product_name: str,
    pdf_urls: List[str],
    original_pdf_paths: List[Optional[str]],
    markdown_doc_infos: List[Dict[str, Any]]
) -> None:
    """Saves product configuration: URLs, paths to original PDFs, and markdown document info."""
    product_dir = get_product_data_dir(product_name)
    os.makedirs(product_dir, exist_ok=True)
    config_path = os.path.join(product_dir, "_config.json")

    relative_pdf_paths = []
    for p in original_pdf_paths:
        if p and os.path.exists(p):
            try:
                rel_path = os.path.relpath(p, product_dir)
                relative_pdf_paths.append(rel_path)
            except ValueError:
                relative_pdf_paths.append(p)
        else:
            relative_pdf_paths.append(p if p else "")

    for doc_info in markdown_doc_infos:
        if 'markdown_pages_paths' in doc_info:
            relative_paths = []
            for p in doc_info['markdown_pages_paths']:
                if p and os.path.exists(p):
                    try:
                        rel_path = os.path.relpath(p, product_dir)
                        relative_paths.append(rel_path)
                    except ValueError:
                        relative_paths.append(p)
                else:
                    relative_paths.append(p if p else "")
            doc_info['markdown_pages_paths'] = relative_paths

    config_data = {
        "product_name": product_name,
        "pdf_urls": pdf_urls,
        "original_pdf_paths_relative": relative_pdf_paths,
        "markdown_document_infos": markdown_doc_infos,
        "status": "Processed to Markdown" if markdown_doc_infos else "Pending"
    }

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Saved product configuration to {config_path}")
    except IOError as e:
        logger.error(f"Error saving product config to {config_path}: {e}")


def load_product_config(product_identifier: str) -> Optional[Dict[str, Any]]:
    """
    Loads a product's configuration.
    Tries both original name and cleaned filename.
    """
    from utils import clean_filename

    product_dir = os.path.join(PRODUCTS_DIR, product_identifier)
    if not os.path.exists(product_dir):
        cleaned_name = clean_filename(product_identifier)
        product_dir = os.path.join(PRODUCTS_DIR, cleaned_name)
        if not os.path.exists(product_dir):
            logger.error(f"Product directory not found for: {product_identifier} or {cleaned_name}")
            return None

    config_path = os.path.join(product_dir, "_config.json")
    logger.debug(f"Looking for product config at: {config_path}")

    if not os.path.exists(config_path):
        logger.warning(f"Product config file not found: {config_path}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if 'original_pdf_paths_relative' in config:
            config['original_pdf_paths'] = [
                os.path.abspath(os.path.join(product_dir, p)) if p else None
                for p in config['original_pdf_paths_relative']
            ]

        if 'markdown_document_infos' in config:
            for doc_info in config['markdown_document_infos']:
                if 'markdown_pages_paths' in doc_info:
                    doc_info['markdown_pages_paths_absolute'] = [
                        os.path.abspath(os.path.join(product_dir, p)) if p else None
                        for p in doc_info['markdown_pages_paths']
                    ]
                    doc_info['markdown_pages_paths'] = doc_info['markdown_pages_paths_absolute']

        logger.info(f"Successfully loaded product config for {product_identifier}")
        return config

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading product config from {config_path}: {e}")
        return None


def list_saved_products() -> List[str]:
    """Lists all product directory names for which configuration has been saved."""
    if not os.path.exists(PRODUCTS_DIR):
        return []
    return [
        d for d in os.listdir(PRODUCTS_DIR)
        if os.path.isdir(os.path.join(PRODUCTS_DIR, d))
    ]


def save_markdown_page(product_name: str, doc_name: str, page_num: int, markdown_content: str) -> Optional[str]:
    """Saves a single markdown page for a document of a product."""
    from utils import clean_filename
    product_data_dir_path = get_product_data_dir(product_name)

    cleaned_doc_name = clean_filename(doc_name)
    doc_md_dir = os.path.join(product_data_dir_path, f"{cleaned_doc_name}_md_pages")
    os.makedirs(doc_md_dir, exist_ok=True)

    page_filename = f"page_{page_num + 1}.md"
    page_path = os.path.join(doc_md_dir, page_filename)

    try:
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Saved markdown for {product_name} - {doc_name} - Page {page_num + 1} to {page_path}")
        return page_path
    except IOError as e:
        logger.error(f"Error saving markdown page for {product_name}, {doc_name}, page {page_num + 1}: {e}")
        return None


def load_markdown_page(page_path: str) -> Optional[str]:
    """Load markdown content from a file with better path handling."""
    if not page_path:
        logger.error("Empty page path provided")
        return None

    try:
        abs_path = os.path.abspath(page_path)
        logger.debug(f"Loading markdown from: {abs_path}")

        if not os.path.isfile(abs_path):
            logger.error(f"Not a file or doesn't exist: {abs_path}")
            return None

        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
    except Exception as e:
        logger.error(f"Error loading markdown from {page_path}: {str(e)}")
        return None


def save_questions_config(config_data: Dict[str, Any]) -> None:
    """Saves the question configurations."""
    try:
        with open(QUESTIONS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Question configuration saved to {QUESTIONS_CONFIG_PATH}")
    except IOError as e:
        logger.error(f"Error saving question configuration: {e}")


def load_questions_config() -> Dict[str, Any]:
    """Loads the question configurations, returning a default structure on failure."""
    default_config = {"categories": [], "questions": []}
    if not os.path.exists(QUESTIONS_CONFIG_PATH):
        logger.info(f"Questions config file not found at {QUESTIONS_CONFIG_PATH}. Returning default.")
        return default_config
    try:
        with open(QUESTIONS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading question configuration from {QUESTIONS_CONFIG_PATH}: {e}. Returning default.")
        return default_config


def get_extracted_data_path(product_name: str) -> str:
    """Gets the file path for a product's extracted data using a cleaned name."""
    from utils import clean_filename
    safe_product_name = clean_filename(product_name)
    return os.path.join(EXTRACTED_DATA_DIR, f"{safe_product_name}_extracted.json")


def save_extracted_data(product_name: str, data: Dict[str, Any]) -> None:
    """Saves extracted (and potentially corrected) answers for a product."""
    filepath = get_extracted_data_path(product_name)
    data_to_save = data.copy()
    data_to_save['product_name_original'] = product_name
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)
        logger.info(f"Saved extracted data for product: {product_name} to {filepath}")
    except IOError as e:
        logger.error(f"Error saving extracted data for {product_name}: {e}")


def load_extracted_data(product_name: str) -> Optional[Dict[str, Any]]:
    """Loads extracted answers for a product."""
    filepath = get_extracted_data_path(product_name)
    if not os.path.exists(filepath):
        logger.debug(f"Extracted data file not found for {product_name} at {filepath}")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if 'product_name_original' in data:
            data['product_name'] = data['product_name_original']
        elif 'product_name' not in data:
            data['product_name'] = product_name
        return data
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading extracted data for {product_name} from {filepath}: {e}")
        return None


def save_settings(settings: Dict[str, str]) -> None:
    """Saves UI-configurable settings."""
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        logger.info(f"Settings saved to {SETTINGS_PATH}")
    except IOError as e:
        logger.error(f"Error saving settings to {SETTINGS_PATH}: {e}")


def load_settings() -> Dict[str, str]:
    """Loads UI-configurable settings, falling back to .env defaults."""
    env_defaults = {
        "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
        "azure_openai_api_key": AZURE_OPENAI_API_KEY,
        "azure_openai_api_version": AZURE_OPENAI_API_VERSION,
        "azure_document_intelligence_endpoint": AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        "azure_document_intelligence_key": AZURE_DOCUMENT_INTELLIGENCE_KEY,
    }
    current_settings = env_defaults.copy()

    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                loaded_from_file = json.load(f)
            for key, value in loaded_from_file.items():
                if value and isinstance(value, str) and value.strip():
                    current_settings[key] = value
                elif key not in current_settings:
                    current_settings[key] = value

        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading settings from {SETTINGS_PATH}, using .env defaults: {e}")
    else:
        logger.info(f"Settings file not found at {SETTINGS_PATH}. Using .env defaults.")
    return current_settings


def export_data_to_excel(
    all_products_data: List[Dict[str, Any]],
    questions_config: Dict[str, Any]
) -> Optional[str]:
    """
    Exports the extracted data for all products to an Excel file.
    Each product is a sheet. Rows are questions, columns are categories.
    """
    if not all_products_data:
        logger.warning("No data provided for Excel export.")
        return None

    output_filename = os.path.join(EXTRACTED_DATA_DIR, "comparison_export.xlsx")
    question_map = {q['id']: q['text'] for q in questions_config.get('questions', [])}

    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            for product_data_item in all_products_data:
                product_name_original = product_data_item.get(
                    "product_name_original",
                    product_data_item.get("product_name", "UnknownProduct")
                )
                answers = product_data_item.get("answers", [])
                sheet_name = re.sub(r'[\[\]\*:\?/\\"]', '_', product_name_original)[:31]

                if not answers:
                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                    continue

                data_for_df: Dict[str, Dict[str, str]] = {}
                for ans_item in answers:
                    q_id = ans_item.get("question_id")
                    q_text = question_map.get(q_id, q_id if q_id else "Unknown Question")
                    category = ans_item.get("category", "Uncategorized")
                    answer_text = ans_item.get("answer", "")

                    if q_text not in data_for_df:
                        data_for_df[q_text] = {}
                    data_for_df[q_text][category] = str(answer_text)

                df = pd.DataFrame.from_dict(data_for_df, orient='index')
                if not df.empty:
                    df = df.reindex(sorted(df.columns), axis=1)
                    df.index.name = "Question"
                    df = df.sort_index()
                df.to_excel(writer, sheet_name=sheet_name)
        logger.info(f"Data exported to Excel: {output_filename}")
        return output_filename
    except Exception as e:
        logger.error(f"Error exporting data to Excel: {e}")
        return None
