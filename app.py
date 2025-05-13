"""
Main Gradio application for the Insurance Product Comparison Assistant.
Initializes the application state and sets up the UI tabs.
"""
import os
from typing import Any, Dict, List

import gradio as gr

from config import PRODUCTS_DIR, logger
from local_storage import load_product_config, load_questions_config, load_settings, list_saved_products
from ui_tabs.analysis_tab import create_analysis_tab
from ui_tabs.extraction_tab import create_extraction_tab
from ui_tabs.ingestion_tab import create_ingestion_tab
from ui_tabs.questions_tab import create_questions_tab
from ui_tabs.settings_tab import create_settings_tab


def initialize_products_list_from_storage() -> List[Dict[str, Any]]:
    """
    Pre-loads product information from disk to populate the initial app state.
    This ensures the app is aware of products processed in previous sessions.
    """
    products_list: List[Dict[str, Any]] = []
    saved_product_dirs = list_saved_products()
    logger.info(f"Found {len(saved_product_dirs)} product directories: {saved_product_dirs}")

    for product_dir_name in saved_product_dirs:
        try:
            logger.debug(f"Loading product config for directory: {product_dir_name}")
            product_config = load_product_config(product_dir_name)
            if product_config:
                product_name_original = product_config.get("product_name", product_dir_name)
                has_markdown_docs = bool(product_config.get("markdown_document_infos"))

                product_entry = {
                    "name": product_name_original,
                    "pdf_urls": product_config.get("pdf_urls", []),
                    "status": "Processed to Markdown" if has_markdown_docs else product_config.get("status", "Pending"),
                    "markdown_docs": product_config.get("markdown_document_infos", []),
                    "extraction_status": None
                }

                from local_storage import load_extracted_data
                extracted_data = load_extracted_data(product_name_original)
                if extracted_data:
                    product_entry["extraction_status"] = "Extracted" # Basic assumption

                products_list.append(product_entry)
                logger.info(f"Loaded product from storage: {product_name_original}, Status: {product_entry['status']}")
            else:
                logger.warning(f"Could not load config for product directory: {product_dir_name}")
        except Exception as e:
            logger.error(f"Error loading product config for {product_dir_name}: {e}", exc_info=True)

    logger.info(f"Initialized products list with {len(products_list)} products from storage.")
    return products_list


def create_initial_app_state() -> Dict[str, Any]:
    """Creates the initial global application state."""
    initial_settings = load_settings()
    initial_questions_config = load_questions_config()
    initial_products = initialize_products_list_from_storage()

    return {
        "products_list": initial_products,
        "questions_config": initial_questions_config,
        "current_azure_settings": initial_settings,
    }


def main() -> None:
    """Sets up and launches the Gradio application."""
    logger.info("Starting Insurance Product Comparison Assistant...")

    app_state_initial_dict = create_initial_app_state()

    logger.debug(f"Initial App State Products: {len(app_state_initial_dict['products_list'])} products")
    logger.debug(f"Initial App State Questions Config Categories: {len(app_state_initial_dict['questions_config'].get('categories',[]))}")
    logger.debug(f"Initial App State Questions Config Questions: {len(app_state_initial_dict['questions_config'].get('questions',[]))}")


    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue), title="Insurance Product Comparison Assistant") as demo:
        app_state = gr.State(value=app_state_initial_dict)

        gr.Markdown("# ☂️ Insurance Product Comparison Assistant")
        gr.Markdown(
            "An AI-powered tool to ingest, analyze, and compare insurance product terms. "
            "Follow the steps in the tabs below."
        )

        with gr.Tabs():
            with gr.TabItem("STEP 1: Document Ingestion & Preparation"):
                create_ingestion_tab(app_state)
            with gr.TabItem("STEP 2: Configure Questions & Categories"):
                create_questions_tab(app_state)
            with gr.TabItem("STEP 3: Data Extraction & Correction"):
                create_extraction_tab(app_state)
            with gr.TabItem("STEP 4: Analysis & Export"):
                create_analysis_tab(app_state)
            with gr.TabItem("⚙️ Settings"):
                create_settings_tab()

    logger.info("Gradio interface created. Launching application...")
    demo.queue().launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
