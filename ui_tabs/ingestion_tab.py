
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from config import DEFAULT_PRODUCTS, logger
from document_processor import pdf_to_markdown_pages_for_doc
from local_storage import (get_product_data_dir, list_saved_products,
                           load_markdown_page, load_product_config,
                           save_markdown_page, save_product_config)
from ui_components.pdf_previewer import (create_pdf_preview_html,
                                         create_side_by_side_view)
from utils import get_document_name_from_url, download_pdf

MAX_PDFS_PER_PRODUCT = 2


def create_ingestion_tab(app_state: gr.State) -> gr.Blocks:
    """Creates the Gradio UI for the Document Ingestion tab."""

    def initialize_app_state_with_defaults(current_app_state_value: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize app state with default products if empty."""
        if not current_app_state_value.get('products_list'):
            current_app_state_value['products_list'] = [
                {
                    "name": product["name"],
                    "pdf_urls": product["pdf_urls"],
                    "status": "Pending",
                    "markdown_docs": [],
                    "extraction_status": None
                }
                for product in DEFAULT_PRODUCTS
            ]
            logger.info(f"Initialized app state with {len(DEFAULT_PRODUCTS)} default products")
        return current_app_state_value

    def add_product_to_list_and_state(
        current_app_state_value: Dict[str, Any], product_name: str, pdf_urls_str: str
    ) -> Tuple[Dict[str, Any], str]:
        """Adds a new product to the list in app_state and saves initial config."""
        if not product_name.strip():
            return current_app_state_value, "Product name cannot be empty."

        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
        if any(p['name'] == product_name for p in products_list):
            return current_app_state_value, f"Product '{product_name}' already exists in the list."

        pdf_urls = [url.strip() for url in pdf_urls_str.split(',') if url.strip()]
        if not pdf_urls:
            return current_app_state_value, "Please provide at least one PDF URL."
        if len(pdf_urls) > MAX_PDFS_PER_PRODUCT:
            return current_app_state_value, f"Maximum {MAX_PDFS_PER_PRODUCT} PDF URLs allowed."

        new_product_entry = {
            "name": product_name,
            "pdf_urls": pdf_urls,
            "status": "Pending",
            "markdown_docs": [],
            "extraction_status": None
        }
        products_list.append(new_product_entry)
        current_app_state_value['products_list'] = products_list

        product_dir = get_product_data_dir(product_name)
        os.makedirs(product_dir, exist_ok=True)
        temp_config_path = os.path.join(product_dir, "_config.json")
        try:
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "product_name": product_name,
                    "pdf_urls": pdf_urls,
                    "status": "Pending",
                    "markdown_document_infos": []
                }, f, indent=4)
        except IOError as e:
            logger.error(f"Error saving initial product config for {product_name}: {e}")

        return current_app_state_value, f"Product '{product_name}' added with {len(pdf_urls)} PDF(s)."


    def process_all_products_handler(
        current_app_state_value: Dict[str, Any], progress: gr.Progress = gr.Progress(track_tqdm=True)
    ) -> Tuple[Dict[str, Any], str]:
        """Processes all 'Pending' or 'Error' products to Markdown."""
        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
        if not products_list:
            return current_app_state_value, "No products in the list to process."

        overall_status_messages: List[str] = []
        product_map_for_update = {p['name']: p.copy() for p in products_list}

        products_to_process = [
            p for p in products_list
            if p.get('status') in ["Pending", "Error downloading PDF", "Error in DI", "DI returned no content or error"]
        ]

        if not products_to_process:
            return current_app_state_value, "No products requiring processing."

        num_products = len(products_to_process)
        for i, product_entry in enumerate(products_to_process):
            product_name = product_entry['name']
            pdf_urls = product_entry['pdf_urls']
            progress((i + 1) / num_products, desc=f"Processing {product_name}...")
            logger.info(f"Processing product: {product_name}")

            product_succeeded = True
            all_original_pdf_paths: List[Optional[str]] = []
            all_markdown_document_infos: List[Dict[str, Any]] = []

            for doc_idx, url in enumerate(pdf_urls):
                doc_name = get_document_name_from_url(url)
                if not doc_name or doc_name == ".pdf":
                    doc_name = f"document_{doc_idx + 1}.pdf"

                pdf_path = download_pdf(url, product_name, doc_name)
                if not pdf_path:
                    overall_status_messages.append(f"Error downloading PDF for {product_name} from {url}.")
                    product_map_for_update[product_name]['status'] = "Error downloading PDF"
                    product_succeeded = False
                    break
                all_original_pdf_paths.append(pdf_path)

                page_paths, _ = pdf_to_markdown_pages_for_doc(pdf_path, product_name, doc_name)
                if page_paths is None:
                    overall_status_messages.append(f"Critical error converting PDF to Markdown for {product_name} - {doc_name}.")
                    product_map_for_update[product_name]['status'] = f"Error in DI for {doc_name}"
                    product_succeeded = False
                    break
                if not page_paths:
                     overall_status_messages.append(f"DI returned no content for {product_name} - {doc_name}.")

                product_data_dir = get_product_data_dir(product_name)
                relative_page_paths = [os.path.relpath(p, product_data_dir) for p in page_paths]

                all_markdown_document_infos.append({
                    "doc_name": doc_name,
                    "original_pdf_path": os.path.relpath(pdf_path, product_data_dir),
                    "markdown_pages_paths": relative_page_paths
                })

            if not product_succeeded:
                continue

            if all_markdown_document_infos:
                save_product_config(product_name, pdf_urls, all_original_pdf_paths, all_markdown_document_infos)
                product_map_for_update[product_name]['status'] = "Processed to Markdown"
                product_map_for_update[product_name]['markdown_docs'] = all_markdown_document_infos
                overall_status_messages.append(f"Successfully processed {product_name} to Markdown.")
            else:
                product_map_for_update[product_name]['status'] = "DI returned no content or error"
                overall_status_messages.append(f"DI processing yielded no content for {product_name}.")

        current_app_state_value['products_list'] = list(product_map_for_update.values())
        final_status = "\n".join(overall_status_messages) if overall_status_messages else "Processing complete. No issues."
        return current_app_state_value, final_status


    def get_product_names_for_review_dropdown(current_app_state_value: Dict[str, Any]) -> gr.Dropdown:
        """Populates dropdown with products that have been processed to Markdown."""
        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
        processed_product_names = sorted(list(set(
            p['name'] for p in products_list
            if p.get('status') == "Processed to Markdown" and p.get('markdown_docs')
        )))

        saved_product_dirs = list_saved_products()
        for product_dir_name in saved_product_dirs:
            config = load_product_config(product_dir_name)
            if config and config.get("product_name") and config.get("markdown_document_infos"):
                if config["product_name"] not in processed_product_names:
                    processed_product_names.append(config["product_name"])

        processed_product_names = sorted(list(set(processed_product_names)))

        return gr.Dropdown(
            choices=processed_product_names,
            label="Select Product to Review Markdown",
            interactive=True,
            value=processed_product_names[0] if processed_product_names else None
        )

    def get_doc_names_for_product_dropdown(
        current_app_state_value: Dict[str, Any], selected_product_name: Optional[str]
    ) -> gr.Dropdown:
        """Populates document dropdown based on selected product."""
        if not selected_product_name:
            return gr.Dropdown(choices=[], label="Select Document", interactive=False, value=None)

        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
        doc_names: List[str] = []

        product_entry = next((p for p in products_list if p['name'] == selected_product_name), None)

        if product_entry and product_entry.get('markdown_docs'):
            doc_names = [md_doc['doc_name'] for md_doc in product_entry.get('markdown_docs', []) if md_doc.get('doc_name')]
        else:
            config = load_product_config(selected_product_name)
            if config and config.get('markdown_document_infos'):
                 doc_names = [md_doc['doc_name'] for md_doc in config.get('markdown_document_infos', []) if md_doc.get('doc_name')]

        return gr.Dropdown(
            choices=sorted(list(set(doc_names))),
            label="Select Document",
            interactive=True,
            value=doc_names[0] if doc_names else None
        )

    def load_markdown_for_review_handler(
        current_app_state_value: Dict[str, Any], product_name: Optional[str], doc_name: Optional[str]
    ) -> Tuple[Optional[str], int, int, List[str], str, str, str]:
        """Loads markdown content for the selected product and document for review."""
        empty_return = (None, 0, 0, [], "Select product and document.", "", "")
        if not product_name or not doc_name:
            return empty_return

        product_entry = next((p for p in current_app_state_value.get('products_list', []) if p['name'] == product_name), None)

        if not product_entry or not product_entry.get('markdown_docs'):
            disk_config = load_product_config(product_name)
            if disk_config and disk_config.get('markdown_document_infos'):
                product_entry = {
                    "name": disk_config.get("product_name", product_name),
                    "markdown_docs": disk_config.get("markdown_document_infos", [])
                }
            else:
                return ("Product config or markdown documents not found on disk.", 0, 0, [], "Markdown data missing.", "", "")

        if not product_entry or not product_entry.get('markdown_docs'):
             return ("Product or its markdown documents not found.", 0, 0, [], "Markdown data missing.", "", "")


        md_doc_info = next((md for md in product_entry['markdown_docs'] if md.get('doc_name') == doc_name), None)
        if not md_doc_info or not md_doc_info.get('markdown_pages_paths'):
            return ("Selected document's markdown page paths not found.", 0, 0, [], "Markdown pages missing.", "", "")

        product_data_dir = get_product_data_dir(product_name)
        page_abs_paths = [
            os.path.join(product_data_dir, p) for p in md_doc_info['markdown_pages_paths'] if p
        ]

        if not page_abs_paths:
            return ("No valid markdown page paths found.", 0, 0, [], "No pages.", "", "")

        first_page_content = load_markdown_page(page_abs_paths[0])
        if first_page_content is None:
            return (f"Error loading first page from {page_abs_paths[0]}.", 0, 0, page_abs_paths, "Load error.", "", "")

        pdf_path_relative = md_doc_info.get('original_pdf_path')
        pdf_abs_path = ""
        if pdf_path_relative:
            pdf_abs_path = os.path.join(product_data_dir, pdf_path_relative)
            if not os.path.exists(pdf_abs_path):
                logger.warning(f"PDF path {pdf_abs_path} from md_doc_info not found. Trying product config.")
                pdf_abs_path = ""

        if not pdf_abs_path:
            disk_config = load_product_config(product_name)
            if disk_config and disk_config.get('original_pdf_paths'):
                doc_name_base = os.path.splitext(doc_name)[0]
                for p in disk_config['original_pdf_paths']:
                    if p and os.path.exists(p) and doc_name_base in os.path.basename(p):
                        pdf_abs_path = p
                        break

        side_by_side_html = ""
        if pdf_abs_path and os.path.exists(pdf_abs_path):
            pdf_preview_html_content = create_pdf_preview_html(pdf_abs_path, 0)
            side_by_side_html = create_side_by_side_view(first_page_content, pdf_preview_html_content)
        else:
            logger.warning(f"No valid PDF found for preview for {product_name}/{doc_name}")
            side_by_side_html = create_side_by_side_view(first_page_content, "<p>PDF preview not available.</p>")


        status_text = f"Displaying Page 1 of {len(page_abs_paths)} for {doc_name} ({product_name})"
        return first_page_content, 0, len(page_abs_paths), page_abs_paths, status_text, pdf_abs_path, side_by_side_html


    def navigate_page_handler(
        current_page_index: int,
        total_pages: int,
        page_paths_state: List[str],
        pdf_abs_path: str,
        direction: str,
        product_name: Optional[str] = None,
        doc_name: Optional[str] = None
    ) -> Tuple[str, int, str, str]:
        """Handles navigation between markdown pages."""
        if not page_paths_state or total_pages == 0:
            return "No pages to navigate.", current_page_index, "No pages loaded.", ""

        new_page_index = current_page_index
        if direction == "prev":
            new_page_index = max(0, current_page_index - 1)
        elif direction == "next":
            new_page_index = min(total_pages - 1, current_page_index + 1)

        logger.debug(f"Navigating page: current={current_page_index}, new={new_page_index}, total={total_pages}")
        logger.debug(f"Using page path: {page_paths_state[new_page_index]}")

        page_content = load_markdown_page(page_paths_state[new_page_index])
        if page_content is None:
            error_msg = f"Error loading page content from {page_paths_state[new_page_index]}"
            logger.error(error_msg)
            return error_msg, current_page_index, "Error loading page", ""

        side_by_side_html = ""
        if pdf_abs_path and os.path.exists(pdf_abs_path):
            pdf_preview_html_content = create_pdf_preview_html(pdf_abs_path, new_page_index)
            side_by_side_html = create_side_by_side_view(page_content, pdf_preview_html_content)
        else:
            side_by_side_html = create_side_by_side_view(page_content, "<p>PDF preview not available.</p>")

        context = f" for {doc_name} ({product_name})" if product_name and doc_name else ""
        status_text = f"Displaying Page {new_page_index + 1} of {total_pages}{context}"
        return page_content, new_page_index, status_text, side_by_side_html


    def save_edited_markdown_handler(
        page_paths_state: List[str], current_page_index: int, edited_content: str,
        product_name: Optional[str], doc_name: Optional[str]
    ) -> str:
        """Saves edited markdown content to the corresponding file."""
        if not product_name or not doc_name:
            return "Error: Product or document name not specified for saving."
        if not page_paths_state or not (0 <= current_page_index < len(page_paths_state)):
            return "Error: Page context not found for saving."

        page_num_0_indexed = current_page_index

        saved_path = save_markdown_page(product_name, doc_name, page_num_0_indexed, edited_content)
        if saved_path and os.path.exists(saved_path):
            return f"Page {current_page_index + 1} of {doc_name} saved successfully!"
        else:
            return f"Error saving Page {current_page_index + 1} of {doc_name}. Check logs."


    with gr.Blocks() as ingestion_tab_ui:


        with gr.Row():
            new_product_name_ui = gr.Textbox(label="Product Name", placeholder="e.g., My Household Insurance")
            new_product_urls_ui = gr.Textbox(
                label=f"PDF URLs (comma-separated, max {MAX_PDFS_PER_PRODUCT})",
                placeholder="https://example.com/doc1.pdf,https://example.com/doc2.pdf"
            )
        add_product_button_ui = gr.Button("Add Product to List", variant="secondary")
        add_product_status_ui = gr.Markdown("")

        gr.Markdown("### Current Product List")
        products_df_ui = gr.DataFrame(
            headers=["Name", "PDF URLs", "Status"],
            col_count=(3, "fixed"),
            interactive=False,
        )

        process_all_button_ui = gr.Button("Process All Pending/Error Products to Markdown", variant="primary")
        processing_status_ui = gr.Markdown("")

        gr.Markdown("---")
        gr.Markdown("### Review and Edit Markdown")

        with gr.Row():
            review_product_dropdown_ui = gr.Dropdown(
                choices=[], label="Select Product to Review Markdown", interactive=True
            )
            review_doc_dropdown_ui = gr.Dropdown(choices=[], label="Select Document", interactive=False)

        review_current_page_index_state = gr.State(0)
        review_total_pages_state = gr.State(0)
        review_page_paths_state = gr.State([])
        pdf_abs_path_state = gr.State("")

        with gr.Row():
            prev_page_button_ui = gr.Button("◀ Previous Page")
            next_page_button_ui = gr.Button("Next Page ▶")
            save_markdown_button_ui = gr.Button("Save Current Page Changes", variant="primary")
        save_markdown_status_ui = gr.Markdown("")

        with gr.Tabs():
            with gr.TabItem("Edit Markdown"):
                markdown_content_ui = gr.Textbox(
                    label="Markdown Content (Page View)", lines=20, interactive=True, show_copy_button=True
                )
                markdown_page_status_ui = gr.Textbox(label="Current Page Status", interactive=False)
            with gr.TabItem("Side-by-Side View (PDF & Markdown)"):
                side_by_side_view_ui = gr.HTML(label="PDF and Markdown Side-by-Side View")


        def update_products_df_display_from_app_state(
            current_app_state_value: Dict[str, Any]
        ) -> pd.DataFrame:
            """Helper to format products_list from app_state for DataFrame display."""
            products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
            display_data = [
                [
                    p.get('name', 'N/A'),
                    ", ".join(p.get('pdf_urls', [])),
                    p.get('status', 'Unknown')
                ] for p in products_list
            ]
            return pd.DataFrame(display_data, columns=["Name", "PDF URLs", "Status"])

        add_product_button_ui.click(
            add_product_to_list_and_state,
            inputs=[app_state, new_product_name_ui, new_product_urls_ui],
            outputs=[app_state, add_product_status_ui]
        ).then(
            update_products_df_display_from_app_state,
            inputs=[app_state],
            outputs=[products_df_ui]
        ).then(
            lambda: (gr.Textbox(value=""), gr.Textbox(value="")),
            outputs=[new_product_name_ui, new_product_urls_ui]
        )

        process_all_button_ui.click(
            process_all_products_handler,
            inputs=[app_state],
            outputs=[app_state, processing_status_ui]
        ).then(
            update_products_df_display_from_app_state,
            inputs=[app_state],
            outputs=[products_df_ui]
        ).then(
            get_product_names_for_review_dropdown,
            inputs=[app_state],
            outputs=[review_product_dropdown_ui]
        )

        review_product_dropdown_ui.change(
            get_doc_names_for_product_dropdown,
            inputs=[app_state, review_product_dropdown_ui],
            outputs=[review_doc_dropdown_ui]
        ).then(
            lambda: (None, 0, 0, [], "Select a document.", "", ""),
            outputs=[
                markdown_content_ui, review_current_page_index_state, review_total_pages_state,
                review_page_paths_state, markdown_page_status_ui, pdf_abs_path_state, side_by_side_view_ui
            ]
        )

        review_doc_dropdown_ui.change(
            load_markdown_for_review_handler,
            inputs=[app_state, review_product_dropdown_ui, review_doc_dropdown_ui],
            outputs=[
                markdown_content_ui, review_current_page_index_state, review_total_pages_state,
                review_page_paths_state, markdown_page_status_ui, pdf_abs_path_state, side_by_side_view_ui
            ]
        )

        prev_page_button_ui.click(
            navigate_page_handler,
            inputs=[
                review_current_page_index_state,
                review_total_pages_state,
                review_page_paths_state,
                pdf_abs_path_state,
                gr.State("prev"),
                review_product_dropdown_ui,
                review_doc_dropdown_ui
            ],
            outputs=[
                markdown_content_ui,
                review_current_page_index_state,
                markdown_page_status_ui,
                side_by_side_view_ui
            ]
        )
        next_page_button_ui.click(
            navigate_page_handler,
            inputs=[
                review_current_page_index_state,
                review_total_pages_state,
                review_page_paths_state,
                pdf_abs_path_state,
                gr.State("next"),
                review_product_dropdown_ui,
                review_doc_dropdown_ui
            ],
            outputs=[
                markdown_content_ui,
                review_current_page_index_state,
                markdown_page_status_ui,
                side_by_side_view_ui
            ]
        )

        save_markdown_button_ui.click(
            save_edited_markdown_handler,
            inputs=[
                review_page_paths_state, review_current_page_index_state, markdown_content_ui,
                review_product_dropdown_ui, review_doc_dropdown_ui
            ],
            outputs=[save_markdown_status_ui]
        )

        ingestion_tab_ui.load(
            lambda current_app_state_val: (
                update_products_df_display_from_app_state(
                    initialize_app_state_with_defaults(current_app_state_val)
                ),
                get_product_names_for_review_dropdown(current_app_state_val)
            ),
            inputs=[app_state],
            outputs=[products_df_ui, review_product_dropdown_ui]
        )
    return ingestion_tab_ui