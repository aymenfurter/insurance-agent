
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from config import (AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT,
                    AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT,
                    EXTRACTED_DATA_DIR, logger)
from data_extractor import (apply_corrections, check_document_size,
                            extract_answers_for_product, self_correct_answers)
from local_storage import (load_extracted_data, load_product_config,
                           load_questions_config, save_extracted_data)


def create_extraction_tab(app_state: gr.State) -> gr.Blocks:
    """Creates the Gradio UI for the Data Extraction and Correction tab."""

    if app_state.value is None: app_state.value = {}
    if 'products_list' not in app_state.value:
        app_state.value['products_list'] = []
        logger.warning("extraction_tab: 'products_list' initialized as empty in app_state.")
    if 'questions_config' not in app_state.value or not app_state.value['questions_config']:
        app_state.value['questions_config'] = load_questions_config()
        logger.info("extraction_tab: 'questions_config' loaded/initialized in app_state.")


    def handle_extract_all_answers_action(
        current_app_state_value: Dict[str, Any],
        model_choice: str,
        progress: gr.Progress = gr.Progress(track_tqdm=True)
    ) -> Tuple[Dict[str, Any], str, List[str]]:
        """Extracts answers for all products ready for extraction."""
        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
        questions_cfg = current_app_state_value.get('questions_config')

        if not questions_cfg or not questions_cfg.get('questions'):
            extracted_names = [
                p['name'] for p in products_list if p.get('extraction_status') == "Extracted"
            ]
            return current_app_state_value, "No questions configured. Please configure in Step 2.", extracted_names

        products_to_extract_names = [
            p['name'] for p in products_list if
            p.get('status') == "Processed to Markdown" and
            p.get('extraction_status') not in ["Extracted", "Corrected"]
        ]

        if not products_to_extract_names:
            extracted_names = [
                p['name'] for p in products_list if p.get('extraction_status') in ["Extracted", "Corrected"]
            ]
            return current_app_state_value, "No products ready or requiring answer extraction.", extracted_names

        overall_status: List[str] = []
        product_map_for_state_update = {p['name']: p.copy() for p in products_list}

        num_to_process = len(products_to_extract_names)
        for i, product_name in enumerate(products_to_extract_names):
            progress((i + 1) / num_to_process, desc=f"Extracting for {product_name}...")
            logger.info(f"Attempting extraction for product: {product_name}")

            product_disk_config = load_product_config(product_name)
            if not product_disk_config or not product_disk_config.get('markdown_document_infos'):
                overall_status.append(f"Skipping {product_name}: Markdown data not found or incomplete on disk.")
                logger.warning(f"Markdown data issue for {product_name}, skipping extraction.")
                product_map_for_state_update[product_name]['extraction_status'] = "Markdown Missing"
                continue

            extracted_data = extract_answers_for_product(product_name, questions_cfg, extract_by_category=True, model_choice=model_choice)

            current_product_entry = product_map_for_state_update[product_name]
            if extracted_data and extracted_data.get("answers"):
                save_extracted_data(product_name, extracted_data)
                overall_status.append(f"Successfully extracted answers for {product_name}.")
                current_product_entry['extraction_status'] = "Extracted"
            else:
                overall_status.append(f"Failed to extract answers for {product_name}. Check logs.")
                current_product_entry['extraction_status'] = "Extraction Failed"

        current_app_state_value['products_list'] = list(product_map_for_state_update.values())
        final_extracted_product_names = sorted(list(set(
            p['name'] for p in current_app_state_value['products_list']
            if p.get('extraction_status') in ["Extracted", "Corrected"]
        )))
        return current_app_state_value, "\n".join(overall_status) if overall_status else "Extraction complete.", final_extracted_product_names


    def get_display_data_for_product_df(product_name: Optional[str]) -> pd.DataFrame:
        """Formats extracted data for a product into a DataFrame."""
        columns = ["Question ID", "Question Text", "Category", "Answer", "Status"]
        if not product_name:
            return pd.DataFrame(columns=columns)

        data = load_extracted_data(product_name)
        if data and "answers" in data and isinstance(data["answers"], list):
            df_rows = [
                {
                    "Question ID": ans.get("question_id"),
                    "Question Text": ans.get("question_text"),
                    "Category": ans.get("category"),
                    "Answer": str(ans.get("answer")),
                    "Status": ans.get("status", "raw")
                }
                for ans in data["answers"]
                if isinstance(ans, dict) and all(k in ans for k in ["question_id", "question_text", "category", "answer"])
            ]
            return pd.DataFrame(df_rows, columns=columns)
        return pd.DataFrame(columns=columns)


    def handle_self_correction_action(
        product_name: Optional[str], model_choice: str, progress: gr.Progress = gr.Progress(track_tqdm=True)
    ) -> Tuple[str, pd.DataFrame]:
        """Performs AI self-correction review for a product's extracted answers."""
        df_columns = ["Question ID", "Question Text", "Category", "Answer", "Status"]
        if not product_name:
            return "Please select a product for self-correction.", pd.DataFrame(columns=df_columns)

        progress(0.1, desc=f"Loading data for {product_name}...")
        extracted_data_dict = load_extracted_data(product_name)
        if not extracted_data_dict or "answers" not in extracted_data_dict:
            return f"No extracted data found for {product_name}.", pd.DataFrame(columns=df_columns)

        extracted_answers = extracted_data_dict["answers"]
        progress(0.3, desc=f"Requesting LLM review for {product_name}...")
        corrections_list = self_correct_answers(product_name, extracted_answers, model_choice=model_choice)

        current_df = get_display_data_for_product_df(product_name)

        if corrections_list is None:
            return f"Self-correction review failed for {product_name}. Check logs.", current_df

        if not corrections_list:
            return f"No corrections suggested by AI for {product_name}. Answers appear consistent.", current_df

        temp_corrections_dir = os.path.join(EXTRACTED_DATA_DIR, "temp_corrections")
        os.makedirs(temp_corrections_dir, exist_ok=True)
        safe_product_filename = product_name.replace(' ', '_').replace('/', '_').lower()
        temp_corrections_path = os.path.join(temp_corrections_dir, f"{safe_product_filename}_corrections_temp.json")
        try:
            with open(temp_corrections_path, "w", encoding="utf-8") as f:
                json.dump(corrections_list, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save temporary corrections for {product_name}: {e}")
            return f"Failed to save temporary corrections: {e}", current_df

        correction_q_ids = {c['question_id'] for c in corrections_list}
        if not current_df.empty:
            current_df['Status'] = current_df.apply(
                lambda row: 'review_suggested' if row['Question ID'] in correction_q_ids and row['Status'] != 'corrected' else row['Status'],
                axis=1
            )

        summary = f"AI suggested {len(corrections_list)} corrections for {product_name}. Review and apply if appropriate."
        return summary, current_df


    def handle_apply_corrections_action_button(
        product_name: Optional[str], progress: gr.Progress = gr.Progress(track_tqdm=True) # model_choice removed, not used by apply_corrections
    ) -> Tuple[str, pd.DataFrame]:
        """Applies previously suggested AI corrections."""
        df_columns = ["Question ID", "Question Text", "Category", "Answer", "Status"]
        if not product_name:
            return "Please select a product to apply corrections.", pd.DataFrame(columns=df_columns)

        progress(0.1, desc=f"Loading data and corrections for {product_name}...")
        original_data_dict = load_extracted_data(product_name)
        if not original_data_dict or "answers" not in original_data_dict:
            return f"Original extracted data not found for {product_name}.", pd.DataFrame(columns=df_columns)

        temp_corrections_dir = os.path.join(EXTRACTED_DATA_DIR, "temp_corrections")
        safe_product_filename = product_name.replace(' ', '_').replace('/', '_').lower()
        temp_corrections_path = os.path.join(temp_corrections_dir, f"{safe_product_filename}_corrections_temp.json")

        if not os.path.exists(temp_corrections_path):
            return "No corrections found to apply. Run self-correction review first.", get_display_data_for_product_df(product_name)

        try:
            with open(temp_corrections_path, "r", encoding="utf-8") as f:
                corrections_list = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load temporary corrections for {product_name}: {e}")
            return f"Failed to load temporary corrections: {e}", get_display_data_for_product_df(product_name)

        progress(0.5, desc=f"Applying corrections for {product_name}...")
        corrected_answers = apply_corrections(original_data_dict["answers"], corrections_list, product_name)

        original_data_dict["answers"] = corrected_answers
        save_extracted_data(product_name, original_data_dict)

        logger.info(f"Product {product_name} extraction_status should be updated to 'Corrected'.")


        try:
            os.remove(temp_corrections_path)
        except OSError as e:
            logger.warning(f"Could not remove temporary corrections file {temp_corrections_path}: {e}")

        progress(1.0, desc="Corrections applied.")
        return f"Corrections applied and saved for {product_name}.", get_display_data_for_product_df(product_name)


    def check_products_truncation_status(current_app_state_value: Dict[str, Any]) -> str:
        """Checks document sizes and reports which might be truncated."""
        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])
        if not products_list:
            return "No products available to check for truncation."

        status_lines = ["### Document Size Status for Extraction"]
        found_processed = False
        for p_entry in products_list:
            if p_entry.get('status') == "Processed to Markdown":
                found_processed = True
                needs_truncation, size_chars = check_document_size(p_entry['name'])
                size_kb = size_chars / 1024
                status_icon = "🔴 Will be truncated" if needs_truncation else "🟢 Fits context"
                status_lines.append(f"- **{p_entry['name']}**: {status_icon} (Size: {size_kb:.1f} KB)")

        if not found_processed:
            return "No products currently processed to Markdown for size checking."
        return "\n".join(status_lines)


    with gr.Blocks() as extraction_tab_ui:
        with gr.Row():
            model_choice_ui = gr.Dropdown(
                label="Model for Extraction & Correction",
                choices=[AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT, AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT],
                value=AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT,
                info="Select model for answer extraction and self-correction review."
            )
        truncation_status_ui = gr.Markdown("Loading document size status...")
        extract_all_button_ui = gr.Button("Extract Answers for All Ready Products", variant="primary")
        extraction_status_ui = gr.Markdown("")

        gr.Markdown("### Review and Correct Extracted Data")

        def get_extracted_product_names_for_dropdown(current_app_state_value: Dict[str, Any]) -> gr.Dropdown:
            """Populates dropdown with products that have extracted data."""
            products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])

            extracted_product_names = sorted(list(set(
                p['name'] for p in products_list
                if p.get('extraction_status') in ["Extracted", "Corrected", "review_suggested"]
            )))

            if not extracted_product_names and os.path.exists(EXTRACTED_DATA_DIR):
                disk_extracted_names = []
                for f_name in os.listdir(EXTRACTED_DATA_DIR):
                    if f_name.endswith("_extracted.json"):
                        product_name_from_file = f_name.replace("_extracted.json", "").replace("_", " ").title()
                        data = load_extracted_data(product_name_from_file)
                        if data and data.get("product_name_original"):
                             disk_extracted_names.append(data["product_name_original"])
                        elif data:
                             disk_extracted_names.append(product_name_from_file)


                extracted_product_names = sorted(list(set(disk_extracted_names)))

            return gr.Dropdown(
                choices=extracted_product_names,
                label="Select Product to Review/Correct Answers",
                interactive=True,
                value=extracted_product_names[0] if extracted_product_names else None
            )

        product_select_for_review_ui = gr.Dropdown(
            label="Select Product to Review/Correct Answers", interactive=True, choices=[]
        )
        extracted_answers_df_ui = gr.DataFrame(
            label="Extracted Answers", interactive=False,
            headers=["Question ID", "Question Text", "Category", "Answer", "Status"]
        )
        with gr.Row():
            self_correct_button_ui = gr.Button("Run AI Self-Correction Review", variant="secondary")
            apply_corrections_button_ui = gr.Button("Apply Suggested Corrections", variant="primary")
        correction_status_ui = gr.Markdown("")

        extract_all_button_ui.click(
            handle_extract_all_answers_action,
            inputs=[app_state, model_choice_ui],
            outputs=[app_state, extraction_status_ui, product_select_for_review_ui]
        ).then(
            check_products_truncation_status,
            inputs=[app_state],
            outputs=[truncation_status_ui]
        )

        product_select_for_review_ui.change(
            get_display_data_for_product_df,
            inputs=[product_select_for_review_ui],
            outputs=[extracted_answers_df_ui]
        ).then(lambda: "", outputs=[correction_status_ui])

        self_correct_button_ui.click(
            handle_self_correction_action,
            inputs=[product_select_for_review_ui, model_choice_ui],
            outputs=[correction_status_ui, extracted_answers_df_ui]
        )

        apply_corrections_button_ui.click(
            handle_apply_corrections_action_button,
            inputs=[product_select_for_review_ui],
            outputs=[correction_status_ui, extracted_answers_df_ui]
        )

        extraction_tab_ui.load(
            lambda current_app_state_val: (
                check_products_truncation_status(current_app_state_val),
                get_extracted_product_names_for_dropdown(current_app_state_val)
            ),
            inputs=[app_state],
            outputs=[truncation_status_ui, product_select_for_review_ui]
        )

    return extraction_tab_ui
