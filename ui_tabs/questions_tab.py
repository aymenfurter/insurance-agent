
import os
import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

from config import (AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT,
                    AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT,
                    get_default_categories, get_default_questions, logger)
from local_storage import (get_product_data_dir, list_saved_products,
                           load_markdown_page, load_product_config,
                           load_questions_config, save_questions_config)
from question_manager import suggest_categories_and_questions


def create_questions_tab(app_state: gr.State) -> gr.Blocks:
    """Creates the Gradio UI for the Question & Category Configuration tab."""

    if app_state.value is None: app_state.value = {}
    if 'questions_config' not in app_state.value or not app_state.value['questions_config']:
        app_state.value['questions_config'] = load_questions_config()
        if not app_state.value['questions_config'].get('categories') and \
           not app_state.value['questions_config'].get('questions'):
            logger.info("Initialized empty questions_config in app_state from load_questions_config.")


    def update_and_save_app_state_questions_config(current_app_state_value: Dict[str, Any], new_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates questions_config in app_state and saves it to disk."""
        current_app_state_value['questions_config'] = new_config_data
        save_questions_config(new_config_data)
        return current_app_state_value

    def get_all_markdown_content_for_suggestion(current_app_state_value: Dict[str, Any]) -> Dict[str, List[str]]:
        """Aggregates markdown content from all processed products for question suggestion."""
        all_md_content_map: Dict[str, List[str]] = {}
        products_list: List[Dict[str, Any]] = current_app_state_value.get('products_list', [])

        product_names_to_process = set()
        for prod_entry in products_list:
            if prod_entry.get('name') and prod_entry.get('status') == "Processed to Markdown" and prod_entry.get('markdown_docs'):
                product_names_to_process.add(prod_entry['name'])

        saved_product_dirs = list_saved_products()
        for product_dir_name in saved_product_dirs:
            config = load_product_config(product_dir_name)
            if config and config.get("product_name") and config.get("markdown_document_infos"):
                product_names_to_process.add(config["product_name"])

        logger.info(f"Products considered for markdown aggregation: {product_names_to_process}")

        for product_name_orig in product_names_to_process:
            product_md_list: List[str] = []
            prod_from_state = next((p for p in products_list if p.get('name') == product_name_orig), None)
            product_data_dir = get_product_data_dir(product_name_orig)

            if prod_from_state and prod_from_state.get('markdown_docs'):
                logger.info(f"Using markdown_docs from app_state for {product_name_orig}")
                for md_doc_info in prod_from_state['markdown_docs']:
                    for rel_page_path in md_doc_info.get('markdown_pages_paths', []):
                        if rel_page_path:
                            abs_page_path = os.path.join(product_data_dir, rel_page_path)
                            page_content = load_markdown_page(abs_page_path)
                            if page_content:
                                product_md_list.append(page_content)

            if not product_md_list:
                logger.info(f"Trying to load markdown_docs from disk config for {product_name_orig}")
                disk_config = load_product_config(product_name_orig) # load_product_config handles name variants
                if disk_config and disk_config.get('markdown_document_infos'):
                    for md_doc_info in disk_config['markdown_document_infos']:
                        for abs_page_path in md_doc_info.get('markdown_pages_paths_absolute', []):
                            if abs_page_path:
                                page_content = load_markdown_page(abs_page_path)
                                if page_content:
                                    product_md_list.append(page_content)

            if product_md_list:
                logger.info(f"Loaded {len(product_md_list)} markdown pages for {product_name_orig}")
                all_md_content_map[product_name_orig] = product_md_list
            else:
                logger.warning(f"No markdown content found for {product_name_orig}")

        return all_md_content_map

    def format_questions_for_df(questions_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Formats the list of question dicts into a DataFrame for display."""
        if not questions_list:
            return pd.DataFrame(columns=["Select", "ID", "Question Text", "Applies to Categories"])

        df_data = [
            [
                False,
                q.get('id', ''),
                q.get('text', ''),
                ", ".join(q.get('applies_to_categories', []))
            ] for q in questions_list
        ]
        return pd.DataFrame(df_data, columns=["Select", "ID", "Question Text", "Applies to Categories"])

    def handle_suggest_questions_action(
        current_app_state_value: Dict[str, Any], sample_categories: str, sample_questions: str, model_choice: str
    ) -> Tuple[Dict[str, Any], str, pd.DataFrame, List[str]]:
        """Suggests categories and questions based on processed documents."""
        all_md_content = get_all_markdown_content_for_suggestion(current_app_state_value)
        if not all_md_content:
            questions_config = current_app_state_value.get('questions_config', {"categories": [], "questions": []})
            df_val = format_questions_for_df(questions_config.get('questions', []))
            return current_app_state_value, "No Markdown content found from processed products to suggest questions.", df_val, questions_config.get('categories', [])

        suggested_config = suggest_categories_and_questions(all_md_content, sample_categories, sample_questions, model_choice)

        if suggested_config and suggested_config.get('categories'):
            updated_app_state = update_and_save_app_state_questions_config(current_app_state_value, suggested_config)
            status = "Successfully suggested categories and questions."
            df_val = format_questions_for_df(suggested_config.get('questions', []))
            return updated_app_state, status, df_val, suggested_config.get('categories', [])
        else:
            status = "Failed to suggest categories and questions, or no categories were generated. Check logs."
            questions_config = current_app_state_value.get('questions_config', {"categories": [], "questions": []})
            df_val = format_questions_for_df(questions_config.get('questions', []))
            return current_app_state_value, status, df_val, questions_config.get('categories', [])


    def add_category_action(
        current_app_state_value: Dict[str, Any], new_category_name: str
    ) -> Tuple[Dict[str, Any], str, List[str]]:
        """Adds a new category to the configuration."""
        questions_config = current_app_state_value.get('questions_config', {"categories": [], "questions": []})
        current_categories = questions_config.get('categories', [])

        if not new_category_name.strip():
            return current_app_state_value, "Category name cannot be empty.", current_categories

        cleaned_name = new_category_name.strip()
        if cleaned_name in current_categories:
            return current_app_state_value, f"Category '{cleaned_name}' already exists.", current_categories

        current_categories.append(cleaned_name)
        current_categories.sort()
        questions_config['categories'] = current_categories

        updated_app_state = update_and_save_app_state_questions_config(current_app_state_value, questions_config)
        return updated_app_state, f"Category '{cleaned_name}' added.", current_categories


    def add_question_action(
        current_app_state_value: Dict[str, Any], q_text: str, q_categories: List[str]
    ) -> Tuple[Dict[str, Any], str, pd.DataFrame]:
        """Adds a new question to the configuration."""
        questions_config = current_app_state_value.get('questions_config', {"categories": [], "questions": []})
        current_questions = questions_config.get('questions', [])

        if not q_text.strip():
            return current_app_state_value, "Question text cannot be empty.", format_questions_for_df(current_questions)
        if not q_categories:
            return current_app_state_value, "Please select at least one category.", format_questions_for_df(current_questions)

        new_q_id = f"q_manual_{uuid.uuid4().hex[:4]}"
        new_question = {
            "id": new_q_id,
            "text": q_text.strip(),
            "applies_to_categories": q_categories
        }
        current_questions.append(new_question)
        questions_config['questions'] = current_questions

        updated_app_state = update_and_save_app_state_questions_config(current_app_state_value, questions_config)
        return updated_app_state, f"Question '{new_q_id}' added.", format_questions_for_df(current_questions)


    def delete_selected_questions_action(
        current_app_state_value: Dict[str, Any], questions_df_with_selection: Dict
    ) -> Tuple[Dict[str, Any], str, pd.DataFrame]:
        """Deletes questions selected in the DataFrame."""
        questions_config = current_app_state_value.get('questions_config', {"categories": [], "questions": []})
        original_questions = questions_config.get('questions', [])

        if not questions_df_with_selection or 'data' not in questions_df_with_selection or not questions_df_with_selection['data']:
            return current_app_state_value, "No questions data to process for deletion.", format_questions_for_df(original_questions)

        selected_ids_to_delete = [
            row[1] for row in questions_df_with_selection['data'] if row[0] is True
        ]

        if not selected_ids_to_delete:
            return current_app_state_value, "No questions selected for deletion.", format_questions_for_df(original_questions)

        updated_questions = [q for q in original_questions if q['id'] not in selected_ids_to_delete]
        num_deleted = len(original_questions) - len(updated_questions)
        questions_config['questions'] = updated_questions

        updated_app_state = update_and_save_app_state_questions_config(current_app_state_value, questions_config)
        return updated_app_state, f"Deleted {num_deleted} questions.", format_questions_for_df(updated_questions)


    with gr.Blocks() as questions_tab_ui:
        initial_q_config = app_state.value.get('questions_config', {"categories": [], "questions": []})

        with gr.Accordion("Question Generation Settings", open=True):
            model_choice_ui = gr.Dropdown(
                label="Question Generation Model",
                choices=[AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT, AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT],
                value=AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT, # Default to non-reasoning for this task
                info="Select model for generating categories and questions."
            )
            sample_categories_ui = gr.TextArea(
                label="Sample Categories (Optional, one per line)",
                placeholder="e.g., Fire Damage\nWater Damage\nTheft",
                info="Guide the model with example categories.",
                value="\n".join(get_default_categories())
            )
            sample_questions_ui = gr.TextArea(
                label="Sample Questions (Optional, one per line)",
                placeholder="e.g., What is the coverage limit for X?\nIs Y excluded?",
                info="Guide the model with example question formats.",
                value="\n".join(get_default_questions())
            )
        suggest_button_ui = gr.Button("Suggest Categories & Questions from Processed Documents", variant="primary")
        suggestion_status_ui = gr.Markdown("")

        gr.Markdown("### Manage Categories")
        with gr.Row():
            new_category_name_ui = gr.Textbox(label="New Category Name", scale=3)
            add_category_button_ui = gr.Button("Add Category", scale=1)

        current_categories_list = initial_q_config.get('categories', [])
        current_categories_ui = gr.CheckboxGroup(
            label="Current Categories", choices=current_categories_list, value=current_categories_list, interactive=True
        )
        category_status_ui = gr.Markdown("")

        gr.Markdown("### Manage Questions")
        current_questions_df_ui = gr.DataFrame(
            headers=["Select", "ID", "Question Text", "Applies to Categories"],
            datatype=["bool", "str", "str", "str"],
            value=format_questions_for_df(initial_q_config.get('questions', [])).values.tolist(),
            label="Current Questions"
        )
        delete_question_button_ui = gr.Button("Delete Selected Questions", variant="secondary")

        with gr.Accordion("Add New Question Manually", open=False):
            new_question_text_ui = gr.Textbox(label="Question Text")
            new_question_categories_ui = gr.CheckboxGroup(
                label="Applies to which categories?", choices=current_categories_list
            )
            add_question_button_ui = gr.Button("Add Question")
        question_status_ui = gr.Markdown("")

        suggest_button_ui.click(
            handle_suggest_questions_action,
            inputs=[app_state, sample_categories_ui, sample_questions_ui, model_choice_ui],
            outputs=[app_state, suggestion_status_ui, current_questions_df_ui, current_categories_ui]
        ).then(
            lambda current_app_state_val: gr.CheckboxGroup(choices=current_app_state_val.get('questions_config', {}).get('categories', [])),
            inputs=[app_state],
            outputs=[new_question_categories_ui]
        )

        add_category_button_ui.click(
            add_category_action,
            inputs=[app_state, new_category_name_ui],
            outputs=[app_state, category_status_ui, current_categories_ui]
        ).then(
            lambda current_app_state_val: gr.CheckboxGroup(choices=current_app_state_val.get('questions_config', {}).get('categories', [])),
            inputs=[app_state],
            outputs=[new_question_categories_ui]
        ).then(
            lambda: gr.Textbox(value=""), outputs=[new_category_name_ui]
        )

        current_categories_ui.change(
            lambda updated_cats: gr.CheckboxGroup(choices=updated_cats if updated_cats else [], value=[]),
            inputs=[current_categories_ui],
            outputs=[new_question_categories_ui]
        )

        add_question_button_ui.click(
            add_question_action,
            inputs=[app_state, new_question_text_ui, new_question_categories_ui],
            outputs=[app_state, question_status_ui, current_questions_df_ui]
        ).then(
            lambda: (gr.Textbox(value=""), gr.CheckboxGroup(value=[])),
            outputs=[new_question_text_ui, new_question_categories_ui]
        )

        delete_question_button_ui.click(
            delete_selected_questions_action,
            inputs=[app_state, current_questions_df_ui],
            outputs=[app_state, question_status_ui, current_questions_df_ui]
        )

        def load_initial_question_data(current_app_state_value: Dict[str, Any]):
            q_config = current_app_state_value.get('questions_config', {"categories": [], "questions": []})
            cats = q_config.get('categories', [])
            quests_df = format_questions_for_df(q_config.get('questions', []))
            return cats, quests_df, cats

        questions_tab_ui.load(
            load_initial_question_data,
            inputs=[app_state],
            outputs=[current_categories_ui, current_questions_df_ui, new_question_categories_ui]
        )

    return questions_tab_ui
