import gradio as gr
import pandas as pd
import os
from config import logger, EXTRACTED_DATA_DIR
from local_storage import list_saved_products, load_extracted_data, export_data_to_excel, load_questions_config
from typing import List, Dict, Any, Tuple, Optional
from utils import format_error_message
import base64
from agent_service import ANALYSIS_TEMPLATES
from analyzer import execute_analysis_with_agent
from azure_clients import call_llm
from config import AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT
import re
from analysis_storage import save_analysis, list_saved_analyses, load_analysis, delete_analysis

try:
    import markdown as _md
except ImportError:
    _md = None

def create_analysis_tab(app_state: gr.State):
    """Creates the Gradio UI for the Analysis tab."""

    if app_state.value is None: app_state.value = {}
    if 'questions_config' not in app_state.value:
        app_state.value['questions_config'] = load_questions_config()

    def load_all_product_comparison_data() -> List[Dict[str, Any]]:
        all_data = []
        product_names_to_load = set()
        if 'products_list' in app_state.value:
            for p_entry in app_state.value['products_list']:
                if p_entry.get('extraction_status') in ["Extracted", "Corrected"] or \
                   load_extracted_data(p_entry['name']):
                    product_names_to_load.add(p_entry['name'])

        if not product_names_to_load:
            stored_product_files = os.listdir(EXTRACTED_DATA_DIR)
            for f_name in stored_product_files:
                if f_name.endswith("_extracted.json"):
                    pass

        for product_name in product_names_to_load:
            data = load_extracted_data(product_name)
            if data:
                all_data.append(data)
        return all_data

    def handle_generate_analysis(analysis_prompt_text: str, progress=gr.Progress(track_tqdm=True)) -> Tuple[str, str, List[Any]]:
        all_data = load_all_product_comparison_data()
        if not all_data:
            return "", "No data available for analysis.", []

        progress(0.3, desc="Running analysis with AI agent...")
        results = execute_analysis_with_agent(analysis_prompt_text, all_data)

        if results.get("error"):
            return "", results["error"], []

        html_outputs = []

        for plot in results.get("plots", []):
            if isinstance(plot, dict) and plot.get("image_base64"):
                html_outputs.append(gr.HTML(f"""
                    <div class='analysis-plot'>
                        <h3>{plot.get('title', 'Analysis Plot')}</h3>
                        <img src='data:image/png;base64,{plot["image_base64"]}'
                             alt='{plot.get("title", "Plot")}'
                             style='max-width:100%; height:auto; margin:10px 0;'/>
                    </div>
                """))

        for table in results.get("tables", []):
            if isinstance(table, dict) and table.get("data_html"):
                html_outputs.append(gr.HTML(f"""
                    <div class='analysis-table'>
                        <h3>{table.get('title', 'Analysis Table')}</h3>
                        <div style='overflow-x:auto;'>{table["data_html"]}</div>
                    </div>
                """))

        if results.get("explanation"):
            html_outputs.append(gr.Markdown(results["explanation"]))

        return "", "Analysis complete!", html_outputs

    def handle_export_excel() -> Tuple[Optional[str], str]:
        all_data = load_all_product_comparison_data()
        questions_cfg = app_state.value.get('questions_config')

        if not all_data:
            return None, "No data to export."
        if not questions_cfg:
            return None, "Questions configuration not found, cannot map question IDs to text for export."

        file_path = export_data_to_excel(all_data, questions_cfg)
        if file_path:
            return file_path, f"Data exported to {file_path}"
        else:
            return None, "Failed to export data to Excel. Check logs."

    def _html_join(parts: List[str]) -> str:
        """Joins a list of HTML / markdown snippets into one HTML string."""
        return "\n".join(parts)

    def summarize_with_llm(text: str) -> str:
        """
        Summarise arbitrary text to 1-2 plain sentences via Azure OpenAI.
        Falls back to heuristic trimming on failure.
        """
        if not text or len(text) < 50:
            return text.strip().split("\n")[0][:250]

        prompt = [
            {"role": "system", "content":
                "You are an assistant that writes very concise, plain-text summaries of the key insights. "
                "(1-2 sentences, NO markdown)."},
            {"role": "user", "content": text[:8000]}
        ]
        resp = call_llm(prompt, AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT,
                        temperature=0.2, max_tokens=120)
        if resp:
            return resp.strip().replace("\n", " ")
        plain = re.sub(r'[#*_`]', '', text)[:500]
        sent = plain.split('.')
        return '.'.join(sent[:2]) + '.'

    def get_analysis_summary(explanation: str, title: str) -> str:
        """Get a 1-2 sentence plain-text summary of the analysis using LLM."""
        return summarize_with_llm(explanation or f"{title}: no explanation.")

    def create_accordion_section(title: str, summary: str, content: str) -> str:
        """Create an accordion section with summary and content"""
        safe_id = ''.join(c if c.isalnum() else '_' for c in title.lower())
        accordion_id = f"acc_{safe_id}"

        return f"""
        <div class="analysis-accordion">
            <div class="accordion-header" data-target="{accordion_id}">
                <h3>{title}</h3>
                <div class="summary">{summary}</div>
                <div class="accordion-icon">▼</div>
            </div>
            <div id="{accordion_id}" class="accordion-content">
                {content}
            </div>
        </div>
        """

    def format_markdown_content(markdown_text: str) -> str:
        """Convert markdown to HTML so it renders correctly inside gr.HTML."""
        if not markdown_text:
            return ""
        try:
            if _md:
                rendered_html = _md.markdown(
                    markdown_text,
                    extensions=["tables", "fenced_code", "toc", "sane_lists"]
                )
            else:
                rendered_html = (
                    markdown_text.replace("\n", "<br>")
                                  .replace("**", "<b>").replace("__", "<i>")
                )
        except Exception as e:
            logger.error(f"Markdown render error: {e}")
            rendered_html = markdown_text.replace("\n", "<br>")

        return f"""
        <div class='analysis-explanation'>
            {rendered_html}
        </div>
        """

    def run_selected_analyses(
        selected_analyses: List[str],
        all_products_data: List[Dict[str, Any]],
        progress=gr.Progress()
    ) -> Tuple[str, gr.update, Dict[str, str]]:
        if not selected_analyses:
            return "Please select at least one analysis type.", gr.update(choices=[], value=None), {}

        if not all_products_data:
            return "No product data available for analysis.", gr.update(choices=[], value=None), {}

        logger.info(f"Running {len(selected_analyses)} selected analyses")
        progress(0.1, "Preparing analyses...")
        detail_map: Dict[str, str] = {}
        total = len(selected_analyses)

        for idx, analysis_key in enumerate(selected_analyses):
            template = ANALYSIS_TEMPLATES.get(analysis_key)
            if not template:
                continue

            progress((idx + 1) / total, f"Running {template['name']}...")
            logger.info(f"Processing analysis template: {template['name']}")

            try:
                results = execute_analysis_with_agent(template["prompt"], all_products_data)
                logger.info(f"Got results for {template['name']}: {len(results.get('plots', []))} plots, {len(results.get('tables', []))} tables")

                if results.get("error"):
                    logger.error(f"Error in {template['name']}: {results['error']}")
                    continue

                section_content_parts = []

                for plot in results.get("plots", []):
                    if plot.get("image_base64"):
                        logger.info(f"Adding plot: {plot.get('title', 'Untitled')}")
                        section_content_parts.append(f"""
                            <div class='analysis-plot'>
                                <h4>{plot.get('title', 'Analysis Plot')}</h4>
                                <img src='data:image/png;base64,{plot["image_base64"]}'
                                     alt='{plot.get('title', "Plot")}'
                                     style='max-width:100%; height:auto; margin:10px 0;'/>
                            </div>
                        """)

                for table in results.get("tables", []):
                    if table.get("data_html"):
                        logger.info("Adding table to output")
                        section_content_parts.append(f"""
                            <div class='analysis-table'>
                                <h4>{table.get('title', 'Analysis Table')}</h4>
                                <div style='overflow-x:auto;'>{table["data_html"]}</div>
                            </div>
                        """)

                explanation_html = ""
                if results.get("explanation"):
                    logger.info("Adding explanation text")
                    explanation_html = format_markdown_content(results["explanation"])
                    section_content_parts.append(explanation_html)

                section_content = "".join(section_content_parts)

                detail_map[template['name']] = section_content

            except Exception as e:
                logger.error(f"Error in analysis {template['name']}: {e}")

        if not detail_map:
            return (
                "No analysis results generated.",
                gr.update(choices=[], value=None),
                {}
            )

        logger.info(f"Analysis complete with {len(detail_map)} output components")
        return (
            "Analyses complete!",
            _mk_dropdown_update(detail_map),
            detail_map
        )

    def run_selected_analyses_with_custom(
        selected_analyses: List[str],
        custom_prompt: str,
        all_products_data: List[Dict[str, Any]],
        progress=gr.Progress()
    ) -> Tuple[str, gr.update, Dict[str, str]]:
        """Runs selected analyses plus custom analysis."""
        detail_map: Dict[str, str] = {}

        total_analyses = len(selected_analyses) + (1 if custom_prompt else 0)
        current_progress = 0

        if selected_analyses:
            progress(0.1, "Running predefined analyses...")
            for idx, analysis_key in enumerate(selected_analyses):
                template = ANALYSIS_TEMPLATES.get(analysis_key)
                if not template:
                    continue

                current_progress = (idx + 1) / total_analyses
                progress(current_progress, f"Running {template['name']}...")

                try:
                    results = execute_analysis_with_agent(template["prompt"], all_products_data)

                    section_content_parts = []

                    for plot in results.get("image_base64", []):
                        if plot.get("image_base64"):
                            section_content_parts.append(f"""
                                <div class='analysis-plot'>
                                    <h4>{plot.get('title', 'Analysis Plot')}</h4>
                                    <img src='data:image/png;base64,{plot["image_base64"]}'
                                         alt='{plot.get('title', "Plot")}'
                                         style='max-width:100%; height:auto; margin:10px 0;'/>
                                </div>
                            """)

                    for table in results.get("data_html", []):
                        if table.get("data_html"):
                            section_content_parts.append(f"""
                                <div class='analysis-table'>
                                    <h4>{table.get('title', 'Analysis Table')}</h4>
                                    <div style='overflow-x:auto;'>{table["data_html"]}</div>
                                </div>
                            """)

                    if results.get("explanation"):
                        explanation_html = format_markdown_content(results["explanation"])
                        section_content_parts.append(explanation_html)

                    section_content = "".join(section_content_parts)

                    detail_map[template['name']] = section_content

                except Exception as e:
                    logger.error(f"Error in analysis {template['name']}: {e}")

        if custom_prompt:
            progress(current_progress, "Running custom analysis...")
            try:
                results = execute_analysis_with_agent(custom_prompt, all_products_data)

                section_content_parts = []

                for plot in results.get("image_base64", []):
                    if plot.get("image_base64"):
                        section_content_parts.append(f"""
                            <div class='analysis-plot'>
                                <h4>{plot.get('title', 'Analysis Plot')}</h4>
                                <img src='data:image/png;base64,{plot["image_base64"]}'
                                     alt='{plot.get('title', "Plot")}'
                                     style='max-width:100%; height:auto; margin:10px 0;'/>
                            </div>
                        """)

                for table in results.get("data_html", []):
                    if table.get("data_html"):
                        section_content_parts.append(f"""
                            <div class='analysis-table'>
                                <h4>{table.get('title', 'Analysis Table')}</h4>
                                <div style='overflow-x:auto;'>{table["data_html"]}</div>
                            </div>
                        """)

                if results.get("explanation"):
                    explanation_html = format_markdown_content(results["explanation"])
                    section_content_parts.append(explanation_html)

                section_content = "".join(section_content_parts)

                detail_map["Custom Analysis"] = section_content

            except Exception as e:
                logger.error(f"Error in custom analysis: {e}")

        if not detail_map:
            return (
                "No results generated.",
                gr.update(choices=[], value=None),
                {}
            )

        return (
            "Analyses complete!",
            _mk_dropdown_update(detail_map),
            detail_map
        )

    def get_selected_analyses(checkbox_states):
        """Gets list of selected analysis keys from checkbox states."""
        return [key for key, value in checkbox_states.items() if value]

    with gr.Blocks() as analysis_tab:


        with gr.Row():
            with gr.Column(elem_classes="analysis-container"):
                gr.Markdown("### Select Pre-defined Analyses")

                with gr.Row(elem_classes="analysis-options"):
                    analysis_checkboxes = {}
                    checkbox_outputs = []

                    for key, template in ANALYSIS_TEMPLATES.items():
                        with gr.Column(elem_classes="analysis-option", scale=1):
                            analysis_checkboxes[key] = gr.Checkbox(
                                label=template["name"],
                                info=template["description"],
                                value=False
                            )
                            checkbox_outputs.append(analysis_checkboxes[key])

                with gr.Accordion("Create Custom Analysis", open=False):
                    custom_analysis_prompt = gr.Textbox(
                        label="Custom Analysis Description",
                        placeholder="Describe the analysis you want, e.g., 'Create a horizontal bar chart comparing the maximum covered amounts for water damage across all products.'",
                        lines=3
                    )
                    custom_analysis_guide = gr.Markdown("""
                    **Guide for Custom Analysis:**
                    - Be specific about the visualization type (bar chart, scatter plot, table, etc.)
                    - Mention which aspects to compare (coverage amounts, deductibles, etc.)
                    - Specify any formatting preferences (sorting, grouping, etc.)
                    """)
                    custom_analysis_active = gr.Checkbox(label="Include Custom Analysis", value=False)
                    checkbox_outputs.append(custom_analysis_active)

                gr.HTML("<div style='height: 20px;'></div>")

                with gr.Row(elem_classes="analysis-btn"):
                    run_analysis_button = gr.Button("Run Selected Analyses", variant="primary", size="large")

                analysis_status = gr.Markdown("", elem_classes="status-message")

        with gr.Column(elem_classes="analysis-container") as results_container:
            gr.Markdown("### Analysis Results")

            detail_select = gr.Dropdown(
                label="Select analysis to view",
                choices=[],
                interactive=True
            )

            with gr.Accordion("Manage Analyses", open=False):
                with gr.Row():
                    with gr.Column(scale=3):
                        analysis_name = gr.Textbox(
                            label="Save current analysis as",
                            placeholder="Enter a name"
                        )
                    with gr.Column(scale=3):
                        saved_analyses = gr.Dropdown(
                            label="Previously saved analyses",
                            choices=list_saved_analyses(),
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column(scale=3):
                        save_analysis_btn = gr.Button("💾 Save")

                    with gr.Column(scale=3):
                        with gr.Row():
                            load_analysis_btn = gr.Button("📂")
                            delete_analysis_btn = gr.Button("🗑️")
                            refresh_saved_btn = gr.Button("🔄")

            detail_area = gr.HTML(value="Select an analysis to view details.")
            detail_state = gr.State({})

        # Add handlers for save/load functionality
        def save_current_analysis(detail_map: Dict[str, str], name: str) -> str:
            if not detail_map:
                return "No analysis results to save."
            filename = save_analysis(detail_map, name)
            return f"Analysis saved as {filename}"

        def load_saved_analysis(filename: str) -> Tuple[str, gr.update, Dict[str, str]]:
            if not filename:
                return "Please select an analysis to load.", gr.update(), {}
            data = load_analysis(filename)
            if not data:
                return "Failed to load analysis.", gr.update(), {}
            return "Analysis loaded successfully!", _mk_dropdown_update(data), data

        def delete_saved_analysis(filename: str) -> Tuple[str, gr.update]:
            if not filename:
                return "Please select an analysis to delete.", gr.update(choices=list_saved_analyses())
            if delete_analysis(filename):
                return f"Deleted {filename}", gr.update(choices=list_saved_analyses(), value=None)
            return "Failed to delete analysis.", gr.update(choices=list_saved_analyses())

        def refresh_saved_analyses():
            return gr.update(choices=list_saved_analyses())

        save_analysis_btn.click(
            save_current_analysis,
            inputs=[detail_state, analysis_name],
            outputs=[analysis_status]
        )

        load_analysis_btn.click(
            load_saved_analysis,
            inputs=[saved_analyses],
            outputs=[analysis_status, detail_select, detail_state]
        )

        delete_analysis_btn.click(
            delete_saved_analysis,
            inputs=[saved_analyses],
            outputs=[analysis_status, saved_analyses]
        )

        refresh_saved_btn.click(
            refresh_saved_analyses,
            outputs=[saved_analyses]
        )

        def run_analyses_with_custom(*states):
            checkbox_states = dict(zip(ANALYSIS_TEMPLATES.keys(), states[:-2]))
            custom_prompt = states[-2]
            include_custom = states[-1]

            selected_analyses = get_selected_analyses(checkbox_states)
            all_data = load_all_product_comparison_data()

            if not (selected_analyses or (include_custom and custom_prompt.strip())):
                return "Please select at least one analysis or provide a custom analysis.", gr.update(choices=[], value=None), {}

            if include_custom and custom_prompt.strip():
                return run_selected_analyses_with_custom(selected_analyses, custom_prompt, all_data)
            else:
                return run_selected_analyses(selected_analyses, all_data)

        run_analysis_button.click(
            run_analyses_with_custom,
            inputs=checkbox_outputs + [custom_analysis_prompt, custom_analysis_active],
            outputs=[analysis_status, detail_select, detail_state]
        )

        def _show_detail(det_state: Dict[str, str], sel: str) -> str:
            return det_state.get(sel, "<div>Select an analysis to view details.</div>")

        detail_select.change(
            _show_detail,
            inputs=[detail_state, detail_select],
            outputs=[detail_area]
        )

        gr.Markdown("---")
        with gr.Row():
            export_excel_button = gr.Button("Export All Data to Excel", variant="secondary")
            export_status = gr.File(label="Download Excel Export", interactive=False)
            export_message = gr.Markdown("")

        export_excel_button.click(
            handle_export_excel,
            outputs=[export_status, export_message]
        )

    return analysis_tab

def _mk_dropdown_update(detail_map: Dict[str, str]):
    """Return a Gradio update dict for the detail dropdown."""
    choices = list(detail_map.keys())
    return gr.update(choices=choices, value=(choices[0] if choices else None))