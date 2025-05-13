
from typing import Tuple

import gradio as gr
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from config import (AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                   AZURE_DOCUMENT_INTELLIGENCE_KEY, AZURE_OPENAI_API_KEY,
                   AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT,
                   AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT,
                   AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT, logger)
from local_storage import load_settings, save_settings


def create_settings_tab() -> gr.Blocks:
    """Creates the Gradio UI for the Settings tab."""

    _azure_openai_client = None
    _document_intelligence_client = None

    def initialize_clients():
        nonlocal _azure_openai_client, _document_intelligence_client
        try:
            _azure_openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            _document_intelligence_client = DocumentIntelligenceClient(
                endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
            )
            return "✅ Successfully initialized Azure clients"
        except Exception as e:
            return f"❌ Failed to initialize Azure clients: {str(e)}"

    def handle_save_settings_action(
        endpoint_openai: str, key_openai: str, version_openai: str,
        dep_reasoning: str, dep_nonreasoning: str,
        endpoint_di: str, key_di: str
    ) -> str:
        """Saves settings to local file and resets client instances."""
        nonlocal _azure_openai_client, _document_intelligence_client

        new_settings = {
            "azure_openai_endpoint": endpoint_openai.strip(),
            "azure_openai_api_key": key_openai,
            "azure_openai_api_version": version_openai.strip(),
            "azure_openai_reasoning_model_deployment": dep_reasoning.strip(),
            "azure_openai_nonreasoning_model_deployment": dep_nonreasoning.strip(),
            "azure_document_intelligence_endpoint": endpoint_di.strip(),
            "azure_document_intelligence_key": key_di,
        }
        save_settings(new_settings)

        _azure_openai_client = None
        _document_intelligence_client = None
        logger.info("Settings saved. Azure clients will be re-initialized on next use.")
        return "Settings saved successfully. Clients will use new settings on their next operation."

    def handle_reload_settings_ui_action() -> Tuple[str, str, str, str, str, str, str, str]:
        """Reloads settings from file/defaults and updates UI fields."""
        reloaded_settings = load_settings()
        return (
            reloaded_settings.get("azure_openai_endpoint", ""),
            reloaded_settings.get("azure_openai_api_key", ""),
            reloaded_settings.get("azure_openai_api_version", "2024-12-01-preview"),
            reloaded_settings.get("azure_openai_reasoning_model_deployment", AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT),
            reloaded_settings.get("azure_openai_nonreasoning_model_deployment", AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT),
            reloaded_settings.get("azure_document_intelligence_endpoint", ""),
            reloaded_settings.get("azure_document_intelligence_key", ""),
            "Settings reloaded from file/defaults and UI updated."
        )

    with gr.Blocks() as settings_tab_ui:
        gr.Markdown("## Azure Service Configuration")
        gr.Markdown(
            "Configure Azure OpenAI and Document Intelligence credentials. "
            "These are saved locally in `data/settings.json` and override `.env` values if set."
        )

        current_settings = load_settings()

        with gr.Accordion("Azure OpenAI Service", open=True):
            azure_openai_endpoint_ui = gr.Textbox(
                label="Azure OpenAI Endpoint",
                value=current_settings.get("azure_openai_endpoint", ""),
                info="e.g., https://your-resource-name.openai.azure.com/"
            )
            azure_openai_api_key_ui = gr.Textbox(
                label="Azure OpenAI API Key", type="password",
                value=current_settings.get("azure_openai_api_key", "")
            )
            azure_openai_api_version_ui = gr.Textbox(
                label="Azure OpenAI API Version",
                value=current_settings.get("azure_openai_api_version", "2024-12-01-preview")
            )
            azure_openai_reasoning_model_deployment_ui = gr.Textbox(
                label="Reasoning Model Deployment Name",
                value=current_settings.get("azure_openai_reasoning_model_deployment", AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT),
                info="For complex tasks (extraction, analysis)."
            )
            azure_openai_nonreasoning_model_deployment_ui = gr.Textbox(
                label="Non-Reasoning Model Deployment Name",
                value=current_settings.get("azure_openai_nonreasoning_model_deployment", AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT),
                info="For simpler tasks (e.g. summarization, question generation)."
            )

        with gr.Accordion("Azure Document Intelligence Service", open=True):
            azure_document_intelligence_endpoint_ui = gr.Textbox(
                label="Document Intelligence Endpoint",
                value=current_settings.get("azure_document_intelligence_endpoint", ""),
                info="e.g., https://your-di-resource.cognitiveservices.azure.com/"
            )
            azure_document_intelligence_key_ui = gr.Textbox(
                label="Document Intelligence API Key", type="password",
                value=current_settings.get("azure_document_intelligence_key", "")
            )

        save_settings_button = gr.Button("Save Settings", variant="primary")
        settings_status_ui = gr.Markdown("")
        reload_settings_button = gr.Button("Reload Settings from File/Defaults")

        save_settings_button.click(
            handle_save_settings_action,
            inputs=[
                azure_openai_endpoint_ui, azure_openai_api_key_ui, azure_openai_api_version_ui,
                azure_openai_reasoning_model_deployment_ui, azure_openai_nonreasoning_model_deployment_ui,
                azure_document_intelligence_endpoint_ui, azure_document_intelligence_key_ui
            ],
            outputs=[settings_status_ui]
        )

        reload_settings_button.click(
            handle_reload_settings_ui_action,
            outputs=[
                azure_openai_endpoint_ui, azure_openai_api_key_ui, azure_openai_api_version_ui,
                azure_openai_reasoning_model_deployment_ui, azure_openai_nonreasoning_model_deployment_ui,
                azure_document_intelligence_endpoint_ui, azure_document_intelligence_key_ui,
                settings_status_ui
            ]
        )

        initialize_button = gr.Button("Initialize Azure Clients")
        status_text = gr.Textbox(label="Status", interactive=False)

        initialize_button.click(
            fn=initialize_clients,
            outputs=status_text
        )

    return settings_tab_ui
