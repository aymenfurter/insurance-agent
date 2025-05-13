"""
Handles the initialization of Azure service clients and LLM calls.
"""
from typing import Any, Dict, List, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from config import get_azure_config, logger

_azure_openai_client: Optional[AzureOpenAI] = None
_document_intelligence_client: Optional[DocumentIntelligenceClient] = None


def get_azure_openai_client_instance() -> Optional[AzureOpenAI]:
    """Initializes and returns a singleton AzureOpenAI client instance."""
    global _azure_openai_client
    config = get_azure_config()
    if not config.get("azure_openai_endpoint") or not config.get("azure_openai_api_key"):
        logger.error("Azure OpenAI endpoint or API key is not configured.")
        return None

    if _azure_openai_client is None:
        try:
            _azure_openai_client = AzureOpenAI(
                azure_endpoint=config["azure_openai_endpoint"],
                api_key=config["azure_openai_api_key"],
                api_version=config.get("azure_openai_api_version", "2024-12-01-preview"),
            )
            logger.info("Azure OpenAI client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            _azure_openai_client = None
    return _azure_openai_client


def get_document_intelligence_client_instance() -> Optional[DocumentIntelligenceClient]:
    """Initializes and returns a singleton DocumentIntelligenceClient instance."""
    global _document_intelligence_client
    config = get_azure_config()
    if not config.get("azure_document_intelligence_endpoint") or not config.get("azure_document_intelligence_key"):
        logger.error("Azure Document Intelligence endpoint or key is not configured.")
        return None

    if _document_intelligence_client is None:
        try:
            _document_intelligence_client = DocumentIntelligenceClient(
                endpoint=config["azure_document_intelligence_endpoint"],
                credential=AzureKeyCredential(config["azure_document_intelligence_key"])
            )
            logger.info("Azure Document Intelligence client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Document Intelligence client: {e}")
            _document_intelligence_client = None
    return _document_intelligence_client


def call_llm(
    messages: List[Dict[str, str]],
    model_deployment_name: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    json_mode: bool = False
) -> Optional[str]:
    """
    Makes a call to the specified Azure OpenAI LLM.

    Args:
        messages: List of message objects for the chat.
        model_deployment_name: The deployment name of the model to use.
        temperature: Sampling temperature.
        max_tokens: Maximum number of tokens to generate.
        json_mode: Whether to enable JSON output mode.

    Returns:
        The content of the LLM's response, or None if an error occurs.
    """
    client = get_azure_openai_client_instance()
    if not client:
        logger.error("LLM client not available for call_llm.")
        return None

    try:
        completion_params: Dict[str, Any] = {
            "model": model_deployment_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if model_deployment_name == "o4-mini":
            completion_params["max_completion_tokens"] = max_tokens
            completion_params.pop("max_tokens", None)
            completion_params.pop("temperature", None)

        if json_mode:
            completion_params["response_format"] = {"type": "json_object"}


        response = client.chat.completions.create(**completion_params)

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if response.usage:
                 logger.info(
                    f"LLM call successful to {model_deployment_name}. "
                    f"Usage: Prompt Tokens: {response.usage.prompt_tokens}, "
                    f"Completion Tokens: {response.usage.completion_tokens}, "
                    f"Total Tokens: {response.usage.total_tokens}"
                )
            else:
                logger.info(f"LLM call successful to {model_deployment_name}. Usage data not available.")
            return content
        logger.warning(f"LLM call to {model_deployment_name} returned no content in choices.")
        return None

    except Exception as e:
        logger.error(f"Error calling LLM ({model_deployment_name}): {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"LLM API Response Status: {e.response.status_code}")
            try:
                logger.error(f"LLM API Response Body: {e.response.json()}")
            except ValueError:
                logger.error(f"LLM API Response Body (text): {e.response.text}")
        return None
