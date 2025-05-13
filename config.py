"""
Configuration management for the Insurance Comparison Assistant.
Loads environment variables and sets up constants.
"""
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT", "o4-mini")
AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_NONREASONING_MODEL_DEPLOYMENT", AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT
)

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRODUCTS_DIR = os.path.join(DATA_DIR, "insurance_products")
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted_data")
QUESTIONS_CONFIG_PATH = os.path.join(DATA_DIR, "questions_config.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

DEFAULT_PRODUCTS: List[Dict[str, Any]] = []
for i in range(1, 11):
    name_key = f"DEFAULT_PRODUCT_{i}_NAME"
    urls_key = f"DEFAULT_PRODUCT_{i}_URLS"
    name = os.getenv(name_key)
    urls_str = os.getenv(urls_key)

    logger.debug(f"Loading product {i}: name={name}, urls={urls_str}")

    if name and urls_str:
        urls = []
        for url in urls_str.split(','):
            url = url.strip()
            if url:
                url = url.replace("%3A", ":")
                urls.append(url)

        if urls:
            product = {"name": name, "pdf_urls": urls}
            DEFAULT_PRODUCTS.append(product)
            logger.info(f"Loaded default product: {name} with {len(urls)} URLs")

logger.info(f"Loaded {len(DEFAULT_PRODUCTS)} default products")

DEFAULT_SAMPLE_CATEGORIES_STR = os.getenv("DEFAULT_SAMPLE_CATEGORIES", "")
DEFAULT_SAMPLE_QUESTIONS_STR = os.getenv("DEFAULT_SAMPLE_QUESTIONS", "")

def get_default_categories() -> List[str]:
    """Returns default categories as a list from comma-separated environment variable."""
    logger.debug(f"Raw default categories from env: {DEFAULT_SAMPLE_CATEGORIES_STR[:100]}...")
    if not DEFAULT_SAMPLE_CATEGORIES_STR:
        logger.warning("No default categories found in environment variables (DEFAULT_SAMPLE_CATEGORIES).")
        return []
    return [cat.strip() for cat in DEFAULT_SAMPLE_CATEGORIES_STR.split(',') if cat.strip()]

def get_default_questions() -> List[str]:
    """Returns default questions as a list from comma-separated environment variable."""
    logger.debug(f"Raw default questions from env: {DEFAULT_SAMPLE_QUESTIONS_STR[:100]}...")
    if not DEFAULT_SAMPLE_QUESTIONS_STR:
        logger.warning("No default questions found in environment variables (DEFAULT_SAMPLE_QUESTIONS).")
        return []
    return [q.strip() for q in DEFAULT_SAMPLE_QUESTIONS_STR.split(',') if q.strip()]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRODUCTS_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)

def get_azure_config() -> Dict[str, str]:
    """Returns the current Azure configurations, loaded from settings.json or .env."""
    from local_storage import load_settings
    return load_settings()

logger.info("Configuration loaded.")

_initial_settings_from_env = {
    "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
    "azure_openai_api_key": AZURE_OPENAI_API_KEY,
    "azure_document_intelligence_endpoint": AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    "azure_document_intelligence_key": AZURE_DOCUMENT_INTELLIGENCE_KEY,
}
if not all(_initial_settings_from_env.values()):
    if not os.path.exists(SETTINGS_PATH):
        logger.warning(
            "One or more Azure service credentials are not set directly in the .env file. "
            "Ensure they are configured via the Settings UI or that data/settings.json exists and is populated."
        )
