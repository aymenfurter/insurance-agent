"""
Handles data extraction from insurance product documents based on configured questions
and provides self-correction capabilities using LLMs.
"""
import json
import time
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from azure_clients import call_llm
from config import AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT, logger, PRODUCTS_DIR
from local_storage import load_markdown_page, load_product_config
from prompts import (EXTRACTION_SYSTEM_PROMPT_TEMPLATE,
                     EXTRACTION_USER_PROMPT_TEMPLATE,
                     SELF_CORRECTION_SYSTEM_PROMPT,
                     SELF_CORRECTION_USER_PROMPT_TEMPLATE)

MAX_MARKDOWN_CONTEXT_CHARS = 280000
MAX_LLM_RETRIES = 3
LLM_RETRY_DELAY_SECONDS = 5


def _call_llm_with_retry(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 32000,
    retries: int = MAX_LLM_RETRIES
) -> Optional[str]:
    """Helper function to call LLM with retries for JSON mode."""
    for attempt in range(retries):
        try:
            response = call_llm(messages, model, json_mode=True, max_tokens=max_tokens)
            if response:
                stripped_response = response.strip()
                if stripped_response.startswith("{") and stripped_response.endswith("}") or \
                   stripped_response.startswith("[") and stripped_response.endswith("]"):
                    return response
                logger.warning(f"LLM response on attempt {attempt + 1} is not valid JSON: {response[:100]}...")
            else:
                logger.warning(f"Empty response from LLM on attempt {attempt + 1} for model {model}.")
        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt + 1} for model {model}: {e}")

        if attempt < retries - 1:
            logger.info(f"Retrying LLM call in {LLM_RETRY_DELAY_SECONDS} seconds...")
            time.sleep(LLM_RETRY_DELAY_SECONDS)
    logger.error(f"LLM call failed after {retries} retries for model {model}.")
    return None


def _get_full_markdown_for_product(product_name: str) -> Optional[str]:
    """Loads and concatenates all markdown content for a product."""
    from utils import clean_filename

    safe_product_name = clean_filename(product_name)
    product_config = load_product_config(safe_product_name)

    if not product_config:
        logger.error(f"No config found for product: {product_name} (safe name: {safe_product_name})")
        return None
    if 'markdown_document_infos' not in product_config:
        logger.error(f"No markdown document info in config for product: {product_name}")
        return None

    full_md_content_parts: List[str] = []
    for doc_info in product_config['markdown_document_infos']:
        doc_name = doc_info.get('doc_name', 'unnamed_document')
        full_md_content_parts.append(f"\n\n--- Content from Document: {doc_name} ---\n")

        page_paths = doc_info.get('markdown_pages_paths_absolute', []) or doc_info.get('markdown_pages_paths', [])
        if not page_paths:
            logger.warning(f"No markdown page paths for doc {doc_name} in product {product_name}")
            continue

        product_dir = os.path.join(PRODUCTS_DIR, safe_product_name)

        for page_path in page_paths:
            if not page_path:
                logger.warning(f"Encountered None page_path for doc {doc_name} in product {product_name}")
                continue

            page_content = load_markdown_page(page_path)
            if page_content is not None:
                full_md_content_parts.append(page_content + "\n")
            else:
                logger.warning(f"Could not load markdown page: {page_path} for product {product_name}")

    full_md_content = "".join(full_md_content_parts)
    if not full_md_content.strip():
        logger.warning(f"No markdown content aggregated for product {product_name}.")
        return None
    return full_md_content


def check_document_size(product_name: str) -> Tuple[bool, int]:
    """Checks if a product's markdown content exceeds the context limit."""
    product_markdown = _get_full_markdown_for_product(product_name)
    if not product_markdown:
        return False, 0
    content_len = len(product_markdown)
    return content_len > MAX_MARKDOWN_CONTEXT_CHARS, content_len


def _truncate_markdown(markdown_content: str, product_name: str) -> str:
    """Truncates markdown if it exceeds the character limit."""
    if len(markdown_content) > MAX_MARKDOWN_CONTEXT_CHARS:
        truncated = markdown_content[:MAX_MARKDOWN_CONTEXT_CHARS] + "\n...[DOCUMENT TRUNCATED]"
        logger.warning(
            f"Markdown for {product_name} truncated. Original size: {len(markdown_content)} chars, "
            f"Limit: {MAX_MARKDOWN_CONTEXT_CHARS} chars."
        )
        return truncated
    return markdown_content


def extract_answers_for_product(
    product_name: str,
    questions_config: Dict[str, Any],
    extract_by_category: bool = True,
    model_choice: str = AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT
) -> Optional[Dict[str, Any]]:
    """
    Extracts answers for all configured questions for a single product.
    """
    logger.info(f"Starting answer extraction for product: {product_name} (by_category={extract_by_category}) using model {model_choice}.")
    product_markdown_full = _get_full_markdown_for_product(product_name)
    if not product_markdown_full:
        logger.error(f"No markdown content for product {product_name}. Cannot extract.")
        return None

    product_markdown_truncated = _truncate_markdown(product_markdown_full, product_name)
    all_answers_for_product: List[Dict[str, Any]] = []
    categories = questions_config.get("categories", [])
    all_questions = questions_config.get("questions", [])

    if not categories or not all_questions:
        logger.warning(f"No categories or questions configured for {product_name}.")
        return {"product_name": product_name, "answers": []}

    if extract_by_category:
        for category in categories:
            logger.info(f"Extracting answers for category: '{category}' in product: {product_name}")
            questions_for_category = [
                q for q in all_questions if category in q.get("applies_to_categories", [])
            ]
            if not questions_for_category:
                logger.info(f"No questions for category '{category}'. Skipping.")
                continue

            prompt_questions_str = "\n".join(
                [f"{i+1}. (ID: {q['id']}) {q['text']}" for i, q in enumerate(questions_for_category)]
            )
            system_prompt = EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(category=category)
            user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
                product_name=product_name,
                category=category,
                truncated_markdown=product_markdown_truncated,
                prompt_questions_str=prompt_questions_str
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            response_str = _call_llm_with_retry(messages, model_choice)

            if not response_str:
                logger.error(f"LLM call failed for category '{category}', product '{product_name}'.")
                for q_dict in questions_for_category:
                    all_answers_for_product.append({
                        "question_id": q_dict['id'], "question_text": q_dict['text'],
                        "answer": "Error: LLM extraction failed", "category": category, "status": "error_llm"
                    })
                continue

            try:
                cleaned_response_str = re.sub(r"```json\n(.*?)\n```", r"\1", response_str, flags=re.DOTALL)
                cleaned_response_str = re.sub(r"```(.*?)\n```", r"\1", cleaned_response_str, flags=re.DOTALL)
                answers_json = json.loads(cleaned_response_str.strip())

                if not isinstance(answers_json, dict):
                    raise ValueError("LLM response is not a JSON object.")

                for q_dict in questions_for_category:
                    q_id = q_dict['id']
                    answer_text = answers_json.get(q_id, "Answer not found by LLM.")
                    all_answers_for_product.append({
                        "question_id": q_id, "question_text": q_dict['text'],
                        "answer": str(answer_text), "category": category, "status": "raw"
                    })
                logger.info(f"Extracted {len(answers_json)} answers for category '{category}', product '{product_name}'.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM response for '{category}', '{product_name}': {e}. Response: {response_str[:500]}")
                for q_dict in questions_for_category:
                    all_answers_for_product.append({
                        "question_id": q_dict['id'], "question_text": q_dict['text'],
                        "answer": "Error: Parsing LLM response failed", "category": category, "status": "error_parsing"
                    })
    else:
        logger.warning("Non-category-based extraction is not currently implemented in detail.")
        for q_dict in all_questions:
            all_answers_for_product.append({
                "question_id": q_dict['id'], "question_text": q_dict['text'],
                "answer": "Error: Batch extraction not fully implemented",
                "category": ", ".join(q_dict.get("applies_to_categories", ["N/A"])),
                "status": "error_not_implemented"
            })


    return {"product_name": product_name, "answers": all_answers_for_product}


def self_correct_answers(
    product_name: str,
    extracted_answers: List[Dict[str, Any]],
    model_choice: str = AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT
) -> Optional[List[Dict[str, Any]]]:
    """
    Performs self-correction review of extracted answers using an LLM.
    """
    logger.info(f"Starting self-correction review for product: {product_name} using model {model_choice}.")
    product_markdown_full = _get_full_markdown_for_product(product_name)
    if not product_markdown_full:
        logger.error(f"No markdown for {product_name}. Cannot perform self-correction.")
        return None

    product_markdown_truncated = _truncate_markdown(product_markdown_full, product_name)
    answers_str_parts: List[str] = []
    for i, ans_item in enumerate(extracted_answers):
        answers_str_parts.append(
            f"{i+1}. Category: {ans_item.get('category', 'N/A')}\n"
            f"   Question (ID: {ans_item['question_id']}): {ans_item['question_text']}\n"
            f"   Extracted Answer: {ans_item['answer']}\n"
        )
    answers_str = "\n".join(answers_str_parts)

    user_prompt = SELF_CORRECTION_USER_PROMPT_TEMPLATE.format(
        product_name=product_name,
        truncated_markdown=product_markdown_truncated,
        answers_str=answers_str
    )
    messages = [{"role": "system", "content": SELF_CORRECTION_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]

    response_str = _call_llm_with_retry(messages, model_choice)
    if not response_str:
        logger.error(f"Self-correction LLM call failed for product '{product_name}'.")
        return None

    try:
        cleaned_response_str = re.sub(r"```json\n(.*?)\n```", r"\1", response_str, flags=re.DOTALL)
        cleaned_response_str = re.sub(r"```(.*?)\n```", r"\1", cleaned_response_str, flags=re.DOTALL)
        review_data = json.loads(cleaned_response_str.strip())

        if "corrections" not in review_data or not isinstance(review_data["corrections"], list):
            logger.error(f"Malformed corrections response for '{product_name}'. Response: {response_str[:500]}")
            raise ValueError("LLM response for corrections is malformed.")

        corrections_list = review_data["corrections"]
        logger.info(f"Self-correction review for '{product_name}' suggested {len(corrections_list)} corrections.")

        valid_corrections: List[Dict[str, Any]] = []
        for item in corrections_list:
            if isinstance(item, dict) and all(k in item for k in ["question_id", "original_answer", "suggested_correction", "reason"]):
                valid_corrections.append(item)
            else:
                logger.warning(f"Malformed correction item ignored for '{product_name}': {item}")
        return valid_corrections
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing self-correction response for '{product_name}': {e}. Response: {response_str[:500]}")
        return None


def apply_corrections(
    original_answers: List[Dict[str, Any]],
    corrections_list: List[Dict[str, Any]],
    product_name: str
) -> List[Dict[str, Any]]:
    """
    Applies suggested corrections to the original list of answers.
    """
    logger.info(f"Applying {len(corrections_list)} corrections for product: {product_name}.")
    updated_answers_map = {ans['question_id']: ans.copy() for ans in original_answers}

    for correction in corrections_list:
        q_id = correction.get("question_id")
        suggested_answer = correction.get("suggested_correction")

        if q_id in updated_answers_map:
            logger.info(
                f"Applying correction for Q_ID {q_id} in {product_name}: "
                f"'{updated_answers_map[q_id]['answer']}' -> '{suggested_answer}'"
            )
            updated_answers_map[q_id]['answer'] = str(suggested_answer)
            updated_answers_map[q_id]['status'] = "corrected"
            updated_answers_map[q_id]['correction_reason'] = correction.get("reason", "No reason provided.")
        else:
            logger.warning(f"Correction for non-existent Q_ID {q_id} in {product_name}. Ignoring.")

    final_answers_list = list(updated_answers_map.values())
    logger.info(f"Finished applying corrections for {product_name}.")
    return final_answers_list
