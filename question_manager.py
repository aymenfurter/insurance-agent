"""
Manages the generation and configuration of questions and categories
for insurance product comparison using LLMs.
"""
import json
from typing import Any, Dict, List, Optional

from azure_clients import call_llm
from config import (AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT, logger)
from prompts import (CATEGORIES_SYSTEM_PROMPT, CATEGORIES_USER_PROMPT_TEMPLATE,
                     QUESTIONS_SYSTEM_PROMPT, QUESTIONS_USER_PROMPT_TEMPLATE)

MAX_PROMPT_TOKEN_APPROXIMATION = 250000

def _prepare_text_corpus(all_docs_content_map: Dict[str, List[str]]) -> str:
    """Combines all document content into a single string, with truncation if necessary."""
    full_text_corpus = []
    for product_name, doc_contents in all_docs_content_map.items():
        full_text_corpus.append(f"\n\n--- Content from Product: {product_name} ---\n")
        for content in doc_contents:
            full_text_corpus.append(content + "\n")

    corpus_str = "".join(full_text_corpus)
    if len(corpus_str) > MAX_PROMPT_TOKEN_APPROXIMATION:
        corpus_str = corpus_str[:MAX_PROMPT_TOKEN_APPROXIMATION] + "\n... [CONTENT TRUNCATED]"
        logger.warning(f"Full text corpus truncated to ~{MAX_PROMPT_TOKEN_APPROXIMATION} chars for LLM.")
    return corpus_str


def _parse_llm_json_response(response_str: Optional[str], key_name: str) -> Optional[Any]:
    """Parses JSON response from LLM, expecting a specific key."""
    if not response_str:
        return None
    try:
        data = json.loads(response_str)
        return data.get(key_name)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error from LLM response: {e}. Response: {response_str[:500]}...")
        return None


def suggest_categories_and_questions(
    all_docs_content_map: Dict[str, List[str]],
    sample_categories_str: str = "",
    sample_questions_str: str = "",
    model_deployment_name: str = AZURE_OPENAI_REASONING_MODEL_DEPLOYMENT
) -> Optional[Dict[str, Any]]:
    """
    Suggests insurance categories and questions based on document content.
    """
    if not all_docs_content_map:
        logger.warning("No document content provided to suggest categories/questions.")
        return {"categories": [], "questions": []}

    logger.info(f"Suggesting C&Q using model: {model_deployment_name}")
    full_text_corpus = _prepare_text_corpus(all_docs_content_map)

    sample_categories_text = ""
    if sample_categories_str.strip():
        cats = [cat.strip() for cat in sample_categories_str.strip().split('\n') if cat.strip()]
        if cats:
            sample_categories_text = "Consider including these sample categories if relevant:\n" + \
                                     "\n".join([f"- {cat}" for cat in cats])

    categories_user_prompt = CATEGORIES_USER_PROMPT_TEMPLATE.format(
        sample_categories_text=sample_categories_text,
        full_text_corpus=full_text_corpus
    )
    categories_messages = [
        {"role": "system", "content": CATEGORIES_SYSTEM_PROMPT},
        {"role": "user", "content": categories_user_prompt}
    ]

    categories_response_str = call_llm(categories_messages, model_deployment_name, json_mode=True)
    suggested_categories_raw = _parse_llm_json_response(categories_response_str, "categories")

    if not isinstance(suggested_categories_raw, list):
        logger.error(f"LLM returned malformed categories list or parsing failed. Raw: {suggested_categories_raw}")
        return None

    suggested_categories = sorted(list(set(
        c.strip().lower().replace('/', ' and ').title()
        for c in suggested_categories_raw
        if isinstance(c, str) and c.strip() and len(c.strip()) < 100
    )))
    logger.info(f"Suggested categories: {suggested_categories}")

    if not suggested_categories:
        logger.warning("No categories were suggested by the LLM.")
        return {"categories": [], "questions": []}

    sample_questions_text = ""
    if sample_questions_str.strip():
        qs = [q.strip() for q in sample_questions_str.strip().split('\n') if q.strip()]
        if qs:
            sample_questions_text = "Consider including these types of questions if relevant:\n" + \
                                    "\n".join([f"- {q}" for q in qs])

    first_category_example = suggested_categories[0] if suggested_categories else "ExampleCategory"

    questions_user_prompt = QUESTIONS_USER_PROMPT_TEMPLATE.format(
        categories_list="\n".join([f"- {cat}" for cat in suggested_categories]),
        sample_questions_text=sample_questions_text,
        full_text_corpus=full_text_corpus,
        first_category_example=f'"{first_category_example}"'
    )
    questions_messages = [
        {"role": "system", "content": QUESTIONS_SYSTEM_PROMPT},
        {"role": "user", "content": questions_user_prompt}
    ]

    questions_response_str = call_llm(questions_messages, model_deployment_name, json_mode=True, max_tokens=32000)
    questions_list_raw = _parse_llm_json_response(questions_response_str, "questions")

    if not isinstance(questions_list_raw, list):
        logger.error(f"LLM returned malformed questions list or parsing failed. Raw: {questions_list_raw}")
        return {"categories": suggested_categories, "questions": []}

    final_questions: List[Dict[str, Any]] = []
    category_map = {cat.lower(): cat for cat in suggested_categories}

    for i, q_data in enumerate(questions_list_raw):
        if not isinstance(q_data, dict) or \
           not isinstance(q_data.get("text"), str) or \
           not isinstance(q_data.get("applies_to_categories"), list):
            logger.warning(f"Skipping malformed question data: {q_data}")
            continue

        valid_q_categories = []
        for cat_name in q_data["applies_to_categories"]:
            if not isinstance(cat_name, str):
                continue

            normalized_cat_name = cat_name.strip().lower().replace('/', ' and ')
            if normalized_cat_name in category_map:
                 valid_q_categories.append(category_map[normalized_cat_name])
            elif cat_name in suggested_categories : # Exact match
                 valid_q_categories.append(cat_name)
            else:
                logger.debug(f"Question category '{cat_name}' not in suggested list, attempting to add if new.")

        if valid_q_categories:
            final_questions.append({
                "id": f"q{i+1:03d}", # Padded ID
                "text": q_data["text"].strip(),
                "applies_to_categories": sorted(list(set(valid_q_categories)))
            })
        else:
            logger.warning(f"Skipping question '{q_data['text']}' due to no valid/mappable categories.")

    logger.info(f"Generated {len(final_questions)} questions.")
    return {"categories": sorted(list(set(suggested_categories))), "questions": final_questions}
