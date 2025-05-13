"""
Central repository for all LLM prompts used in the application.
"""

# --- Question Manager Prompts ---

CATEGORIES_SYSTEM_PROMPT = "You are an AI assistant specialized in analyzing insurance products. Your task is to identify key coverage categories based on the provided insurance terms documents."
CATEGORIES_USER_PROMPT_TEMPLATE = """
Based on the following combined text from multiple insurance product documents, identify the main coverage categories typically found.
Focus on distinct insurable perils or sections of coverage.

Provide the output ONLY as a valid JSON object with a single key "categories" which is a list of unique category names (strings).
Example Format: {{"categories": ["Fire Damage", "Water Damage", "Theft/Burglary"]}}

Examples: {sample_categories_text}

Combined Insurance Text:
{full_text_corpus}
"""

QUESTIONS_SYSTEM_PROMPT = "You are an AI assistant creating questions for comparing insurance products."
QUESTIONS_USER_PROMPT_TEMPLATE = """
Generate comparison questions for the following categories:

{categories_list}

Requirements:
1. Generate questions that can apply to multiple categories where appropriate.
2. For each category's core coverage, use the standard question: "Is this category covered under the insurance?"
3. Add questions about:
   - Coverage limits
   - Exclusions and conditions
   - Amounts and deductibles
   - Special terms or restrictions
4. Questions should be generic enough to work across different insurance products.
5. The category names in applies_to_categories must EXACTLY match the provided categories.
6. Generate a total of 20-30 questions across all categories.

{sample_questions_text}

Provide ONLY a valid JSON object with key "questions" containing question objects.
Each question object must have:
- "text": The question text
- "applies_to_categories": List of EXACT category names from above list

Example format:
{{
    "questions": [
        {{"text": "Is this category covered under the insurance?", "applies_to_categories": [{first_category_example}]}},
        {{"text": "What is the maximum coverage amount?", "applies_to_categories": [{first_category_example}]}}
    ]
}}

Context from Insurance Documents:
{full_text_corpus}
"""


# --- Data Extractor Prompts ---

EXTRACTION_SYSTEM_PROMPT_TEMPLATE = "You are an AI assistant extracting information from insurance product terms. Focus specifically on the category: '{category}'. Answer precisely based on the provided document excerpts."
EXTRACTION_USER_PROMPT_TEMPLATE = """
Based *only* on the provided insurance document text for product '{product_name}', answer the following questions related to the '{category}' category.
If information for a question is not found, state 'Not Found' or 'Not Specified'.

Document Text:
---
{truncated_markdown}
---

Questions for category '{category}':
{prompt_questions_str}

Provide your answers ONLY as a valid JSON object. The JSON object should map each question ID (e.g., "q1", "q2") to its answer (string).
Example format: {{ "q1": "Yes", "q5": "5000 EUR", "q7": "" }}
Important: Answer as concisely as possible. Respond in keywords / note form. If a question is not applicable, return empty string.

Ensure all question IDs listed above are present as keys in your JSON response.
"""

SELF_CORRECTION_SYSTEM_PROMPT = "You are an expert AI assistant reviewing extracted information from insurance terms. Your goal is to identify inaccuracies or incomplete answers by cross-referencing with the original document text."
SELF_CORRECTION_USER_PROMPT_TEMPLATE = """
Please review the following extracted answers for the insurance product '{product_name}' against the provided document text.
For each extracted answer, verify its accuracy. If you find any mistakes, incorrect interpretations, or significantly incomplete answers, please list them.

Document Text for '{product_name}':
---
{truncated_markdown}
---

Extracted Answers to Review:
---
{answers_str}
---

Provide your review ONLY as a valid JSON object with a single key "corrections".
The "corrections" key should be a list of objects. Each object should have:
- "question_id": The ID of the question whose answer needs correction.
- "original_answer": The original extracted answer.
- "suggested_correction": Your corrected or more complete answer (string).
- "reason": A brief explanation citing the part of the document that supports your correction (e.g., "Document section X states Y...").

If an answer is correct and complete, do not include it in the "corrections" list.
If all answers are correct, return an empty list for "corrections", i.e., {{"corrections": []}}.
Example: {{"corrections": [{{"question_id": "q1", "original_answer": "Yes", "suggested_correction": "No, only for damages above 100 EUR.", "reason": "Section 3.2 specifies a deductible of 100 EUR for this coverage."}}]}}
"""

# --- Agent Service Prompts ---

AGENT_INSTRUCTIONS_GENERAL = """You are an insurance product comparison analysis expert. Analyze insurance data and create clear, informative visualizations.

When creating visualizations:
1. Create ONE plot per visualization - do not combine multiple visualizations in a single image
2. Always include the names of the insurance products in the plot title or legend
3. Use clear, descriptive titles that explain what is being compared
4. Use appropriate chart types for the data comparison
5. Include proper legends, axis labels, and data formatting
6. Use contrasting colors for different insurers to make them easily distinguishable

For each visualization, also include the source data as a markdown table in the analysis text.

Format your analysis text as proper Markdown with:
- Headers (##, ###) for sections
- Bullet points for listing key findings
- **Bold** important insights
- Tables when appropriate
- Clear paragraph breaks"""

AGENT_ANALYSIS_USER_MESSAGE_TEMPLATE = """Please analyze the provided insurance products data and create visualizations.

Here is the specific analysis request:
{analysis_request}

IMPORTANT VISUALIZATION REQUIREMENTS:
1. Create ONE plot per visualization - do not combine multiple visualizations in one image
2. Always include the names of the insurance products in the plot title or legend
3. Each visualization should have a clear, descriptive title explaining what is being compared
4. Use plt.figure(figsize=(12, 8)) for adequate size
5. Include proper axis labels, legends, and data formatting
6. Use contrasting colors for different insurers to make them easily distinguishable
7. Save each plot with: plt.savefig('output.png', dpi=300, bbox_inches='tight')
8. Always use code interpreter for visualizations, do not use any other tools or code.

FORMAT YOUR ANALYSIS:
1. Start with "# Key Findings" as main heading and 1-2 sentence summary
2. Group detailed findings under clear section headings:
   - Use "# " for main sections
   - Use "## " for subsections
   - Use bullet points for listing insights
   - Use **bold** for important points
   - Use proper markdown tables with |---| headers
   - Include blank lines for readability

Format tables using standard markdown:
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

Data to analyze (in structured format):
{formatted_data}
"""


# --- Prompts from ANALYSIS_TEMPLATES in agent_service.py ---

ANALYSIS_PROMPT_COVERAGE_HEATMAP = """Create a heat-map (or conditional-formatted markdown table):
- Rows: Benefit categories
- Columns: insurance product 1 | insurance product 2 | insurance product 3
- Cell values: Full / Partial / None / Not Specified
Colour key: ✅ Full, 🟡 Partial, ❌ None, ⚪ Not Specified.
Open with one executive takeaway, then list 3–5 notable gaps."""

ANALYSIS_PROMPT_COVERAGE_SCORECARD = """For each insurance product calculate:
- % of categories with Full cover
- % with Partial/Conditional cover
- % with No cover
Plot all insurance products on one radar chart with a legend.
Preface with a 1–2-sentence insight on which insurance product offers the broadest protection."""

ANALYSIS_PROMPT_SUBLIMIT_COMPARISON = """Create a horizontal bar chart:
- Y-axis: Benefit categories that state a CHF limit
- X-axis: Limit amounts
- One bar per insurance product (colour-coded)
Sort categories by the highest limit observed.
Comment briefly on categories with the widest limit spread."""

ANALYSIS_PROMPT_DEDUCTIBLE_PROFILE = """Bucket each category in every insurance product into:
  • No cost-sharing
  • ≤10 % co-pay
  • >10 % and ≤25 % co-pay
  • Flat deductible only
Render one stacked column per insurance product.
Begin with a headline sentence on cost-sharing differences."""

ANALYSIS_PROMPT_PROVIDER_PRESCRIPTION_MATRIX = """Generate a markdown table:
- Rows: Benefit categories
- Columns: “Recognised Provider Required?” | “Prescription Required?”
Use ✓ / ✗ / Not Specified.
Bold any row where both conditions apply."""

ANALYSIS_PROMPT_WAITING_PERIOD_TIMELINE = """For every category–insurance product combo with a numeric waiting period:
- Draw a horizontal bar (Y: category, colour: insurance product)
- X-axis: Months from policy start
Annotate bars with the period (e.g. “3 m”).
Open with a brief note on the longest waits."""

ANALYSIS_PROMPT_ILLNESS_VS_ACCIDENT_CHART = """Create three pie charts (insurance product 1 / 2 / 3):
- Slices: Illness only | Accident only | Both | Not Specified
List two key observations underneath."""

ANALYSIS_PROMPT_GEOGRAPHY_COVERAGE_MATRIX = """Build a markdown table:
- Rows: Benefit categories
- Columns: Domestic | Abroad | Both | Not Specified
Mark applicable cells with ✓.
Add a one-sentence comment on high-risk benefits that are domestic-only."""

ANALYSIS_PROMPT_AGE_DEPENDENCY_ANALYSIS = """Produce a markdown table:
- Columns: Benefit | Restriction Type (Age / Dependent / None) | Details
Bold any benefit that ceases entirely at adulthood."""

ANALYSIS_PROMPT_COVERAGE_GAP_DASHBOARD = """For each category compute:
  Gap % = (Max limit across insurance products − Limit in a given insurance product) ÷ Max limit
Display the ten largest gaps per insurance product in a bar chart and provide a sortable markdown table of all gaps.
Start with a single-sentence highlight of the worst gap overall."""

ANALYSIS_PROMPT_UPGRADE_PROPENSITY_HEATMAP = """Heat-map layout:
- Rows: Benefit categories
- Columns: insurance product 1 → insurance product 2 → insurance product 3
- Cells: Full / Partial / None
Highlight cells where a move to the next insurance product upgrades coverage to Full (🟡).
Bullet the five strongest upgrade messages (e.g. “Dental limit increases 6×”)."""

ANALYSIS_PROMPT_CROSS_SELL_OPPORTUNITY_MATRIX = """Create a markdown table:
- Rows: Benefits with None coverage or limit < CHF 1 000 in any insurance product
- Columns: Suggested Add-On | Rationale
If pricing is available, add an “Approx. extra premium (CHF)” column."""


# --- Analysis Tab Prompts ---

SUMMARIZE_WITH_LLM_SYSTEM_PROMPT = "You are an assistant that writes very concise, plain-text summaries of the key insights. (1-2 sentences, NO markdown)."
