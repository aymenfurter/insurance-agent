"""
Handles interaction with Azure AI Agent Service for analysis and visualization.
"""
import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (Agent, AsyncAgentEventHandler,
                                    CodeInterpreterTool, MessageDeltaChunk)
from azure.identity import DefaultAzureCredential

import prompts
from config import logger

ANALYSIS_TOOLS = [{"type": "code_interpreter"}]
ANALYSIS_TEMPLATES = {
    "coverage_heatmap": {
        "name": "Coverage Heat-Map",
        "description": "Colour-coded view of whether each benefit category is covered, partially covered, or not covered in each insurance product.",
        "prompt": prompts.ANALYSIS_PROMPT_COVERAGE_HEATMAP
    },
    "coverage_scorecard": {
        "name": "Coverage Scorecard",
        "description": "Radar (spider) chart showing the share of categories that are fully, partially, or not covered in each insurance product.",
        "prompt": prompts.ANALYSIS_PROMPT_COVERAGE_SCORECARD
    },
    "sublimit_comparison": {
        "name": "Sub-Limit Comparison",
        "description": "Horizontal bar chart comparing CHF limits across insurance products where monetary caps are specified.",
        "prompt": prompts.ANALYSIS_PROMPT_SUBLIMIT_COMPARISON
    },
    "deductible_profile": {
        "name": "Deductible & Co-Pay Profile",
        "description": "Stacked column chart summarising cost-sharing intensity by insurance product.",
        "prompt": prompts.ANALYSIS_PROMPT_DEDUCTIBLE_PROFILE
    },
    "provider_prescription_matrix": {
        "name": "Provider & Prescription Rules Matrix",
        "description": "Table showing, for each benefit, whether recognised providers and/or a medical prescription are required.",
        "prompt": prompts.ANALYSIS_PROMPT_PROVIDER_PRESCRIPTION_MATRIX
    },
    "illness_vs_accident_chart": {
        "name": "Illness vs. Accident Coverage",
        "description": "Pie charts comparing benefits of each insurance product.",
        "prompt": prompts.ANALYSIS_PROMPT_ILLNESS_VS_ACCIDENT_CHART
    },
    "age_dependency_analysis": {
        "name": "Age & Dependent Restrictions",
        "description": "Highlights benefits that change once the insured reaches a certain age.",
        "prompt": prompts.ANALYSIS_PROMPT_AGE_DEPENDENCY_ANALYSIS
    },
    "coverage_gap_dashboard": {
        "name": "Coverage Gap Prioritisation",
        "description": "Ranks the biggest monetary gaps relative to the highest limit available across insurance products.",
        "prompt": prompts.ANALYSIS_PROMPT_COVERAGE_GAP_DASHBOARD
    },
    "cross_sell_opportunity_matrix": {
        "name": "Cross-Sell Opportunity Matrix",
        "description": "Flags benefits with No or very low limits and suggests complementary add-ons.",
        "prompt": prompts.ANALYSIS_PROMPT_CROSS_SELL_OPPORTUNITY_MATRIX
    }
}

class AnalysisStreamHandler(AsyncAgentEventHandler):
    """Handle Analysis LLM streaming events."""
    def __init__(self, client: AIProjectClient):
        self.results = {
            "plots": [],
            "tables": [],
            "explanation": "",
            "error": None
        }
        self.client = client
        self.thread_images = []
        logger.info("Initialized AnalysisStreamHandler")
        super().__init__()

    async def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        if delta.text:
            logger.debug(f"Received message delta: {delta.text[:100]}...")

    async def on_error(self, error: str) -> None:
        logger.error(f"Stream handler error: {error}")
        self.results["error"] = error

    async def on_thread_message(self, message):
        """Handle message outputs including files."""
        logger.info("Processing thread message...")

        if hasattr(message, 'image_contents'):
            for image_content in message.image_contents:
                try:
                    logger.info(f"Processing image: {image_content.image_file.file_id}")
                    file_name = f"{image_content.image_file.file_id}_image.png"
                    await self.client.agents.save_file(
                        file_id=image_content.image_file.file_id,
                        file_name=file_name
                    )

                    with open(file_name, 'rb') as f:
                        image_data = f.read()

                    self.results["plots"].append({
                        "type": "plot",
                        "image_base64": base64.b64encode(image_data).decode(),
                        "title": f"Analysis Plot {len(self.results['plots']) + 1}"
                    })
                    logger.info("Successfully added plot to results")

                    os.remove(file_name)
                except Exception as e:
                    logger.error(f"Error processing image file: {e}")

        if hasattr(message, 'content'):
            for content in message.content:
                if hasattr(content, 'text'):
                    if "<table>" in content.text.value:
                        logger.info("Processing HTML table content")
                        self.results["tables"].append({
                            "type": "table",
                            "data_html": content.text.value,
                            "title": f"Analysis Table {len(self.results['tables']) + 1}"
                        })
                    else:
                        logger.debug(f"Adding explanation text: {content.text.value[:100]}...")
                        self.results["explanation"] += content.text.value

async def initialize_agent_client() -> Optional[AIProjectClient]:
    """Initialize Azure AI Agent Client."""
    try:
        connection_string = os.getenv("PROJECT_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("PROJECT_CONNECTION_STRING environment variable not set")

        client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=connection_string
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize agent client: {e}")
        return None

def format_data_as_markdown(data: List[Dict[str, Any]]) -> str:
    """Convert product comparison data into readable markdown format."""
    markdown = []

    for product in data:
        product_name = product.get('product_name', 'Unknown Product')
        markdown.append(f"\n## Product: {product_name}")

        answers_by_category = {}
        for answer in product.get('answers', []):
            category = answer.get('category', 'Uncategorized')
            if category not in answers_by_category:
                answers_by_category[category] = []
            answers_by_category[category].append(answer)

        for category, answers in sorted(answers_by_category.items()):
            markdown.append(f"\n### {category}")
            for ans in answers:
                q_text = ans.get('question_text', 'Unknown Question')
                answer = ans.get('answer', 'Not specified')
                markdown.append(f"- **{q_text}**: {answer}")

    return "\n".join(markdown)

async def run_analysis(data: List[Dict[str, Any]], analysis_request: str) -> Dict[str, Any]:
    """Run analysis using Azure AI Agent Service."""
    logger.info(f"Starting analysis with request: {analysis_request}")
    logger.info(f"Input data summary: {len(data)} products to analyze")
    logger.info(f"Data sample: {str(data[0])[:200]}..." if data else "No data")

    client = await initialize_agent_client()
    if not client:
        return {"error": "Failed to initialize AI agent client"}

    try:
        async with client:
            code_interpreter = CodeInterpreterTool()

            agent = await client.agents.create_agent(
                model=os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
                name="InsuranceAnalysisAgent",
                instructions=prompts.AGENT_INSTRUCTIONS_GENERAL,
                tools=code_interpreter.definitions,
                tool_resources=code_interpreter.resources,
            )
            logger.info(f"Created agent: {agent.id}")

            thread = await client.agents.create_thread()
            logger.info(f"Created thread: {thread.id}")

            message_content = prompts.AGENT_ANALYSIS_USER_MESSAGE_TEMPLATE.format(
                analysis_request=analysis_request,
                formatted_data=format_data_as_markdown(data)
            )

            await client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=message_content
            )

            handler = AnalysisStreamHandler(client)
            stream = await client.agents.create_stream(
                thread_id=thread.id,
                agent_id=agent.id,
                event_handler=handler
            )

            logger.info("Waiting for stream to finish…")
            async with stream as s:
                await s.until_done()

            logger.info(
                f"Stream finished – gathered {len(handler.results['plots'])} plots "
                f"and {len(handler.results['tables'])} tables."
            )
            return handler.results

    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}", exc_info=True)
        return {
            "plots": [],
            "tables": [],
            "explanation": "",
            "error": f"Agent analysis failed: {str(e)}"
        }