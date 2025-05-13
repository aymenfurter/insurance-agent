"""
Handles the analysis of extracted insurance product data using Code Interpreter.
"""
import json
from config import logger
from typing import List, Dict, Any, Optional
import asyncio
from agent_service import run_analysis

async def execute_analysis_with_agent_async(
    analysis_request: str,
    all_products_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Executes analysis using Azure AI Code Interpreter."""
    try:
        results = await run_analysis(all_products_data, analysis_request)
        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

def execute_analysis_with_agent(
    analysis_request: str,
    all_products_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Synchronous wrapper for agent-based analysis."""
    logger.info("Starting analysis execution")
    logger.info(f"Analysis request: {analysis_request}")
    logger.info(f"Data summary: {len(all_products_data)} products")
    logger.info(f"Sample data structure: {json.dumps(all_products_data[0] if all_products_data else {}, indent=2)[:500]}...")

    try:
        results = asyncio.run(execute_analysis_with_agent_async(analysis_request, all_products_data))

        if not isinstance(results, dict):
            logger.error(f"Unexpected results type: {type(results)}")
            return {
                "plots": [],
                "tables": [],
                "explanation": "",
                "error": f"Unexpected results type: {type(results)}"
            }

        plots = results.get('plots', []) or []
        tables = results.get('tables', []) or []
        explain = results.get('explanation', "") or ""

        logger.info("Analysis results summary:")
        logger.info(f"- Plots: {len(plots)}")
        logger.info(f"- Tables: {len(tables)}")
        logger.info(f"- Explanation length: {len(explain)}")

        if isinstance(results, dict) and results.get('error'):
            logger.error(f"Analysis error: {results['error']}")

        return {
            "plots": plots,
            "tables": tables,
            "explanation": explain,
            "error": results.get("error", "") or ""
        }

    except Exception as e:
        logger.error(f"Analysis execution failed: {e}", exc_info=True)
        return {
            "plots": [],
            "tables": [],
            "explanation": "",
            "error": f"Analysis failed: {str(e)}"
        }