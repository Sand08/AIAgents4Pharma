#!/usr/bin/env python3

"""
Tool for querying the metadata table of the last displayed papers.

This tool loads the most recently displayed papers into a pandas DataFrame and uses an
LLM-driven pandas agent to answer metadata-level questions (e.g., filter by author, list titles).
It is intended for metadata exploration only, and does not perform content-based retrieval
or summarization. For PDF-level question answering, use the 'question_and_answer_agent'.
"""

import logging
from typing import Annotated
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


class QueryDataframeInput(BaseModel):
    """Input schema for the query dataframe tool."""

    question: str = Field(
        description="The metadata query to ask over the papers table. "
        "Can include sorting requests like 'Show top 5 papers by H-index' or "
        "'List papers sorted by citation count'."
    )
    state: Annotated[dict, InjectedState]


@tool("query_dataframe", args_schema=QueryDataframeInput, parse_docstring=True)
def query_dataframe(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Perform a tabular query on the most recently displayed papers.

    This function loads the last displayed papers into a pandas DataFrame and uses a
    pandas DataFrame agent to answer metadata-level questions (e.g., "Which papers have
    'Transformer' in the title?", "List authors of paper X", "Show top 5 papers by H-index").
    It does not perform PDF content analysis or summarization; for content-level question
    answering, use the 'question_and_answer_agent'.
    
    The agent can handle sorting requests using pandas operations like:
    - "Show me the top 5 papers by H-index"
    - "List all papers sorted by citation count in descending order"
    - "Which paper has the highest H-index?"

    Args:
        question (str): The metadata query to ask over the papers table.
        state (dict): The agent's state containing 'last_displayed_papers'
            key referencing the metadata table in state.

    Returns:
        str: The LLM's response to the metadata query.

    Raises:
        NoPapersFoundError: If no papers have been displayed yet.
    """
    logger.info("Querying last displayed papers with question: %s", question)
    llm_model = state.get("llm_model")
    context_val = state.get("last_displayed_papers")

    if not context_val:
        logger.info("No papers displayed so far, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )

    # Support both key reference (str) and direct mapping
    if isinstance(context_val, dict):
        dic_papers = context_val
    else:
        dic_papers = state.get(context_val)

    # Convert to DataFrame
    df_papers = pd.DataFrame.from_dict(dic_papers, orient="index")

    # Pre-process numeric columns for better sorting capability
    # Convert 'Max H-Index' and 'Citation Count' to numeric, handling 'N/A' values
    if 'Max H-Index' in df_papers.columns:
        df_papers['Max H-Index'] = pd.to_numeric(
            df_papers['Max H-Index'].replace('N/A', None),
            errors='coerce'
        )

    if 'Citation Count' in df_papers.columns:
        df_papers['Citation Count'] = pd.to_numeric(
            df_papers['Citation Count'].replace('N/A', None),
            errors='coerce'
        )

    if 'Year' in df_papers.columns:
        df_papers['Year'] = pd.to_numeric(
            df_papers['Year'].replace('N/A', None),
            errors='coerce'
        )

    # Log the actual papers being queried
    logger.info("Querying over %d papers that are currently displayed", len(df_papers))

    # Create pandas agent with enhanced prompt for sorting
    df_agent = create_pandas_dataframe_agent(
        llm_model,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        df=df_papers,
        max_iterations=5,
        include_df_in_prompt=True,
        number_of_head_rows=min(df_papers.shape[0], 10),  # Show up to 10 rows
        verbose=True,
        prefix=(
            f"You are working with a pandas dataframe containing {len(df_papers)} "
            "academic papers metadata. The dataframe has the following columns: "
            + ", ".join(df_papers.columns.tolist()) + ". "
            "IMPORTANT: This dataframe contains ONLY the papers that are currently "
            "displayed to the user. If the user asks about 'these papers' or 'the papers', "
            "they mean ALL papers in this dataframe. Always use the ACTUAL values from "
            "the dataframe. Never make up or hallucinate values. For numeric columns like "
            "'Max H-Index' and 'Citation Count', use the exact values shown in the "
            "dataframe. Numeric columns may contain NaN values which should be handled "
            "appropriately. Use pandas methods like sort_values(), nlargest(), nsmallest() "
            "for sorting tasks. When sorting, use na_position='last' to put NaN values at "
            "the end. When reporting results, ALWAYS use the actual values from the "
            "dataframe, not made-up numbers."
        )
    )

    # Enhance question with sorting hints if applicable
    enhanced_question = question
    sorting_keywords = ['top', 'highest', 'lowest', 'sort', 'rank', 'by h-index',
                        'by h index', 'by citation']
    if any(keyword in question.lower() for keyword in sorting_keywords):
        enhanced_question += (
            " IMPORTANT: Use the EXACT 'Max H-Index' or 'Citation Count' values from "
            "the dataframe. Do not make up any numbers. For sorting, use pandas methods "
            "like sort_values() or nlargest(). When listing results, show the actual "
            "H-Index values from the 'Max H-Index' column."
        )

    llm_result = df_agent.invoke(enhanced_question)
    return llm_result["output"]
