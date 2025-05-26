#!/usr/bin/env python3

"""
Tool for querying the metadata table of the last displayed papers.

This tool loads the most recently displayed papers into a pandas DataFrame and
uses an LLM-driven pandas agent to answer metadata-level questions (e.g.,
filter by author, list titles). It is intended for metadata exploration only,
and does not perform content-based retrieval or summarization. For PDF-level
question answering, use the 'question_and_answer_agent'.
"""

import logging
from typing import Annotated, Optional
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from .utils.query_helper import QueryHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""


class QueryDataframeInput(BaseModel):
    """Input schema for the query dataframe tool."""

    question: str = Field(
        description="The metadata query to ask over the papers table."
    )
    sort_by: Optional[str] = Field(
        default=None,
        description=(
            "Column to sort by before querying. Options: 'Max H-Index', "
            "'Citation Count', 'Year', 'Title', 'Authors'. "
            "If not specified, original order is maintained."
        )
    )
    ascending: bool = Field(
        default=False,
        description=(
            "Sort order when sort_by is specified. "
            "False for descending (highest first), True for ascending."
        )
    )
    limit: Optional[int] = Field(
        default=None,
        description=(
            "Number of top results to consider after sorting. "
            "If not specified, all papers are considered."
        ),
        ge=1,
        le=10
    )
    state: Annotated[dict, InjectedState]


@tool("query_dataframe", args_schema=QueryDataframeInput, parse_docstring=True)
def query_dataframe(
    question: str,
    state: Annotated[dict, InjectedState],
    sort_by: Optional[str] = None,
    ascending: bool = False,
    limit: Optional[int] = None,
) -> str:
    """
    Perform a tabular query on the most recently displayed papers.

    This function loads the last displayed papers into a pandas DataFrame and
    uses a pandas DataFrame agent to answer metadata-level questions (e.g.,
    "Which papers have 'Transformer' in the title?", "List authors of paper X").
    It does not perform PDF content analysis or summarization; for content-level
    question answering, use the 'question_and_answer_agent'.

    The tool can optionally sort the papers by bibliographic metrics before
    performing the query, which is useful for questions like "What are the
    abstracts of the top 5 papers by citation count?"

    Args:
        question (str): The metadata query to ask over the papers table.
        state (dict): The agent's state containing 'last_displayed_papers'
            key referencing the metadata table in state.
        sort_by (str, optional): Column to sort by before querying
        ascending (bool): Sort order - False for descending (default)
        limit (int, optional): Number of top results to consider after sorting

    Returns:
        str: The LLM's response to the metadata query.

    Raises:
        NoPapersFoundError: If no papers have been displayed yet.
    """
    logger.info(
        "Querying papers with question: %s, sort_by: %s, limit: %s",
        question, sort_by, limit
    )

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

    # Use helper to prepare DataFrame with sorting if needed
    helper = QueryHelper(dic_papers)
    df_papers = helper.prepare_dataframe(
        sort_by=sort_by,
        ascending=ascending,
        limit=limit
    )

    # Log the actual papers being queried
    logger.info(
        "Querying over %d papers%s",
        len(df_papers),
        f" (sorted by {sort_by})" if sort_by else ""
    )

    # Create agent with prepared DataFrame
    df_agent = create_pandas_dataframe_agent(
        llm_model,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        df=df_papers,
        max_iterations=5,
        include_df_in_prompt=True,
        number_of_head_rows=min(df_papers.shape[0], 20),  # Limit for performance
        verbose=True,
        prefix=(
            "You are working with a pandas dataframe containing research "
            "papers metadata. The dataframe is already loaded as 'df'. "
            f"It contains {len(df_papers)} papers"
            f"{f' sorted by {sort_by}' if sort_by else ''}. "
            "Column names include: Title, Authors, Year, Citation Count, "
            "Max H-Index, Abstract, URL, etc. "
            "Always refer to the dataframe that's already loaded, "
            "don't try to search for new papers.\n\n"
        )
    )

    llm_result = df_agent.invoke(question, stream_mode=None)
    return llm_result["output"]
