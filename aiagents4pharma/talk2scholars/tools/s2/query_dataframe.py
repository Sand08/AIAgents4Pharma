#!/usr/bin/env python3

"""
Tool for querying the metadata table of the last displayed papers.

This tool loads the most recently displayed papers into a pandas DataFrame and uses an
LLM-driven pandas agent to answer metadata-level questions (e.g., filter by author, 
list titles, H-Index, citation count, year).

IMPORTANT: This tool is intended for metadata exploration ONLY and should be used for questions
about paper attributes like authors, titles, publication years, or citation metrics.
It does NOT perform content-based retrieval or summarization and it does NOT search for new papers.

For questions about paper content (e.g., "Which paper discusses XYZ topic?"), use the
'question_and_answer_agent' instead.

For requests to sort papers by H-Index, Citation Count, or Year, or to limit the number of papers
displayed, use the 'display_dataframe' tool with the appropriate parameters.
"""

import logging
from typing import Annotated
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """Exception raised when no papers are found in the state."""

class QueryDataframeInput(BaseModel):
    """Input schema for the query dataframe tool."""

    question: str = Field(
        description="The metadata query to ask over the EXISTING papers table. "
        "For questions about bibliographic attributes like authors, titles, citation counts, "
        "H-index, or publication years of the papers that have ALREADY been displayed. "
        "Examples: 'Which papers are by Author X?', 'Show papers published after 2020'. "
        "This tool NEVER searches for new papers - it only examines papers already displayed. "
        "For sorting or limiting requests (e.g., 'show top 4 papers by h-index'), "
        "use the display_dataframe tool instead. "
        "For content-based questions, use question_and_answer_agent instead."
    )
    state: Annotated[dict, InjectedState]

@tool("query_dataframe", args_schema=QueryDataframeInput, parse_docstring=True)
def query_dataframe(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Perform a tabular query on the most recently displayed papers' metadata.

    This tool ONLY works with papers that are ALREADY DISPLAYED. It NEVER searches for new papers.
    Use this tool ONLY for metadata-level questions about the currently displayed papers, such as:
    - "Which papers are written by [author name]?"
    - "List papers with more than 100 citations"
    - "Show me papers published after 2020"
    - "Which paper has the highest H-Index?"
    - "Which papers have 'word' in the title?"
    
    This tool will NOT:
    1. Search for new papers not already displayed
    2. Answer content-based questions (use question_and_answer_agent instead)
    3. Sort or limit papers (use display_dataframe instead)

    Args:
        question (str): The metadata query to ask over the ALREADY DISPLAYED papers.
        state (dict): The agent's state containing 'last_displayed_papers'
            key referencing the metadata table in state.

    Returns:
        str: The LLM's response to the metadata query.

    Raises:
        NoPapersFoundError: If no papers have been displayed yet.
    """
    logger.info("Querying last displayed papers with question: %s", question)
    llm_model = state.get("llm_model")
    if not state.get("last_displayed_papers"):
        logger.info("No papers displayed so far, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search needs to be performed first."
        )
    context_key = state.get("last_displayed_papers")
    dic_papers = state.get(context_key)
    df_papers = pd.DataFrame.from_dict(dic_papers, orient="index")

    # Log the number of papers being queried to verify we're using existing papers
    logger.info("Querying existing dataframe with %d papers", len(df_papers))

    df_agent = create_pandas_dataframe_agent(
        llm_model,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        df=df_papers,
        max_iterations=5,
        include_df_in_prompt=True,
        number_of_head_rows=df_papers.shape[0],
        verbose=True,
    )
    llm_result = df_agent.invoke(question, stream_mode=None)

    # Log that we're returning results from existing papers, not from a new search
    logger.info("Returning query results from existing papers, no new search performed")

    return llm_result["output"]
