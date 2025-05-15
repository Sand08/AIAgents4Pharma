#!/usr/bin/env python3

"""
Tool for querying the metadata table of the last displayed papers.

This tool loads the most recently displayed papers into a pandas DataFrame and uses an
LLM-driven pandas agent to answer metadata-level questions. It's designed for exploring
paper metadata (authors, titles, publication years, citation counts, h-indices) without
accessing the paper content itself.
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
        description="The metadata-specific question to ask about the papers table. "
        "Examples: 'Which papers have the highest citation count?', 'Show papers by author X', "
        "'List papers published after 2020', 'Which paper has the highest H-Index?'"
    )
    state: Annotated[dict, InjectedState]

@tool("query_dataframe", args_schema=QueryDataframeInput, parse_docstring=True)
def query_dataframe(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Answer questions about the metadata of displayed papers (not their content).

    Use this tool to ask specific questions about paper metadata such as:
    - "Which papers were published after 2020?"
    - "List papers authored by Smith"
    - "What's the average citation count?"
    - "Show the 3 papers with the highest H-Index"
    - "How many papers mention 'neural networks' in the title?"
    
    This tool only analyzes paper metadata (titles, authors, years, citation counts, h-indices)
    and cannot answer questions about the paper content or perform summarization.
    For content-level questions, use the 'question_and_answer_agent' instead.

    Args:
        question (str): The specific metadata question to answer about the papers.
        state (dict): The agent's state containing 'last_displayed_papers'
            key referencing the metadata table in state.

    Returns:
        str: The answer to the metadata query.

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
    return llm_result["output"]