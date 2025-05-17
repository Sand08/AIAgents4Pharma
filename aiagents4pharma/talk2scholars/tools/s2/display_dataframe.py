#!/usr/bin/env python3
"""
Tool for rendering the most recently displayed papers as a DataFrame artifact for the front-end.
This module defines a tool that retrieves the paper metadata stored under the state key
'last_displayed_papers' and returns it as an artifact (dictionary of papers).

NOTE: This tool does NOT sort papers by default. It only sorts when the sort_by parameter
is provided with a valid value ('Max H-Index', 'Citation Count', or 'Year').

IMPORTANT: 
1. Papers should ONLY be sorted when explicitly requested in queries containing:
   - "sort by [metric]" (e.g., "sort by H-Index", "sort papers by citation count")
   - "top X papers" (e.g., "show top 5 papers")
   - "papers on [topic] by [metric]" (e.g., "papers on embedding by H-Index")
   - "rank papers by [metric]" (e.g., "rank papers by year")

2. DO NOT sort papers when there is no explicit request for sorting, such as:
   - "search for papers on [topic]"
   - "find papers about [topic]"
   - "papers on [topic]" (without mentioning sorting)
"""
import logging
from typing import Annotated, Optional, Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.display_dataframe_helper import DisplayDataHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoPapersFoundError(Exception):
    """
    Exception raised when no research papers are found in the agent's state.
    """

class DisplayDataframeInput(BaseModel):
    """Input schema for the display dataframe tool."""

    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]
    sort_by: Optional[Literal["Max H-Index", "Citation Count", "Year"]] = Field(
        default=None,
        description="Field to sort papers by. Options: 'H-Index', 'Citation Count', 'Year'."
                   "Set for queries like 'sort by H-Index', 'most cited papers', etc."
                   "Default is None (no sorting)."
    )
    ascending: bool = Field(
        default=False,
        description="Sort order: True for ascending (lowest first),"
                   "False for descending (highest first). Default is False (highest values first)."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit the number of results."
                   "For example, limit=4 will show only the top 4 papers."
                   "Set this for queries like 'show top 4 papers', 'limit to 5 results', etc."
    )


@tool("display_dataframe", args_schema=DisplayDataframeInput, parse_docstring=True)
def display_dataframe(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    sort_by: Optional[Literal["Max H-Index", "Citation Count", "Year"]] = None,
    ascending: bool = False,
    limit: Optional[int] = None,
) -> Command:
    """
    Render the last set of retrieved papers as a DataFrame in the front-end.

    This function reads the 'last_displayed_papers' key from state, fetches the
    corresponding metadata dictionary, and returns a Command with a ToolMessage
    containing the artifact (dictionary) for the front-end to render as a DataFrame.

    IMPORTANT: This tool only sorts papers when explicitly asked to via the sort_by parameter.
    By default, papers are displayed in their original order without sorting.
    
    TRIGGER PHRASES: Set sorting parameters for queries with phrases like:
    - "sort by H-Index" or "papers by H-Index" → set sort_by="H-Index"
    - "sort by citation" or "most cited" → set sort_by="Citation Count"
    - "sort by year" or "newest papers" → set sort_by="Year"
    - "top 4" or "limit to 5" → set limit=4 or limit=5

    DO NOT sort papers for queries like:
    - "search for papers on [topic]"
    - "find papers about [topic]"
    - "papers on [topic]" (without explicit sorting mentioned)
    
    DO NOT sort papers unless the query contains such explicit sorting requests.

    Args:
        tool_call_id (InjectedToolCallId): Unique ID of this tool invocation.
        state (dict): The agent's state containing the 'last_displayed_papers' reference.
        sort_by (str, optional): Field to sort papers by. Options: 'H-Index', 'Citation Count', 'Year'.
        ascending (bool): Sort order - True for ascending, False for descending.
        limit (int, optional): Limit the number of results (e.g., limit=4 shows top 4 papers).

    Returns:
        Command: A command whose update contains a ToolMessage with the artifact
                 (papers dict) for DataFrame rendering in the UI.

    Raises:
        NoPapersFoundError: If no entries exist under 'last_displayed_papers' in state.
    """
    logger.info("display_dataframe called with sort_by=%s, ascending=%s, limit=%s",
                sort_by, ascending, limit)
    context_key = state.get("last_displayed_papers")
    logger.info("Retrieved context_key: %s", context_key)

    artifact = state.get(context_key)
    if not artifact:
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search/rec needs to be performed first."
        )

    logger.info("Retrieved artifact with %s papers", len(artifact))

    helper = DisplayDataHelper(artifact, sort_by, ascending, limit)
    result = helper.process_display()

    logger.info("Helper returned result with %s papers", len(result['artifact']))

    update_dict = {
        "messages": [
            ToolMessage(
                content=result["content"],
                tool_call_id=tool_call_id,
                artifact=result["artifact"],
            )
        ],
        context_key: result["artifact"],
    }

    logger.info("Returning Command with update for context_key: %s", context_key)

    return Command(update=update_dict)
