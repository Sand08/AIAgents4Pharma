#!/usr/bin/env python3
"""
Tool for rendering the most recently displayed papers as a DataFrame artifact for the front-end.
This module defines a tool that retrieves the paper metadata stored under the state key
'last_displayed_papers' and returns it as an artifact (dictionary of papers).

The tool can optionally sort papers by bibliographic metrics (Citation Count, H-Index, Year) 
when explicitly requested, but will not sort by default.
"""
import logging
from typing import Annotated, Optional
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
    sort_by: Optional[str] = Field(
        default=None,
        description=(
            "Column to sort by when explicitly requested. Common metrics include "
            "'Citation Count', 'H-Index', or 'Year'. Leave empty for no sorting."
        )
    )
    ascending: bool = Field(
        default=False,
        description=(
            "Sort order: True for ascending, False for descending. "
            "Only applies when sort_by is specified."
        )
    )
    limit: Optional[int] = Field(
        default=None,
        description=(
            "Limit the number of results. For example, "
            "limit=5 will show only the top 5 papers."
        )
    )


@tool("display_dataframe", args_schema=DisplayDataframeInput, parse_docstring=True)
def display_dataframe(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    sort_by: Optional[str] = None,
    ascending: bool = False,
    limit: Optional[int] = None,
) -> Command:
    """
    Display the last set of retrieved papers in the front-end without any default sorting.

    This function simply displays the papers in their original order unless explicitly 
    asked to sort. It reads the 'last_displayed_papers' key from state and returns a 
    Command with a ToolMessage containing the papers artifact for the front-end to render.

    The papers can be optionally sorted by bibliographic metrics only when 
    explicitly requested:
    - Use 'Citation Count' to sort by number of citations
    - Use 'H-Index' to sort by author H-Index values
    - Use 'Year' to sort chronologically
    
    By default, no sorting is applied unless specifically requested.

    Args:
        tool_call_id (InjectedToolCallId): Unique ID of this tool invocation.
        state (dict): The agent's state containing the 'last_displayed_papers' reference.
        sort_by (str, optional): Column to sort by when explicitly requested, such as
            'Citation Count', 'H-Index', or 'Year'. No sorting is applied if this is None.
        ascending (bool, optional): Sort order - True for ascending, False for descending. Only
            applies when sort_by is specified. 
        limit (int, optional): Limit the number of results (e.g., limit=5 shows top 5 papers).

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
