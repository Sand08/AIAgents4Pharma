#!/usr/bin/env python3


"""
Tool for rendering the most recently displayed papers as a DataFrame artifact for the front-end.

This module defines a tool that retrieves the paper metadata stored under the state key
'last_displayed_papers' and returns it as an artifact (dictionary of papers). The front-end
can then render this artifact as a pandas DataFrame for display. If no papers are found,
a NoPapersFoundError is raised to indicate that a search or recommendation should be
performed first.
"""


import logging
from typing import Annotated, Optional
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.display_helper import DisplayHelper


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoPapersFoundError(Exception):
    """
    Exception raised when no research papers are found in the agent's state.

    This exception helps the language model determine whether a new search
    or recommendation should be initiated.

    Example:
        >>> if not papers:
        >>>     raise NoPapersFoundError("No papers found. A search is needed.")
    """


class DisplayDataframeInput(BaseModel):
    """Input schema for the display dataframe tool."""

    sort_by: Optional[str] = Field(
        default=None,
        description="Column to sort by. Options: 'Max H-Index', 'Citation Count', 'Year', etc. "
        "If not specified, papers are displayed in original order."
    )
    ascending: bool = Field(
        default=False,
        description="Sort order. False for descending (highest first), True for ascending."
    )
    limit: Optional[int] = Field(
        default=None,
        description=("Number of top results to display after sorting. "
                     "If not specified, all papers are shown."),
        ge=1,
        le=100
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]


@tool("display_dataframe", args_schema=DisplayDataframeInput, parse_docstring=True)
def display_dataframe(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    sort_by: Optional[str] = None,
    ascending: bool = False,
    limit: Optional[int] = None,
) -> Command:
    """
    Render the last set of retrieved papers as a DataFrame in the front-end.
    
    This function reads the 'last_displayed_papers' key from state, fetches the
    corresponding metadata dictionary, and returns a Command with a ToolMessage
    containing the artifact (dictionary) for the front-end to render as a DataFrame.
    If no papers are found in state, it raises a NoPapersFoundError to indicate
    that a search or recommendation must be performed first.
    
    Optionally sorts the papers by bibliographic metrics before display.

    Args:
        tool_call_id (InjectedToolCallId): Unique ID of this tool invocation.
        state (dict): The agent's state containing the 'last_displayed_papers' reference.
        sort_by (str, optional): Column to sort by ('Max H-Index', 'Citation Count', 'Year', etc.)
        ascending (bool): Sort order - False for descending (default), True for ascending
        limit (int, optional): Number of top results to display after sorting

    Returns:
        Command: A command whose update contains a ToolMessage with the artifact
                 (papers dict) for DataFrame rendering in the UI.

    Raises:
        NoPapersFoundError: If no entries exist under 'last_displayed_papers' in state.
    """
    logger.info("Displaying papers")

    # Get papers from state
    context_val = state.get("last_displayed_papers")
    # Support both key reference (str) and direct mapping
    if isinstance(context_val, dict):
        papers_dict = context_val
    else:
        papers_dict = state.get(context_val)

    if not papers_dict:
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search/rec needs to be performed first."
        )

    # Use helper for formatting
    helper = DisplayHelper(papers_dict)

    # Only apply sorting if sort_by is explicitly specified
    if sort_by:
        logger.info("Applying sorting by %s", sort_by)
        helper.prepare_dataframe(sort_by=sort_by, ascending=ascending, limit=limit)
        artifact = helper.get_sorted_dict()
        # Create appropriate content message with sorting info
        content = helper.format_summary(sort_by=sort_by, limit=limit)

        # IMPORTANT: Update the state with the filtered/sorted papers so query_dataframe sees
        state["last_displayed_papers"] = artifact
    else:
        # No sorting requested - return original papers
        logger.info("No sorting requested, displaying papers in original order")
        artifact = papers_dict
        # Simple message without sorting info
        content = f"{len(papers_dict)} papers found. Papers are attached as an artifact."

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=artifact,
                )
            ],
            # Update the last_displayed_papers to reflect what's actually being shown
            "last_displayed_papers": artifact,
        }
    )
