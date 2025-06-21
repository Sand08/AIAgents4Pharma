#!/usr/bin/env python3

"""
Tool for rendering the most recently displayed papers as a DataFrame artifact.

This module defines a tool that retrieves the paper metadata stored under the
state key 'last_displayed_papers' and returns it as an artifact (dictionary of
papers). The front-end can then render this artifact as a pandas DataFrame for
display. If no papers are found, a NoPapersFoundError is raised to indicate
that a search or recommendation should be performed first.
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
        description=(
            "Column to sort by. Options: 'Max H-Index', 'Citation Count', "
            "'Year', 'Title', 'Authors'. If not specified, papers are "
            "displayed in original order. This parameter should ONLY be set "
            "when the user explicitly requests sorting."
        )
    )
    ascending: bool = Field(
        default=False,
        description=(
            "Sort order. False for descending (highest first), "
            "True for ascending. Only used when sort_by is specified."
        )
    )
    limit: Optional[int] = Field(
        default=None,
        description=(
            "Number of top results to display after sorting. "
            "If not specified, all papers are shown. Only used when sort_by is specified."
        ),
        ge=1,
        le=100  # Increased from 10 to allow more flexibility
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
    containing the artifact (dictionary) for the front-end to render as a
    DataFrame. If no papers are found in state, it raises a NoPapersFoundError
    to indicate that a search or recommendation must be performed first.
    
    IMPORTANT: Only sorts papers when explicitly requested. Default behavior is
    to display papers in their original order without any sorting.

    Args:
        tool_call_id (InjectedToolCallId): Unique ID of this tool invocation.
        state (dict): The agent's state containing 'last_displayed_papers'.
        sort_by (str, optional): Column to sort by. Should ONLY be set when
            user explicitly requests sorting.
        ascending (bool): Sort order - False for descending (default),
            True for ascending. Only used when sort_by is specified.
        limit (int, optional): Number of top results to display after sorting.
            Only used when sort_by is specified.

    Returns:
        Command: A command whose update contains a ToolMessage with the
                 artifact (papers dict) for DataFrame rendering in the UI.

    Raises:
        NoPapersFoundError: If no entries exist under 'last_displayed_papers'
            in state.
    """
    # Clear logging to show exact parameters received
    logger.info(
        "display_dataframe called with: sort_by=%s, ascending=%s, limit=%s",
        sort_by, ascending, limit
    )

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
            "No papers found. A search/recommendation needs to be performed first."
        )

    # Initialize helper
    helper = DisplayHelper(papers_dict)

    # CRITICAL: Only apply sorting if sort_by is explicitly specified
    # This prevents carrying over sorting from previous calls
    if sort_by is not None and sort_by != "":
        logger.info("Sorting explicitly requested by: %s", sort_by)
        helper.prepare_dataframe(
            sort_by=sort_by,
            ascending=ascending,
            limit=limit
        )
        artifact = helper.get_sorted_dict()
        # Create appropriate content message with sorting info
        content = helper.format_summary(sort_by=sort_by, limit=limit)
    else:
        # No sorting requested - return original papers in their original order
        logger.info("No sorting requested, displaying papers in original order")
        artifact = papers_dict
        # Simple message without sorting info
        displayed_count = len(papers_dict)
        content = (
            f"{displayed_count} papers found. "
            "Papers are displayed in their original order. "
            "Papers are attached as an artifact."
        )

    # Log what we're returning for debugging
    logger.info(
        "Returning %d papers, sorted=%s",
        len(artifact),
        sort_by is not None
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=artifact,
                )
            ],
        }
    )
