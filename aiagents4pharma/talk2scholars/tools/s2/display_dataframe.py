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
from typing import Annotated, Optional

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
    
    state: Annotated[dict, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]
    sort_by: Optional[str] = Field(
        default=None,
        description="Column to sort by. Common options include 'Citation Count', 'H-Index', or 'Year'."
    )
    ascending: bool = Field(
        default=False,
        description="Sort order: True for ascending, False for descending (default)."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit the number of results. For example, limit=5 will show only the top 5 papers."
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
    Render the last set of retrieved papers as a DataFrame in the front-end.
    
    This function reads the 'last_displayed_papers' key from state, fetches the
    corresponding metadata dictionary, and returns a Command with a ToolMessage
    containing the artifact (dictionary) for the front-end to render as a DataFrame.
    If no papers are found in state, it raises a NoPapersFoundError to indicate
    that a search or recommendation must be performed first.
    
    The results can be sorted by bibliographic metrics such as 'Citation Count', 'Year', 
    or 'H-Index'.
    
    Args:
        tool_call_id (InjectedToolCallId): Unique ID of this tool invocation.
        state (dict): The agent's state containing the 'last_displayed_papers' reference.
        sort_by (str, optional): Column to sort by, such as 'Citation Count', 'H-Index', or 'Year'.
        ascending (bool, optional): Sort order - True for ascending, False for descending.
    
    Returns:
        Command: A command whose update contains a ToolMessage with the artifact
                 (papers dict) for DataFrame rendering in the UI.
    
    Raises:
        NoPapersFoundError: If no entries exist under 'last_displayed_papers' in state.
    """
    logger.info("Displaying papers")
    context_key = state.get("last_displayed_papers")
    artifact = state.get(context_key)
    if not artifact:
        logger.info("No papers found in state, raising NoPapersFoundError")
        raise NoPapersFoundError(
            "No papers found. A search/rec needs to be performed first."
        )
    
    # Sort papers if sorting parameters are provided
    if sort_by:
        logger.info(f"Sorting papers by {sort_by}, ascending={ascending}")
        try:
            # Convert papers dict to list for sorting
            papers_list = list(artifact.values())
            
            # Check if the sort column exists in the papers
            if papers_list and sort_by in papers_list[0]:
                # For numeric fields, convert to appropriate type before sorting
                if sort_by in ['Citation Count', 'H-Index', 'Year']:
                    # Handle 'N/A' values by placing them at the end
                    sorted_papers = sorted(
                        papers_list,
                        key=lambda x: float(x[sort_by]) if x[sort_by] != 'N/A' and str(x[sort_by]).replace('.', '', 1).isdigit() else float('-inf' if ascending else 'inf'),
                        reverse=not ascending
                    )
                else:
                    # For string fields
                    sorted_papers = sorted(
                        papers_list,
                        key=lambda x: x[sort_by] if x[sort_by] != 'N/A' else '',
                        reverse=not ascending
                    )
                
# Limit the number of results if requested
                if limit is not None and limit > 0:
                    logger.info(f"Limiting results to top {limit} papers")
                    sorted_papers = sorted_papers[:limit]
                
                # Convert back to dictionary with paper IDs as keys
                artifact = {paper['semantic_scholar_paper_id']: paper for paper in sorted_papers}
                
                logger.info(f"Successfully sorted papers by {sort_by}")
            else:
                logger.warning(f"Sort field '{sort_by}' not found in papers data")
        except Exception as e:
            logger.error(f"Error sorting papers: {e}")
            # Continue with unsorted papers if sorting fails
    
    # Limit the number of results even without sorting
    elif limit is not None and limit > 0:
        logger.info(f"Limiting results to top {limit} papers")
        papers_list = list(artifact.values())
        limited_papers = papers_list[:limit]
        artifact = {paper['semantic_scholar_paper_id']: paper for paper in limited_papers}
    
    content = f"{len(artifact)} papers found. Papers are attached as an artifact."
    if sort_by:
        content += f" Papers sorted by {sort_by} in {'ascending' if ascending else 'descending'} order."
    if limit is not None and limit > 0:
        content += f" Showing top {limit} results."
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    artifact=artifact,
                )
            ],
            context_key: artifact,
        }
    )