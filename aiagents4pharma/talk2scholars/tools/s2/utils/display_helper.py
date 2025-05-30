#!/usr/bin/env python3

"""
Utility for display_dataframe tool - handles sorting and formatting of papers.
"""

import logging
from typing import Dict, Optional, Any
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisplayHelper:
    """Helper class for display dataframe operations."""

    def __init__(self, papers_dict: Dict[str, Dict[str, Any]]):
        """
        Initialize DisplayHelper with papers dictionary.
        
        Args:
            papers_dict: Dictionary of papers with paper_id as key
        """
        self.papers_dict = papers_dict
        self.df = None

    def prepare_dataframe(
        self,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert papers dictionary to DataFrame and apply sorting if requested.
        
        Args:
            sort_by: Column name to sort by ('Max H-Index', 'Citation Count',
                     'Year', 'Title', 'Authors')
            ascending: Sort order (False for descending, True for ascending)
            limit: Number of top results to return after sorting
            
        Returns:
            pd.DataFrame: Formatted and optionally sorted DataFrame
        """
        # Convert to DataFrame
        self.df = pd.DataFrame.from_dict(self.papers_dict, orient='index')

        # Handle sorting if requested
        if sort_by and sort_by in self.df.columns:
            logger.info(
                "Sorting by %s (%s)",
                sort_by,
                'ascending' if ascending else 'descending'
            )

            # Convert sorting column to numeric if needed
            if sort_by in ['Max H-Index', 'Citation Count', 'Year']:
                # Handle various representations of N/A values
                na_values = ['N/A', 'n/a', 'NA', 'None', '', None]

                # Replace N/A values with None for proper sorting
                self.df[sort_by] = self.df[sort_by].replace(na_values, None)

                # Convert to numeric
                self.df[sort_by] = pd.to_numeric(
                    self.df[sort_by],
                    errors='coerce'
                )

            # Sort the dataframe
            self.df = self.df.sort_values(
                by=sort_by,
                ascending=ascending,
                na_position='last'  # Put N/A values at the end
            )

            # Apply limit if specified
            if limit and limit > 0:
                self.df = self.df.head(limit)
                logger.info("Limited results to top %d papers", limit)

        return self.df

    def get_sorted_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the sorted papers as a dictionary maintaining the sorted order.
        
        Returns:
            Dict: Sorted papers dictionary
        """
        if self.df is None:
            return self.papers_dict

        # Convert back to dictionary maintaining order
        sorted_dict = {}
        for idx in self.df.index:
            sorted_dict[idx] = self.papers_dict[idx]

        return sorted_dict

    def format_summary(
        self,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Create a formatted summary of the papers.
        
        Args:
            sort_by: Column that was used for sorting
            limit: Number of papers displayed
            
        Returns:
            str: Formatted summary message
        """
        total_papers = len(self.papers_dict)
        displayed_papers = (
            len(self.df) if self.df is not None else total_papers
        )

        summary = f"{displayed_papers} papers found."

        if sort_by:
            order = "ascending" if self.df is not None else "descending"
            # Check the actual sort order from the dataframe
            if self.df is not None and len(self.df) > 1:
                if sort_by in ['Max H-Index', 'Citation Count', 'Year']:
                    # For numeric columns, check if values are increasing
                    vals = self.df[sort_by].dropna()
                    if len(vals) > 1:
                        order = (
                            "ascending" if vals.iloc[0] < vals.iloc[-1]
                            else "descending"
                        )

            summary += f" Sorted by {sort_by} ({order})."

            if limit and limit < total_papers:
                summary += f" Showing top {limit} results."

        summary += " Papers are attached as an artifact."

        return summary
