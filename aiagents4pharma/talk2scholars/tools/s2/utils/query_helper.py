#!/usr/bin/env python3

"""
Utility for query_dataframe tool - handles sorting and preparation of papers.
"""

import logging
from typing import Dict, Optional, Any
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryHelper:
    """Helper class for query dataframe operations."""

    def __init__(self, papers_dict: Dict[str, Dict[str, Any]]):
        """
        Initialize QueryHelper with papers dictionary.
        
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
        
        This method prepares the DataFrame for querying, handling:
        - Conversion from dictionary to DataFrame
        - Numeric conversion for bibliographic metrics
        - Sorting by specified column
        - Limiting results to top N papers
        
        Args:
            sort_by: Column name to sort by ('Max H-Index', 'Citation Count',
                     'Year', 'Title', 'Authors')
            ascending: Sort order (False for descending, True for ascending)
            limit: Number of top results to return after sorting
            
        Returns:
            pd.DataFrame: Formatted and optionally sorted DataFrame ready
                         for querying
        """
        # Convert to DataFrame
        self.df = pd.DataFrame.from_dict(self.papers_dict, orient='index')

        # Ensure index has a name for better querying
        self.df.index.name = 'paper_id'

        # Handle sorting if requested
        if sort_by and sort_by in self.df.columns:
            logger.info(
                "Preparing dataframe: sorting by %s (%s)",
                sort_by,
                'ascending' if ascending else 'descending'
            )

            # Convert sorting column to numeric if needed
            if sort_by in ['Max H-Index', 'Citation Count', 'Year']:
                # Handle various representations of N/A values
                na_values = ['N/A', 'n/a', 'NA', 'None', '', None]

                # Create a copy to avoid modifying original data
                self.df = self.df.copy()

                # Replace N/A values with None for proper sorting
                self.df[sort_by] = self.df[sort_by].replace(na_values, None)

                # Convert to numeric, handling errors gracefully
                self.df[sort_by] = pd.to_numeric(
                    self.df[sort_by],
                    errors='coerce'
                )

                # Log conversion results for debugging
                na_count = self.df[sort_by].isna().sum()
                if na_count > 0:
                    logger.info(
                        "Found %d N/A values in %s column",
                        na_count, sort_by
                    )

            # Sort the dataframe
            self.df = self.df.sort_values(
                by=sort_by,
                ascending=ascending,
                na_position='last'  # Put N/A values at the end
            )

            # Apply limit if specified
            if limit and limit > 0:
                original_count = len(self.df)
                self.df = self.df.head(limit)
                logger.info(
                    "Limited results from %d to top %d papers",
                    original_count, limit
                )

        # Reset index to make paper_id accessible as a column
        self.df = self.df.reset_index()

        return self.df

    def get_column_info(self) -> Dict[str, str]:
        """
        Get information about available columns and their data types.
        
        Returns:
            Dict: Column names mapped to their data types
        """
        if self.df is None:
            self.df = pd.DataFrame.from_dict(self.papers_dict, orient='index')

        return {
            col: str(dtype) for col, dtype in self.df.dtypes.items()
        }
