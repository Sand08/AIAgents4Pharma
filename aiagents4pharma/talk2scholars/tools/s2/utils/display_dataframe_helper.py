#!/usr/bin/env python3

"""
Utility for handling display dataframe operations.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisplayDataHelper:
    """Helper class to organize display dataframe operations."""

    def __init__(self, artifact: Dict[str, Any], sort_by: Optional[str] = None, 
                 ascending: bool = False, limit: Optional[int] = None):
        """
        Initialize the DisplayDataHelper with papers data and sorting preferences.
        
        Args:
            artifact: Dictionary containing paper data
            sort_by: Column to sort by
            ascending: Sort order (True for ascending, False for descending)
            limit: Maximum number of results to show
        """
        self.artifact = artifact
        self.sort_by = sort_by
        self.ascending = ascending
        self.limit = limit
        logger.info(f"DisplayDataHelper initialized with {len(artifact)} papers, sort_by={sort_by}, ascending={ascending}, limit={limit}")

    def sort_papers(self) -> Dict[str, Any]:
        """
        Sort papers according to the specified column and order.
        
        Returns:
            Dictionary containing sorted paper data
        """
        artifact = self.artifact.copy()  # Work with a copy to avoid modifying the original
        sort_by = self.sort_by
        ascending = self.ascending
        limit = self.limit
        
        original_count = len(artifact)
        logger.info(f"sort_papers starting with {original_count} papers")
        
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
                        logger.info(f"After limiting: {len(sorted_papers)} papers")
                    
                    # Convert back to dictionary with paper IDs as keys
                    artifact = {paper['semantic_scholar_paper_id']: paper for paper in sorted_papers}
                    
                    logger.info(f"Successfully sorted papers by {sort_by}, result has {len(artifact)} papers")
                else:
                    logger.warning(f"Sort field '{sort_by}' not found in papers data")
            except Exception as e:
                logger.error(f"Error sorting papers: {e}")
                # Continue with unsorted papers if sorting fails
        
        # Limit the number of results even without sorting
        elif limit is not None and limit > 0:
            logger.info(f"Limiting results to top {limit} papers without sorting")
            papers_list = list(artifact.values())
            limited_papers = papers_list[:limit]
            artifact = {paper['semantic_scholar_paper_id']: paper for paper in limited_papers}
            logger.info(f"After limiting without sorting: {len(artifact)} papers")
        
        return artifact

    def process_display(self) -> Dict[str, Any]:
        """
        Process the display request and return results.
        
        Returns:
            Dictionary containing processed papers and content message
        """
        sorted_artifact = self.sort_papers()
        
        # Create content based on the SORTED artifact, not the original
        content = f"{len(sorted_artifact)} papers found. Papers are attached as an artifact."
        if self.sort_by:
            content += f" Papers sorted by {self.sort_by} in {'ascending' if self.ascending else 'descending'} order."
        if self.limit is not None and self.limit > 0:
            content += f" Showing top {self.limit} results."
        
        logger.info(f"process_display returning artifact with {len(sorted_artifact)} papers")
        
        return {
            "artifact": sorted_artifact,
            "content": content
        }