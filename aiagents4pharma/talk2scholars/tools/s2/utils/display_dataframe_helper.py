"""
Utility for handling display dataframe operations.
"""

import logging
from typing import Dict, Any, Optional

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
        logger.info("DisplayDataHelper initialized with %s papers,sort_by=%s,ascending=%s,limit=%s",
                    len(artifact), sort_by, ascending, limit)

    def sort_papers(self) -> Dict[str, Any]:
        """
        Sort papers according to the specified column and order.
        
        Returns:
            Dictionary containing sorted paper data
        """
        artifact = self.artifact.copy()
        sort_by = self.sort_by
        ascending = self.ascending
        limit = self.limit

        original_count = len(artifact)
        logger.info("sort_papers starting with %s papers", original_count)

        if sort_by:
            logger.info("Sorting papers by %s, ascending=%s", sort_by, ascending)

            # Convert papers dict to list for sorting
            papers_list = list(artifact.values())

            # Check if the sort column exists in the papers
            if papers_list and sort_by in papers_list[0]:
                # For numeric fields, convert to appropriate type before sorting
                if sort_by in ['Citation Count', 'H-Index', 'Year']:
                    # Filter out 'N/A' values and normal values into separate lists
                    normal_papers = [p for p in papers_list if p[sort_by] != 'N/A']
                    na_papers = [p for p in papers_list if p[sort_by] == 'N/A']

                    # Sort normal papers
                    sorted_normal = sorted(
                        normal_papers,
                        key=lambda x: float(x[sort_by]),
                        reverse=not ascending
                    )

                    # Combine normal and N/A papers based on sort order
                    # In descending order, N/A values should be at the end
                    # In ascending order, N/A values can be at the beginning
                    if ascending:
                        sorted_papers = na_papers + sorted_normal
                    else:
                        sorted_papers = sorted_normal + na_papers
                else:
                    # For string fields
                    sorted_papers = sorted(
                        papers_list,
                        key=lambda x: x[sort_by] if x[sort_by] != 'N/A' else '',
                        reverse=not ascending
                    )

                # Limit the number of results if requested
                if limit is not None and limit > 0:
                    logger.info("Limiting results to top %s papers", limit)
                    sorted_papers = sorted_papers[:limit]
                    logger.info("After limiting: %s papers", len(sorted_papers))

                # Convert back to dictionary with paper IDs as keys
                artifact = {paper['semantic_scholar_paper_id']: paper for paper in sorted_papers}

                logger.info("Successfully sorted papers by %s, result has %s papers",
                            sort_by, len(artifact))
            else:
                logger.warning("Sort field '%s' not found in papers data", sort_by)

        # Limit the number of results even without sorting
        elif limit is not None and limit > 0:
            logger.info("Limiting results to top %s papers without sorting", limit)
            papers_list = list(artifact.values())
            limited_papers = papers_list[:limit]
            artifact = {paper['semantic_scholar_paper_id']: paper for paper in limited_papers}
            logger.info("After limiting without sorting: %s papers", len(artifact))

        return artifact

    def process_display(self) -> Dict[str, Any]:
        """
        Process the display request and return results.
        
        Returns:
            Dictionary containing processed papers and content message
        """
        sorted_artifact = self.sort_papers()

        content = f"{len(sorted_artifact)} papers found. Papers are attached as an artifact."
        if self.sort_by:
            order_text = 'ascending' if self.ascending else 'descending'
            content += f" Papers sorted by {self.sort_by} in {order_text} order."
        if self.limit is not None and self.limit > 0:
            content += f" Showing top {self.limit} results."

        logger.info("process_display returning artifact with %s papers", len(sorted_artifact))

        return {
            "artifact": sorted_artifact,
            "content": content
        }
