from collections import OrderedDict
import heapq
import logging
from typing import Dict, Any, Optional, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUMERIC_FIELDS = {"Citation Count", "Max H-Index", "Year"}

class DisplayDataHelper:
    """Helper class to organize display dataframe operations."""

    def __init__(
        self,
        artifact: Dict[str, Dict[str, Any]],
        sort_by: Optional[str] = None,
        ascending: bool = False,
        limit: Optional[int] = None,
    ):
        self.artifact = artifact
        self.sort_by = sort_by
        self.ascending = ascending
        self.limit = limit
        logger.info(
            "Initialized with %s papers, sort_by=%s, ascending=%s, limit=%s",
            len(artifact), sort_by, ascending, limit,
        )

    def _get_sort_key(self, item: Tuple[str, Dict[str, Any]]) -> Any:
        """
        Compute sort key for an (id, paper) tuple based on self.sort_by.
        Numeric fields convert to float, invalid/N-A map to extremes.
        String fields default to empty or reversed high-string.
        """
        _, paper = item
        field = self.sort_by
        v = paper.get(field, None)

        # Numeric field handling
        if field in NUMERIC_FIELDS:
            try:
                return float(v)
            except Exception:
                # Push invalids to end or start
                return float('inf') if self.ascending else float('-inf')

        # String field handling
        if v is None or v == 'N/A':
            return '' if self.ascending else chr(0x10FFFF)
        return str(v)

    def sort_papers(self) -> Dict[str, Dict[str, Any]]:
        """
        Sort (and/or limit) the papers and return a new OrderedDict preserving order.
        """
        if not self.artifact:
            logger.warning("No papers to process")
            return OrderedDict()

        items = list(self.artifact.items())

        # If sorting requested and field exists in any item
        if self.sort_by and any(self.sort_by in paper for _, paper in items):
            logger.info("Sorting by %s, ascending=%s", self.sort_by, self.ascending)
            # Choose full sort or top-K
            if self.limit and self.limit > 0 and len(items) > self.limit:
                if self.ascending:
                    selected = heapq.nsmallest(
                        self.limit, items, key=self._get_sort_key
                    )
                else:
                    selected = heapq.nlargest(
                        self.limit, items, key=self._get_sort_key
                    )
                sorted_items = selected
            else:
                sorted_items = sorted(
                    items,
                    key=self._get_sort_key,
                    reverse=not self.ascending,
                )
        else:
            # No sort or invalid field: apply limit only
            if self.limit and self.limit > 0:
                sorted_items = items[: self.limit]
            else:
                sorted_items = items

        # Build OrderedDict to preserve new order
        result = OrderedDict()
        for key, paper in sorted_items:
            result[key] = paper
        logger.info(
            "Returning %s papers after sort/limit", len(result)
        )
        return result

    def process_display(self) -> Dict[str, Any]:
        """
        Apply sort+limit and produce display payload.
        """
        sorted_artifact = self.sort_papers()
        count = len(sorted_artifact)
        content = f"{count} papers found. Papers are attached as an artifact."
        if self.sort_by:
            order = 'ascending' if self.ascending else 'descending'
            content += f" Papers sorted by {self.sort_by} in {order} order."
        if self.limit and self.limit > 0:
            content += f" Showing top {self.limit} results."
        return {"artifact": sorted_artifact, "content": content}
