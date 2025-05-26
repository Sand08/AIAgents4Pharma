"""
Comprehensive unit tests for S2 display functionality including display_dataframe and DisplayHelper.
"""

from unittest.mock import patch
import pytest
import pandas as pd
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from aiagents4pharma.talk2scholars.tools.s2.display_dataframe import (
    display_dataframe,
    NoPapersFoundError,
)
from aiagents4pharma.talk2scholars.tools.s2.utils.display_helper import DisplayHelper


# --- Test Data ---
MOCK_PAPERS_DICT = {
    "paper1": {
        "semantic_scholar_paper_id": "paper1",
        "Title": "Deep Learning Fundamentals",
        "Abstract": "A comprehensive guide to deep learning",
        "Year": 2023,
        "Citation Count": 150,
        "Max H-Index": 45,
        "Authors": ["Author A", "Author B"],
        "URL": "https://example.com/paper1",
    },
    "paper2": {
        "semantic_scholar_paper_id": "paper2",
        "Title": "Advanced Machine Learning",
        "Abstract": "Advanced techniques in ML",
        "Year": 2022,
        "Citation Count": 200,
        "Max H-Index": 60,
        "Authors": ["Author C", "Author D"],
        "URL": "https://example.com/paper2",
    },
    "paper3": {
        "semantic_scholar_paper_id": "paper3",
        "Title": "Neural Networks",
        "Abstract": "Introduction to neural networks",
        "Year": 2024,
        "Citation Count": "N/A",
        "Max H-Index": 30,
        "Authors": ["Author E"],
        "URL": "https://example.com/paper3",
    },
}

MOCK_PAPERS_WITH_NA = {
    "paper1": {
        "Title": "Paper with NA values",
        "Year": "N/A",
        "Citation Count": "N/A",
        "Max H-Index": None,
        "Authors": ["Author X"],
    },
    "paper2": {
        "Title": "Paper with values",
        "Year": 2023,
        "Citation Count": 100,
        "Max H-Index": 50,
        "Authors": ["Author Y"],
    },
}


# --- DisplayHelper Tests ---
class TestDisplayHelper:
    """Unit tests for DisplayHelper class"""

    def test_init_with_papers(self):
        """Test DisplayHelper initialization with papers dictionary"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        assert helper.papers_dict == MOCK_PAPERS_DICT
        assert helper.df is None

    def test_prepare_dataframe_no_sorting(self):
        """Test prepare_dataframe without sorting"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Title" in df.columns
        assert "Citation Count" in df.columns

    def test_prepare_dataframe_sort_by_citation_count(self):
        """Test sorting by Citation Count (descending)"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Citation Count", ascending=False)

        # paper2 should be first (200), paper1 second (150), paper3 last (N/A)
        assert df.index[0] == "paper2"
        assert df.index[1] == "paper1"
        assert df.index[2] == "paper3"

    def test_prepare_dataframe_sort_by_year_ascending(self):
        """Test sorting by Year (ascending)"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Year", ascending=True)

        # paper2 (2022) should be first, paper1 (2023) second, paper3 (2024) last
        assert df.index[0] == "paper2"
        assert df.index[1] == "paper1"
        assert df.index[2] == "paper3"

    def test_prepare_dataframe_with_limit(self):
        """Test limiting results"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Max H-Index", ascending=False, limit=2)

        assert len(df) == 2
        # paper2 (60) and paper1 (45) should be included
        assert "paper2" in df.index
        assert "paper1" in df.index
        assert "paper3" not in df.index

    def test_prepare_dataframe_handle_na_values(self):
        """Test handling of N/A values in numeric columns"""
        helper = DisplayHelper(MOCK_PAPERS_WITH_NA)
        df = helper.prepare_dataframe(sort_by="Citation Count", ascending=False)

        # paper2 with value 100 should be first, paper1 with N/A should be last
        assert df.index[0] == "paper2"
        assert df.index[1] == "paper1"
        # Check that N/A was converted to NaN
        assert pd.isna(df.loc["paper1", "Citation Count"])

    def test_get_sorted_dict_no_dataframe(self):
        """Test get_sorted_dict when no dataframe exists"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        sorted_dict = helper.get_sorted_dict()

        assert sorted_dict == MOCK_PAPERS_DICT

    def test_get_sorted_dict_with_sorting(self):
        """Test get_sorted_dict after sorting"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        helper.prepare_dataframe(sort_by="Citation Count", ascending=False)
        sorted_dict = helper.get_sorted_dict()

        # Check order is maintained
        keys = list(sorted_dict.keys())
        assert keys[0] == "paper2"  # 200 citations
        assert keys[1] == "paper1"  # 150 citations
        assert keys[2] == "paper3"  # N/A citations

    def test_format_summary_no_sorting(self):
        """Test format_summary without sorting"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        helper.prepare_dataframe()
        summary = helper.format_summary()

        assert "3 papers found." in summary
        assert "Papers are attached as an artifact." in summary
        assert "Sorted by" not in summary

    def test_format_summary_with_sorting(self):
        """Test format_summary with sorting"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        helper.prepare_dataframe(sort_by="Year", ascending=True)
        summary = helper.format_summary(sort_by="Year")

        assert "3 papers found." in summary
        assert "Sorted by Year" in summary
        assert "ascending" in summary

    def test_format_summary_with_limit(self):
        """Test format_summary with limit"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        helper.prepare_dataframe(sort_by="Citation Count", limit=2)
        summary = helper.format_summary(sort_by="Citation Count", limit=2)

        assert "2 papers found." in summary
        assert "Showing top 2 results." in summary

    def test_sort_by_title(self):
        """Test sorting by Title (alphabetical)"""
        helper = DisplayHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Title", ascending=True)

        # Advanced ML < Deep Learning < Neural Networks
        assert df.index[0] == "paper2"
        assert df.index[1] == "paper1"
        assert df.index[2] == "paper3"


# --- display_dataframe Tool Tests ---
class TestDisplayDataframeTool:
    """Unit tests for display_dataframe tool"""

    def test_display_dataframe_no_papers_in_state(self):
        """Test display_dataframe raises error when no papers in state"""
        state = {"last_displayed_papers": None}

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search/rec needs to be performed first."
        ):
            display_dataframe.invoke({
                "state": state,
                "tool_call_id": "test123"
            })

    def test_display_dataframe_basic_success(self):
        """Test basic display without sorting"""
        state = {
            "last_displayed_papers": "papers",
            "papers": MOCK_PAPERS_DICT
        }

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123"
        })

        assert isinstance(result, Command)
        update = result.update
        assert "messages" in update
        assert "last_displayed_papers" in update

        messages = update["messages"]
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, ToolMessage)
        assert "3 papers found." in msg.content
        assert msg.artifact == MOCK_PAPERS_DICT

    def test_display_dataframe_with_sorting(self):
        """Test display with sorting"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT
        }

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Citation Count",
            "ascending": False
        })

        assert isinstance(result, Command)
        update = result.update

        # Check artifact is sorted
        artifact = update["messages"][0].artifact
        keys = list(artifact.keys())
        assert keys[0] == "paper2"  # 200 citations
        assert keys[1] == "paper1"  # 150 citations

        # Check message contains sorting info
        content = update["messages"][0].content
        assert "Sorted by Citation Count" in content
        assert "descending" in content

    def test_display_dataframe_with_limit(self):
        """Test display with limit"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT
        }

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Max H-Index",
            "ascending": False,
            "limit": 2
        })

        update = result.update
        artifact = update["messages"][0].artifact

        # Should only have 2 papers
        assert len(artifact) == 2
        assert "paper2" in artifact  # H-Index 60
        assert "paper1" in artifact  # H-Index 45
        assert "paper3" not in artifact  # H-Index 30

        # Check message mentions limit
        content = update["messages"][0].content
        assert "Showing top 2 results" in content

    def test_display_dataframe_direct_dict_mapping(self):
        """Test when last_displayed_papers is direct dict instead of key"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT
        }

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123"
        })

        assert isinstance(result, Command)
        assert result.update["messages"][0].artifact == MOCK_PAPERS_DICT

    def test_display_dataframe_updates_state(self):
        """Test that display_dataframe updates state with sorted/filtered results"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT
        }

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Year",
            "limit": 1
        })

        update = result.update
        # State should be updated with the filtered result
        assert "last_displayed_papers" in update
        updated_papers = update["last_displayed_papers"]
        assert len(updated_papers) == 1

    def test_display_dataframe_empty_papers_dict(self):
        """Test handling of empty papers dictionary"""
        state = {
            "last_displayed_papers": {}
        }

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search/rec needs to be performed first."
        ):
            display_dataframe.invoke({
                "state": state,
                "tool_call_id": "test123"
            })

    @patch("aiagents4pharma.talk2scholars.tools.s2.display_dataframe.logger")
    def test_display_dataframe_logging(self, mock_logger):
        """Test that appropriate logging occurs"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT
        }

        display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Year"
        })

        # Check logging calls
        mock_logger.info.assert_any_call(
            "Displaying papers with sort_by=%s, limit=%s", "Year", None
        )
