"""
Comprehensive unit tests for S2 query functionality including query_dataframe and QueryHelper.
"""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from aiagents4pharma.talk2scholars.tools.s2.query_dataframe import (
    query_dataframe,
    NoPapersFoundError,
)
from aiagents4pharma.talk2scholars.tools.s2.utils.query_helper import QueryHelper


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
        "Citation Count": "n/a",
        "Max H-Index": "",
        "Authors": ["Author X"],
    },
    "paper2": {
        "Title": "Paper with values",
        "Year": 2023,
        "Citation Count": 100,
        "Max H-Index": 50,
        "Authors": ["Author Y"],
    },
    "paper3": {
        "Title": "Paper with None",
        "Year": None,
        "Citation Count": "None",
        "Max H-Index": None,
        "Authors": ["Author Z"],
    },
}


# --- QueryHelper Tests ---
class TestQueryHelper:
    """Unit tests for QueryHelper class"""

    def test_init_with_papers(self):
        """Test QueryHelper initialization with papers dictionary"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        assert helper.papers_dict == MOCK_PAPERS_DICT
        assert helper.df is None

    def test_prepare_dataframe_no_sorting(self):
        """Test prepare_dataframe without sorting"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Title" in df.columns
        assert "Citation Count" in df.columns
        assert "paper_id" in df.columns  # Should have reset index
        assert df.index.name is None  # Index should be numeric after reset

    def test_prepare_dataframe_sort_by_citation_count(self):
        """Test sorting by Citation Count (descending)"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Citation Count", ascending=False)

        # Check order in paper_id column
        assert df.iloc[0]["paper_id"] == "paper2"  # 200 citations
        assert df.iloc[1]["paper_id"] == "paper1"  # 150 citations
        assert df.iloc[2]["paper_id"] == "paper3"  # N/A citations

    def test_prepare_dataframe_sort_by_year_ascending(self):
        """Test sorting by Year (ascending)"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Year", ascending=True)

        # Check order
        assert df.iloc[0]["paper_id"] == "paper2"  # 2022
        assert df.iloc[1]["paper_id"] == "paper1"  # 2023
        assert df.iloc[2]["paper_id"] == "paper3"  # 2024

    def test_prepare_dataframe_with_limit(self):
        """Test limiting results"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Max H-Index", ascending=False, limit=2)

        assert len(df) == 2
        # Top 2 by H-Index
        paper_ids = df["paper_id"].tolist()
        assert "paper2" in paper_ids  # H-Index 60
        assert "paper1" in paper_ids  # H-Index 45
        assert "paper3" not in paper_ids  # H-Index 30

    def test_prepare_dataframe_handle_na_values(self):
        """Test handling of various N/A representations"""
        helper = QueryHelper(MOCK_PAPERS_WITH_NA)
        df = helper.prepare_dataframe(sort_by="Citation Count", ascending=False)

        # paper2 with value 100 should be first
        assert df.iloc[0]["paper_id"] == "paper2"

        # Check that various N/A values were converted properly
        paper1_row = df[df["paper_id"] == "paper1"].iloc[0]
        paper3_row = df[df["paper_id"] == "paper3"].iloc[0]

        assert pd.isna(paper1_row["Citation Count"])
        assert pd.isna(paper3_row["Citation Count"])

    def test_prepare_dataframe_preserves_original_data(self):
        """Test that prepare_dataframe doesn't modify original papers_dict"""
        helper = QueryHelper(MOCK_PAPERS_WITH_NA)
        original_citation = helper.papers_dict["paper1"]["Citation Count"]

        helper.prepare_dataframe(sort_by="Citation Count")

        # Original dict should be unchanged
        assert helper.papers_dict["paper1"]["Citation Count"] == original_citation

    def test_get_column_info(self):
        """Test get_column_info returns proper column information"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        column_info = helper.get_column_info()

        assert isinstance(column_info, dict)
        assert "Title" in column_info
        assert "Year" in column_info
        assert "Citation Count" in column_info
        # Check data types are converted to strings
        assert all(isinstance(dtype, str) for dtype in column_info.values())

    @patch("aiagents4pharma.talk2scholars.tools.s2.utils.query_helper.logger")
    def test_prepare_dataframe_logging(self, mock_logger):
        """Test logging during dataframe preparation"""
        helper = QueryHelper(MOCK_PAPERS_WITH_NA)
        helper.prepare_dataframe(sort_by="Year", ascending=True, limit=2)

        # Check logging calls
        mock_logger.info.assert_any_call(
            "Preparing dataframe: sorting by %s (%s)",
            "Year",
            "ascending"
        )
        mock_logger.info.assert_any_call(
            "Limited results from %d to top %d papers",
            3, 2
        )

    def test_sort_by_authors(self):
        """Test sorting by Authors (should work as string sorting)"""
        helper = QueryHelper(MOCK_PAPERS_DICT)
        df = helper.prepare_dataframe(sort_by="Authors", ascending=True)

        # Should sort by string representation of author lists
        assert len(df) == 3
        assert "paper_id" in df.columns

# --- query_dataframe Tool Tests ---
class TestQueryDataframeTool:
    """Unit tests for query_dataframe tool"""

    def test_query_dataframe_no_papers_in_state(self):
        """Test query_dataframe raises error when no papers in state"""
        state = {"last_displayed_papers": None}

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first."
        ):
            query_dataframe.invoke({
                "question": "List all papers",
                "state": state
            })

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_basic_success(self, mock_create_agent):
        """Test basic query without sorting"""
        state = {
            "last_displayed_papers": "papers",
            "papers": MOCK_PAPERS_DICT,
            "llm_model": MagicMock()
        }

        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Found 3 papers in the dataframe"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke({
            "question": "How many papers are there?",
            "state": state
        })

        assert isinstance(result, str)
        assert result == "Found 3 papers in the dataframe"

        # Verify agent was created with correct parameters
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["allow_dangerous_code"] is True
        assert call_kwargs["agent_type"] == "tool-calling"
        assert isinstance(call_kwargs["df"], pd.DataFrame)

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_with_sorting(self, mock_create_agent):
        """Test query with sorting"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT,
            "llm_model": MagicMock()
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Top paper has 200 citations"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke({
            "question": "What's the top paper by citations?",
            "state": state,
            "sort_by": "Citation Count",
            "ascending": False
        })

        assert result == "Top paper has 200 citations"

        # Check that dataframe was sorted before passing to agent
        call_kwargs = mock_create_agent.call_args[1]
        df = call_kwargs["df"]
        assert df.iloc[0]["paper_id"] == "paper2"  # Highest citations

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_with_limit(self, mock_create_agent):
        """Test query with limit"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT,
            "llm_model": MagicMock()
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Analyzed top 2 papers"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke({
            "question": "Analyze the top papers",
            "state": state,
            "sort_by": "Max H-Index",
            "ascending": False,
            "limit": 2
        })

        assert result == "Analyzed top 2 papers"

        # Check that only 2 papers were in the dataframe
        call_kwargs = mock_create_agent.call_args[1]
        df = call_kwargs["df"]
        assert len(df) == 2

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_direct_dict_mapping(self, mock_create_agent):
        """Test when last_displayed_papers is direct dict instead of key"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT,
            "llm_model": MagicMock()
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Direct mapping works"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke({
            "question": "Test direct mapping",
            "state": state
        })

        assert result == "Direct mapping works"

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_agent_configuration(self, mock_create_agent):
        """Test that agent is configured properly"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT,
            "llm_model": MagicMock()
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Config test"}
        mock_create_agent.return_value = mock_agent

        query_dataframe.invoke({
            "question": "Test config",
            "state": state
        })

        # Check agent configuration
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["max_iterations"] == 5
        assert call_kwargs["include_df_in_prompt"] is True
        assert call_kwargs["verbose"] is True
        assert "prefix" in call_kwargs
        assert "already loaded as 'df'" in call_kwargs["prefix"]

    def test_query_dataframe_empty_papers_dict(self):
        """Test handling of empty papers dictionary"""
        state = {
            "last_displayed_papers": {},
            "llm_model": MagicMock()
        }

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first."
        ):
            query_dataframe.invoke({
                "question": "Query empty dict",
                "state": state
            })

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.logger")
    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_logging(self, mock_create_agent, mock_logger):
        """Test that appropriate logging occurs"""
        state = {
            "last_displayed_papers": MOCK_PAPERS_DICT,
            "llm_model": MagicMock()
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Logging test"}
        mock_create_agent.return_value = mock_agent

        query_dataframe.invoke({
            "question": "Test logging",
            "state": state,
            "sort_by": "Year",
            "limit": 2
        })

        # Check logging calls
        mock_logger.info.assert_any_call(
            "Querying papers with question: %s, sort_by: %s, limit: %s",
            "Test logging", "Year", 2
        )
        mock_logger.info.assert_any_call(
            "Querying over %d papers%s",
            2,
            " (sorted by Year)"
        )

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe.create_pandas_dataframe_agent")
    def test_query_dataframe_with_many_papers(self, mock_create_agent):
        """Test handling of large number of papers"""
        # Create 50 mock papers
        large_papers_dict = {
            f"paper{i}": {
                "Title": f"Paper {i}",
                "Year": 2020 + (i % 5),
                "Citation Count": i * 10,
                "Authors": [f"Author {i}"]
            }
            for i in range(50)
        }

        state = {
            "last_displayed_papers": large_papers_dict,
            "llm_model": MagicMock()
        }

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Handled large dataset"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke({
            "question": "Analyze all papers",
            "state": state
        })

        assert result == "Handled large dataset"

        # Check that number_of_head_rows is limited
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["number_of_head_rows"] == 20  # Should be capped at 20
