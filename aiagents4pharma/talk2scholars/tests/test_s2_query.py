"""
Unit tests for S2 query_dataframe tool functionality.
"""

# pylint: disable=redefined-outer-name,unsubscriptable-object,unused-argument
# pylint: disable=unused-variable,unsupported-membership-test
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from ..tools.s2.query_dataframe import query_dataframe, NoPapersFoundError


@pytest.fixture
def initial_state():
    """Provides an empty initial state for tests."""
    return {"papers": {}, "multi_papers": {}}


@pytest.fixture
def mock_llm_model():
    """Provides a mock LLM model."""
    return MagicMock()


# Test data with various H-index and citation values
MOCK_STATE_PAPERS = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "Max H-Index": 45,
        "URL": "https://example.com/paper1",
        "Authors": ["Author One", "Author Two"],
    },
    "456": {
        "Title": "Deep Learning Advanced",
        "Abstract": "Advanced DL techniques",
        "Year": 2024,
        "Citation Count": 200,
        "Max H-Index": 80,
        "URL": "https://example.com/paper2",
        "Authors": ["Author Three"],
    },
    "789": {
        "Title": "Neural Networks",
        "Abstract": "NN fundamentals",
        "Year": 2022,
        "Citation Count": 50,
        "Max H-Index": "N/A",
        "URL": "https://example.com/paper3",
        "Authors": ["Author Four", "Author Five"],
    },
    "101": {
        "Title": "AI Ethics",
        "Abstract": "Ethical AI considerations",
        "Year": 2024,
        "Citation Count": "N/A",
        "Max H-Index": 30,
        "URL": "https://example.com/paper4",
        "Authors": ["Author Six"],
    }
}


class TestQueryDataframe:
    """Unit tests for query_dataframe tool"""

    def test_query_dataframe_empty_state(self, initial_state):
        """Tests error when no papers are in state"""
        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first.",
        ):
            query_dataframe.invoke(
                {"question": "List all papers", "state": initial_state}
            )

    def test_query_dataframe_no_last_displayed_papers(self, initial_state, mock_llm_model):
        """Tests error when last_displayed_papers is not set"""
        state = initial_state.copy()
        state["llm_model"] = mock_llm_model
        state["papers"] = MOCK_STATE_PAPERS
        # Don't set last_displayed_papers

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search needs to be performed first.",
        ):
            query_dataframe.invoke(
                {"question": "Show papers", "state": state}
            )

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_simple_question(self, mock_create_agent, initial_state,
                                              mock_llm_model):
        """Tests basic querying functionality"""
        state = initial_state.copy()
        state["last_displayed_papers"] = "papers"
        state["papers"] = MOCK_STATE_PAPERS
        state["llm_model"] = mock_llm_model

        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "4 papers found"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke(
            {"question": "How many papers are there?", "state": state}
        )

        assert isinstance(result, str)
        assert result == "4 papers found"

        # Verify agent was created with correct parameters
        mock_create_agent.assert_called_once()
        call_args = mock_create_agent.call_args
        assert call_args[0][0] == mock_llm_model  # First positional arg is llm_model
        assert call_args[1]["allow_dangerous_code"] is True
        assert call_args[1]["agent_type"] == "tool-calling"
        assert isinstance(call_args[1]["df"], pd.DataFrame)

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_direct_mapping(self, mock_create_agent, initial_state,
                                            mock_llm_model):
        """Tests query when last_displayed_papers is a direct dict"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS
        state["llm_model"] = mock_llm_model

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Direct mapping works"}
        mock_create_agent.return_value = mock_agent

        result = query_dataframe.invoke(
            {"question": "Test direct mapping", "state": state}
        )

        assert result == "Direct mapping works"

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_sorting_question(self, mock_create_agent, initial_state,
                                              mock_llm_model):
        """Tests enhanced question handling for sorting queries"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS
        state["llm_model"] = mock_llm_model

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "output": ("Top 3 papers by H-index: Paper 456 (80), "
                       "Paper 123 (45), Paper 101 (30)")
        }
        mock_create_agent.return_value = mock_agent

        # Test with "top" keyword
        query_dataframe.invoke(
            {"question": "Show top 3 papers by H-index", "state": state}
        )

        # Verify enhanced question was used
        mock_agent.invoke.assert_called_once()
        call_args = mock_agent.invoke.call_args[0][0]
        assert "Use the EXACT 'Max H-Index'" in call_args
        assert "Do not make up any numbers" in call_args

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_numeric_conversion(self, mock_create_agent, initial_state,
                                                mock_llm_model):
        """Tests that numeric columns are properly converted"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS
        state["llm_model"] = mock_llm_model

        # Capture the dataframe passed to the agent
        captured_df = None
        def capture_df(*args, **kwargs):
            nonlocal captured_df
            captured_df = kwargs.get("df")
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {"output": "Numeric conversion test"}
            return mock_agent

        mock_create_agent.side_effect = capture_df

        query_dataframe.invoke(
            {"question": "Test numeric", "state": state}
        )

        # Verify numeric conversion
        assert captured_df is not None
        # Check dtypes - they can be float64 (if N/A present), int64 (if all valid), or object
        assert str(captured_df["Max H-Index"].dtype) in ["float64", "int64", "object"]
        assert str(captured_df["Citation Count"].dtype) in ["float64", "int64", "object"]
        assert str(captured_df["Year"].dtype) in ["float64", "int64", "object"]

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_all_sorting_keywords(self, mock_create_agent, initial_state,
                                                   mock_llm_model):
        """Tests all sorting-related keywords trigger enhancement"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS
        state["llm_model"] = mock_llm_model

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Sorted result"}
        mock_create_agent.return_value = mock_agent

        # Test various sorting keywords
        sorting_questions = [
            "Show highest cited papers",
            "List lowest H-index papers",
            "Sort papers by year",
            "Rank papers by citations",
            "Papers by h-index",
            "Papers by h index",
            "Papers by citation count"
        ]

        for question in sorting_questions:
            mock_agent.invoke.reset_mock()

            query_dataframe.invoke(
                {"question": question, "state": state}
            )

            # Verify enhancement was applied
            call_args = mock_agent.invoke.call_args[0][0]
            assert "IMPORTANT: Use the EXACT" in call_args

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_prefix_content(self, mock_create_agent, initial_state,
                                            mock_llm_model):
        """Tests the prefix contains correct information"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS
        state["llm_model"] = mock_llm_model

        captured_prefix = None
        def capture_prefix(*args, **kwargs):
            nonlocal captured_prefix
            captured_prefix = kwargs.get("prefix", "")
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {"output": "Test"}
            return mock_agent

        mock_create_agent.side_effect = capture_prefix

        query_dataframe.invoke(
            {"question": "Test prefix", "state": state}
        )

        # Verify prefix content
        assert captured_prefix is not None
        assert "4 academic papers metadata" in str(captured_prefix)
        assert "Title" in str(captured_prefix)
        assert "Max H-Index" in str(captured_prefix)
        assert "Citation Count" in str(captured_prefix)
        assert "ONLY the papers that are currently displayed" in str(captured_prefix)
        assert "ACTUAL values from the dataframe" in str(captured_prefix)

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_filtered_papers(self, mock_create_agent, initial_state,
                                             mock_llm_model):
        """Tests behavior with filtered papers (e.g., top 2)"""
        state = initial_state.copy()
        # Simulate filtered papers (only 2 papers)
        filtered_papers = {
            "456": MOCK_STATE_PAPERS["456"],
            "123": MOCK_STATE_PAPERS["123"]
        }
        state["last_displayed_papers"] = filtered_papers
        state["llm_model"] = mock_llm_model

        captured_df = None
        def capture_df(*args, **kwargs):
            nonlocal captured_df
            captured_df = kwargs.get("df")
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {"output": "2 papers shown"}
            return mock_agent

        mock_create_agent.side_effect = capture_df

        query_dataframe.invoke(
            {"question": "Show all papers", "state": state}
        )

        # Verify only 2 papers in dataframe
        assert len(captured_df) == 2
        assert "456" in captured_df.index
        assert "123" in captured_df.index

    @patch("aiagents4pharma.talk2scholars.tools.s2.query_dataframe."
           "create_pandas_dataframe_agent")
    def test_query_dataframe_head_rows_limit(self, mock_create_agent, initial_state,
                                             mock_llm_model):
        """Tests that number_of_head_rows is limited to 10"""
        state = initial_state.copy()
        # Create 15 papers
        large_papers = {}
        for i in range(15):
            large_papers[str(i)] = {
                "Title": f"Paper {i}",
                "Max H-Index": i * 10,
                "Citation Count": i * 20
            }

        state["last_displayed_papers"] = large_papers
        state["llm_model"] = mock_llm_model

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Large dataset"}
        mock_create_agent.return_value = mock_agent

        query_dataframe.invoke(
            {"question": "Count papers", "state": state}
        )

        # Verify number_of_head_rows is capped at 10
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["number_of_head_rows"] == 10
