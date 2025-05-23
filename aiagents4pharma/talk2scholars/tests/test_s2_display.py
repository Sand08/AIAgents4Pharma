"""
Unit tests for S2 display_dataframe tool functionality.
"""

# pylint: disable=redefined-outer-name
import pytest
from langgraph.types import Command
from ..tools.s2.display_dataframe import (
    display_dataframe,
    NoPapersFoundError,
)


@pytest.fixture
def initial_state():
    """Provides an empty initial state for tests."""
    return {"papers": {}, "multi_papers": {}}


# Fixed test data for deterministic results
MOCK_STATE_PAPERS = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "Max H-Index": 45,
        "URL": "https://example.com/paper1",
        "Authors": ["Test Author 1"],
    },
    "456": {
        "Title": "Deep Learning Advanced",
        "Abstract": "Advanced DL techniques",
        "Year": 2024,
        "Citation Count": 200,
        "Max H-Index": 80,
        "URL": "https://example.com/paper2",
        "Authors": ["Test Author 2"],
    },
    "789": {
        "Title": "Neural Networks",
        "Abstract": "NN fundamentals",
        "Year": 2022,
        "Citation Count": 50,
        "Max H-Index": "N/A",
        "URL": "https://example.com/paper3",
        "Authors": ["Test Author 3"],
    },
    "101": {
        "Title": "AI Ethics",
        "Abstract": "Ethical AI considerations",
        "Year": 2024,
        "Citation Count": "N/A",
        "Max H-Index": 30,
        "URL": "https://example.com/paper4",
        "Authors": ["Test Author 4"],
    }
}


class TestDisplayDataframe:
    """Unit tests for display_dataframe tool"""

    def test_display_dataframe_empty_state(self, initial_state):
        """Verifies display_dataframe raises error when state is empty"""
        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search/rec needs to be performed first.",
        ):
            display_dataframe.invoke(
                {"state": initial_state, "tool_call_id": "test123"}
            )

    def test_display_dataframe_no_last_displayed_papers(self, initial_state):
        """Verifies error when last_displayed_papers key is missing"""
        state = initial_state.copy()
        state["papers"] = MOCK_STATE_PAPERS
        # Don't set last_displayed_papers

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search/rec needs to be performed first.",
        ):
            display_dataframe.invoke(
                {"state": state, "tool_call_id": "test123"}
            )

    def test_display_dataframe_default_no_sorting(self, initial_state):
        """Verifies default behavior without sorting"""
        state = initial_state.copy()
        state["last_displayed_papers"] = "papers"
        state["papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke(
            {"state": state, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
        assert isinstance(result.update, dict)
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1

        message = result.update["messages"][0]
        assert "4 papers found. Papers are attached as an artifact." in message.content
        assert message.artifact == MOCK_STATE_PAPERS  # Original order preserved

        # Verify state is updated
        assert "last_displayed_papers" in result.update
        assert result.update["last_displayed_papers"] == MOCK_STATE_PAPERS

    def test_display_dataframe_direct_mapping(self, initial_state):
        """Verifies handling of direct dict mapping in last_displayed_papers"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke(
            {"state": state, "tool_call_id": "test123"}
        )

        assert isinstance(result, Command)
        messages = result.update.get("messages", [])
        assert len(messages) == 1
        assert messages[0].artifact == MOCK_STATE_PAPERS
        assert "4 papers found" in messages[0].content

    def test_display_dataframe_sort_by_h_index(self, initial_state):
        """Verifies sorting by Max H-Index in descending order"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Max H-Index",
            "ascending": False
        })

        assert isinstance(result, Command)
        message = result.update["messages"][0]

        # Check content mentions sorting
        assert "Sorted by Max H-Index (descending)" in message.content

        # Verify sorting order (80, 45, 30, N/A)
        artifact_keys = list(message.artifact.keys())
        assert artifact_keys == ["456", "123", "101", "789"]

        # Verify state is updated with sorted papers
        assert result.update["last_displayed_papers"] == message.artifact

    def test_display_dataframe_sort_by_citation_count(self, initial_state):
        """Verifies sorting by Citation Count"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Citation Count",
            "ascending": True  # Test ascending order
        })

        message = result.update["messages"][0]

        # Check content
        assert "Sorted by Citation Count" in message.content

        # Verify sorting order (50, 100, 200, N/A)
        artifact_keys = list(message.artifact.keys())
        assert artifact_keys == ["789", "123", "456", "101"]

    def test_display_dataframe_sort_with_limit(self, initial_state):
        """Verifies sorting with limit parameter"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Max H-Index",
            "ascending": False,
            "limit": 2
        })

        message = result.update["messages"][0]

        # Check content mentions limit
        assert "Showing top 2 results" in message.content

        # Verify only 2 papers returned
        assert len(message.artifact) == 2
        artifact_keys = list(message.artifact.keys())
        assert artifact_keys == ["456", "123"]  # Top 2 by H-index

    def test_display_dataframe_sort_by_year(self, initial_state):
        """Verifies sorting by Year"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Year",
            "ascending": False
        })

        message = result.update["messages"][0]

        # Verify sorting order (2024, 2024, 2023, 2022)
        artifact_keys = list(message.artifact.keys())
        # Note: When years are equal, original order is preserved
        assert artifact_keys[0] in ["456", "101"]  # Both are 2024
        assert artifact_keys[1] in ["456", "101"]  # Both are 2024
        assert artifact_keys[2] == "123"  # 2023
        assert artifact_keys[3] == "789"  # 2022

    def test_display_dataframe_invalid_sort_column(self, initial_state):
        """Verifies behavior with invalid sort column"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test123",
            "sort_by": "Invalid Column",
            "ascending": False
        })

        # Should return original order when column doesn't exist
        message = result.update["messages"][0]
        assert message.artifact == MOCK_STATE_PAPERS

    def test_display_dataframe_empty_papers_dict(self, initial_state):
        """Verifies handling of empty papers dictionary"""
        state = initial_state.copy()
        state["last_displayed_papers"] = {}

        with pytest.raises(
            NoPapersFoundError,
            match="No papers found. A search/rec needs to be performed first.",
        ):
            display_dataframe.invoke(
                {"state": state, "tool_call_id": "test123"}
            )

    def test_display_dataframe_all_parameters(self, initial_state):
        """Verifies tool with all parameters specified"""
        state = initial_state.copy()
        state["last_displayed_papers"] = MOCK_STATE_PAPERS

        # Test with all parameters
        result = display_dataframe.invoke({
            "state": state,
            "tool_call_id": "test_all_params",
            "sort_by": "Citation Count",
            "ascending": False,
            "limit": 3
        })

        assert isinstance(result, Command)
        message = result.update["messages"][0]

        # Verify all aspects
        assert "Sorted by Citation Count (descending)" in message.content
        assert "Showing top 3 results" in message.content
        assert len(message.artifact) == 3

        # Verify correct papers and order (200, 100, 50)
        artifact_keys = list(message.artifact.keys())
        assert artifact_keys == ["456", "123", "789"]
