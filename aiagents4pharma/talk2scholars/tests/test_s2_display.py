"""Unit tests for the `display_dataframe` tool and its helper."""
# pylint: disable=redefined-outer-name

import logging
import pytest

from langgraph.types import Command
from aiagents4pharma.talk2scholars.tools.s2.display_dataframe import (
    display_dataframe,
    NoPapersFoundError,
)
from aiagents4pharma.talk2scholars.tools.s2.utils.display_dataframe_helper import (
    DisplayDataHelper,
)

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def mock_state():
    """Provides a mock state containing three papers with varying metrics."""
    return {
        "last_displayed_papers": "papers_data",
        "papers_data": {
            "001": {
                "semantic_scholar_paper_id": "001",
                "Title": "AI in Medicine",
                "Citation Count": 120,
                "H-Index": 30,
                "Year": 2021,
            },
            "002": {
                "semantic_scholar_paper_id": "002",
                "Title": "Biotech Advances",
                "Citation Count": 90,
                "H-Index": 15,
                "Year": 2022,
            },
            "003": {
                "semantic_scholar_paper_id": "003",
                "Title": "Genomics Today",
                "Citation Count": "N/A",
                "H-Index": 10,
                "Year": 2020,
            },
        },
    }


def test_display_dataframe_default(mock_state):
    """Test that the tool returns a Command with the correct content."""
    result = display_dataframe.invoke({
        "tool_call_id": "abc123",
        "state": mock_state,
    })
    assert isinstance(result, Command)
    assert "papers found" in result.update["messages"][0].content


def test_sort_by_citation(mock_state):
    """Test sorting by numeric field 'Citation Count' (descending)."""
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by="Citation Count",
        ascending=False,
        limit=None,
    )
    out = helper.process_display()
    papers = list(out["artifact"].values())
    assert papers[0]["Citation Count"] == 120
    assert papers[1]["Citation Count"] == 90
    assert papers[2]["Citation Count"] == "N/A"


def test_sort_by_hindex(mock_state):
    """Test sorting by numeric field 'H-Index' (ascending)."""
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by="H-Index",
        ascending=True,
        limit=None,
    )
    out = helper.process_display()
    papers = list(out["artifact"].values())
    assert papers[0]["H-Index"] == 10


def test_sort_by_string_field(mock_state):
    """Test sorting by string field 'Title' (ascending lex order)."""
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by="Title",
        ascending=True,
        limit=None,
    )
    out = helper.process_display()
    papers = list(out["artifact"].values())
    titles = [p["Title"] for p in papers]
    assert titles == sorted(titles)


def test_sort_by_invalid_field(mock_state, caplog):
    """Test that an invalid sort field emits a warning and preserves order."""
    caplog.set_level(logging.WARNING)
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by="NonexistentField",
        ascending=True,
        limit=None,
    )
    out = helper.process_display()
    papers = list(out["artifact"].values())
    assert len(papers) == len(mock_state["papers_data"])
    assert "Sort field 'NonexistentField' not found" in caplog.text


def test_limit_results(mock_state):
    """Test that the helper limits the number of results when `limit` is given."""
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by=None,
        ascending=False,
        limit=2,
    )
    out = helper.process_display()
    assert len(out["artifact"]) == 2


def test_limit_without_sort(mock_state):
    """Test limiting without any sorting specified."""
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by=None,
        ascending=False,
        limit=1,
    )
    out = helper.process_display()
    assert len(out["artifact"]) == 1


def test_sort_and_limit(mock_state):
    """Test sort by 'Year' ascending and then limit to one result."""
    helper = DisplayDataHelper(
        artifact=mock_state["papers_data"],
        sort_by="Year",
        ascending=True,
        limit=1,
    )
    out = helper.process_display()
    keys = list(out["artifact"].keys())
    assert keys[0] == "003"


def test_missing_papers_raises():
    """Test that the tool raises when no papers are found in state."""
    state = {"last_displayed_papers": "missing_key"}
    with pytest.raises(NoPapersFoundError):
        display_dataframe.invoke({
            "tool_call_id": "abc123",
            "state": state,
        })
