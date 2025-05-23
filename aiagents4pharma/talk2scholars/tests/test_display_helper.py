"""
Unit tests for display_helper utility.
"""

# pylint: disable=redefined-outer-name,too-many-public-methods
import pandas as pd
from ..tools.s2.utils.display_helper import DisplayHelper


# Test data
MOCK_PAPERS = {
    "123": {
        "Title": "Machine Learning Basics",
        "Abstract": "An introduction to ML",
        "Year": 2023,
        "Citation Count": 100,
        "Max H-Index": 45,
        "URL": "https://example.com/paper1",
    },
    "456": {
        "Title": "Deep Learning Advanced",
        "Abstract": "Advanced DL techniques",
        "Year": 2024,
        "Citation Count": 200,
        "Max H-Index": 80,
        "URL": "https://example.com/paper2",
    },
    "789": {
        "Title": "Neural Networks",
        "Abstract": "NN fundamentals",
        "Year": 2022,
        "Citation Count": 50,
        "Max H-Index": "N/A",
        "URL": "https://example.com/paper3",
    },
    "101": {
        "Title": "AI Ethics",
        "Abstract": "Ethical AI considerations",
        "Year": 2024,
        "Citation Count": "N/A",
        "Max H-Index": 30,
        "URL": "https://example.com/paper4",
    }
}


class TestDisplayHelper:
    """Unit tests for DisplayHelper class"""

    def test_init(self):
        """Test initialization of DisplayHelper"""
        helper = DisplayHelper(MOCK_PAPERS)
        assert helper.papers_dict == MOCK_PAPERS
        assert helper.df is None

    def test_prepare_dataframe_no_sorting(self):
        """Test prepare_dataframe without sorting"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert list(df.index) == ["123", "456", "789", "101"]  # Original order

    def test_prepare_dataframe_sort_by_h_index_descending(self):
        """Test sorting by Max H-Index in descending order"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Max H-Index", ascending=False)

        # Check order: 80, 45, 30, N/A
        assert list(df.index) == ["456", "123", "101", "789"]

        # Verify numeric conversion
        assert pd.api.types.is_numeric_dtype(df["Max H-Index"])

    def test_prepare_dataframe_sort_by_h_index_ascending(self):
        """Test sorting by Max H-Index in ascending order"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Max H-Index", ascending=True)

        # Check order: 30, 45, 80, N/A (NaN at end)
        assert list(df.index) == ["101", "123", "456", "789"]

    def test_prepare_dataframe_sort_by_citation_count(self):
        """Test sorting by Citation Count"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Citation Count", ascending=False)

        # Check order: 200, 100, 50, N/A
        assert list(df.index) == ["456", "123", "789", "101"]

    def test_prepare_dataframe_sort_with_limit(self):
        """Test sorting with limit"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Max H-Index", ascending=False, limit=2)

        assert len(df) == 2
        assert list(df.index) == ["456", "123"]  # Top 2 by H-index

    def test_prepare_dataframe_sort_by_non_numeric_column(self):
        """Test sorting by non-numeric column (Title)"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Title", ascending=True)

        # Alphabetical order
        # AI Ethics, Deep Learning, Machine Learning, Neural Networks
        expected_order = ["101", "456", "123", "789"]
        assert list(df.index) == expected_order

    def test_prepare_dataframe_invalid_column(self):
        """Test sorting by non-existent column"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Invalid Column", ascending=False)

        # Should return original DataFrame when column doesn't exist
        assert list(df.index) == ["123", "456", "789", "101"]

    def test_prepare_dataframe_limit_exceeds_data(self):
        """Test limit larger than available data"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Year", ascending=False, limit=10)

        # Should return all 4 papers
        assert len(df) == 4

    def test_prepare_dataframe_zero_limit(self):
        """Test with limit=0"""
        helper = DisplayHelper(MOCK_PAPERS)
        df = helper.prepare_dataframe(sort_by="Year", ascending=False, limit=0)

        # Should return all papers when limit is 0
        assert len(df) == 4

    def test_get_sorted_dict_without_prepare(self):
        """Test get_sorted_dict when prepare_dataframe hasn't been called"""
        helper = DisplayHelper(MOCK_PAPERS)
        sorted_dict = helper.get_sorted_dict()

        # Should return original papers
        assert sorted_dict == MOCK_PAPERS

    def test_get_sorted_dict_after_sorting(self):
        """Test get_sorted_dict after sorting"""
        helper = DisplayHelper(MOCK_PAPERS)
        helper.prepare_dataframe(sort_by="Citation Count", ascending=False, limit=3)
        sorted_dict = helper.get_sorted_dict()

        # Should maintain sorted order
        keys = list(sorted_dict.keys())
        assert keys == ["456", "123", "789"]
        assert len(sorted_dict) == 3

    def test_format_summary_no_sorting(self):
        """Test summary formatting without sorting"""
        helper = DisplayHelper(MOCK_PAPERS)
        helper.prepare_dataframe()  # No sorting
        summary = helper.format_summary()

        assert summary == "4 papers found. Papers are attached as an artifact."

    def test_format_summary_with_sorting(self):
        """Test summary formatting with sorting"""
        helper = DisplayHelper(MOCK_PAPERS)
        helper.prepare_dataframe(sort_by="Max H-Index", ascending=False)
        summary = helper.format_summary(sort_by="Max H-Index")

        assert "4 papers found." in summary
        assert "Sorted by Max H-Index (descending)." in summary
        assert "Papers are attached as an artifact." in summary

    def test_format_summary_with_sorting_and_limit(self):
        """Test summary formatting with sorting and limit"""
        helper = DisplayHelper(MOCK_PAPERS)
        helper.prepare_dataframe(sort_by="Citation Count", ascending=False, limit=2)
        summary = helper.format_summary(sort_by="Citation Count", limit=2)

        assert "2 papers found." in summary
        assert "Sorted by Citation Count (descending)." in summary
        assert "Showing top 2 results." in summary
        assert "Papers are attached as an artifact." in summary

    def test_format_summary_limit_equals_total(self):
        """Test summary when limit equals total papers"""
        helper = DisplayHelper(MOCK_PAPERS)
        helper.prepare_dataframe(sort_by="Year", ascending=True, limit=4)
        summary = helper.format_summary(sort_by="Year", limit=4)

        # Should not mention "top N" when showing all papers
        assert "4 papers found." in summary
        assert "Sorted by Year" in summary
        assert "Showing top" not in summary  # No need to mention limit when showing all

    def test_empty_papers_dict(self):
        """Test with empty papers dictionary"""
        helper = DisplayHelper({})
        df = helper.prepare_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

        sorted_dict = helper.get_sorted_dict()
        assert sorted_dict == {}

        summary = helper.format_summary()
        assert summary == "0 papers found. Papers are attached as an artifact."

    def test_single_paper(self):
        """Test with single paper"""
        single_paper = {"123": MOCK_PAPERS["123"]}
        helper = DisplayHelper(single_paper)

        df = helper.prepare_dataframe(sort_by="Max H-Index")
        assert len(df) == 1
        assert list(df.index) == ["123"]

    def test_all_na_values_in_sort_column(self):
        """Test sorting when all values are N/A"""
        papers_all_na = {
            "1": {"Title": "Paper 1", "Max H-Index": "N/A", "Citation Count": 10},
            "2": {"Title": "Paper 2", "Max H-Index": "N/A", "Citation Count": 20},
            "3": {"Title": "Paper 3", "Max H-Index": "N/A", "Citation Count": 30},
        }

        helper = DisplayHelper(papers_all_na)
        df = helper.prepare_dataframe(sort_by="Max H-Index", ascending=False)

        # Original order should be preserved when all are N/A
        assert list(df.index) == ["1", "2", "3"]

    def test_mixed_numeric_and_na_values(self):
        """Test proper handling of mixed numeric and N/A values"""
        mixed_papers = {
            "1": {"Title": "Paper 1", "Citation Count": "N/A"},
            "2": {"Title": "Paper 2", "Citation Count": 100},
            "3": {"Title": "Paper 3", "Citation Count": "N/A"},
            "4": {"Title": "Paper 4", "Citation Count": 50},
        }

        helper = DisplayHelper(mixed_papers)
        df = helper.prepare_dataframe(sort_by="Citation Count", ascending=False)

        # Should be: 100, 50, N/A, N/A
        assert list(df.index) == ["2", "4", "1", "3"]

    def test_year_sorting_with_strings(self):
        """Test Year column sorting with mixed types"""
        year_papers = {
            "1": {"Title": "Paper 1", "Year": "2023"},
            "2": {"Title": "Paper 2", "Year": 2024},
            "3": {"Title": "Paper 3", "Year": "N/A"},
            "4": {"Title": "Paper 4", "Year": 2022},
        }

        helper = DisplayHelper(year_papers)
        df = helper.prepare_dataframe(sort_by="Year", ascending=True)

        # Should be: 2022, 2023, 2024, N/A
        assert list(df.index) == ["4", "1", "2", "3"]

    def test_chained_operations(self):
        """Test multiple operations on same helper"""
        helper = DisplayHelper(MOCK_PAPERS)

        # First operation: sort by H-index
        helper.prepare_dataframe(sort_by="Max H-Index", ascending=False, limit=3)
        dict1 = helper.get_sorted_dict()
        assert len(dict1) == 3

        # Second operation: sort by Citation Count
        helper.prepare_dataframe(sort_by="Citation Count", ascending=True)
        dict2 = helper.get_sorted_dict()
        assert len(dict2) == 4  # All papers, different order

        # Verify different results
        assert list(dict1.keys()) != list(dict2.keys())[:3]
