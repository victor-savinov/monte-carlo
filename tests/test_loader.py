"""Tests for reading estimates out of a workbook."""
import pytest

from montecarlo.core.loader import LoaderError, raw_preview, read_sheet, sheet_names


def test_sheet_names_are_listed_in_order(messy_workbook):
    assert sheet_names(messy_workbook) == ["Estimates", "Notes"]


def test_the_first_sheet_is_used_by_default(messy_workbook):
    df = read_sheet(messy_workbook)
    assert "Work package" in df.columns


def test_a_named_sheet_can_be_requested(messy_workbook):
    df = read_sheet(messy_workbook, "Notes")
    assert list(df.columns) == ["Comment"]


def test_fully_blank_rows_are_dropped(messy_workbook):
    df = read_sheet(messy_workbook, "Estimates")
    assert len(df) == 3


def test_fully_blank_columns_are_dropped(messy_workbook):
    df = read_sheet(messy_workbook, "Estimates")
    assert "Notes" not in df.columns


def test_column_names_are_stripped(tmp_path):
    import pandas as pd

    path = tmp_path / "spaced.xlsx"
    pd.DataFrame({"  Task  ": ["a"], " Best ": [1]}).to_excel(path, index=False)
    assert list(read_sheet(path).columns) == ["Task", "Best"]


def test_an_unreadable_file_raises_loader_error(tmp_path):
    path = tmp_path / "not-really.xlsx"
    path.write_bytes(b"this is not a workbook")
    with pytest.raises(LoaderError):
        read_sheet(path)


def test_raw_preview_has_no_header_interpretation(workbook_with_preamble):
    preview = raw_preview(workbook_with_preamble)
    assert list(preview.columns) == ["Column 1", "Column 2", "Column 3", "Column 4"]
    assert preview.iloc[0, 0] == "Project Delivery Plan"
    assert preview.iloc[2, 0] == "Task"


def test_raw_preview_respects_the_row_limit(workbook_with_preamble):
    assert len(raw_preview(workbook_with_preamble, n_rows=2)) == 2


def test_header_row_skips_preamble_rows(workbook_with_preamble):
    df = read_sheet(workbook_with_preamble, header_row=2)
    assert list(df.columns) == ["Task", "Optimistic", "Realistic", "Pessimistic"]
    assert len(df) == 2
    assert df.iloc[0]["Task"] == "Discovery"


def test_header_row_defaults_to_the_first_row(workbook_with_preamble):
    # Without skipping, the preamble text lands in the data, not an error.
    df = read_sheet(workbook_with_preamble)
    assert df.iloc[0, 0] == "Prepared 20 Aug 2026"
