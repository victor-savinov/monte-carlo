"""Tests for input validation and array extraction."""
import numpy as np
import pandas as pd
import pytest

from montecarlo.core.validate import Issue, is_total_row, prepare, strip_total_rows, validate

MAPPING = {
    "task": "Task",
    "optimistic": "Best",
    "realistic": "Expected",
    "pessimistic": "Worst",
    "stream": None,
}


def frame(rows):
    return pd.DataFrame(rows, columns=["Task", "Best", "Expected", "Worst"])


def severities(issues):
    return [issue.severity for issue in issues]


def test_a_clean_table_produces_no_issues():
    df = frame([["Discovery", 8, 12, 20], ["Build", 18, 25, 45]])
    assert validate(df, MAPPING) == []


def test_a_missing_required_role_is_an_error():
    df = frame([["Discovery", 8, 12, 20]])
    broken = dict(MAPPING, realistic=None)
    issues = validate(df, broken)
    assert "error" in severities(issues)
    assert any("realistic" in issue.message.lower() for issue in issues)


def test_text_in_a_number_column_is_an_error_naming_the_row():
    df = frame([["Discovery", 8, 12, 20], ["Build", "soon", 25, 45]])
    issues = validate(df, MAPPING)
    assert any(i.severity == "error" and i.row == 2 and i.column == "Best"
               for i in issues)


def test_a_negative_duration_is_an_error():
    df = frame([["Discovery", -1, 12, 20]])
    assert "error" in severities(validate(df, MAPPING))


def test_an_empty_duration_cell_is_an_error():
    df = frame([["Discovery", None, 12, 20]])
    assert "error" in severities(validate(df, MAPPING))


def test_optimistic_above_pessimistic_is_a_warning_not_an_error():
    df = frame([["Discovery", 30, 12, 20]])
    issues = validate(df, MAPPING)
    assert "warning" in severities(issues)
    assert "error" not in severities(issues)


def test_zero_uncertainty_is_a_warning():
    df = frame([["Discovery", 10, 10, 10]])
    assert severities(validate(df, MAPPING)) == ["warning"]


def test_a_blank_task_name_is_a_warning():
    df = frame([[None, 8, 12, 20]])
    assert "warning" in severities(validate(df, MAPPING))


def test_duplicate_task_names_are_a_warning():
    df = frame([["Build", 8, 12, 20], ["Build", 5, 9, 14]])
    assert "warning" in severities(validate(df, MAPPING))


def test_an_empty_table_is_an_error():
    assert "error" in severities(validate(frame([]), MAPPING))


def test_prepare_returns_aligned_arrays():
    df = frame([["Discovery", 8, 12, 20], ["Build", 18, 25, 45]])
    result = prepare(df, MAPPING)
    assert result.names == ["Discovery", "Build"]
    assert np.array_equal(result.o, np.array([8.0, 18.0]))
    assert np.array_equal(result.p, np.array([20.0, 45.0]))
    assert result.streams is None


def test_prepare_reads_the_stream_column_when_mapped():
    df = pd.DataFrame(
        [["Discovery", "Core", 8, 12, 20], ["UI", "Front", 5, 9, 14]],
        columns=["Task", "Track", "Best", "Expected", "Worst"],
    )
    result = prepare(df, dict(MAPPING, stream="Track"))
    assert result.streams == ["Core", "Front"]


def test_prepare_can_sort_out_of_order_estimates():
    df = frame([["Discovery", 30, 12, 20]])
    result = prepare(df, MAPPING, sort_three_point=True)
    assert (result.o[0], result.m[0], result.p[0]) == (12.0, 20.0, 30.0)


def test_prepare_leaves_estimates_alone_by_default():
    df = frame([["Discovery", 30, 12, 20]])
    result = prepare(df, MAPPING)
    assert result.o[0] == 30.0


def test_prepare_labels_a_blank_task_name_by_row():
    df = frame([[None, 8, 12, 20]])
    assert prepare(df, MAPPING).names == ["Row 1"]


def test_prepare_fills_a_blank_stream_label():
    df = pd.DataFrame(
        [["Discovery", None, 8, 12, 20]],
        columns=["Task", "Track", "Best", "Expected", "Worst"],
    )
    assert prepare(df, dict(MAPPING, stream="Track")).streams == ["(no stream)"]


def test_total_row_is_recognised_in_english_and_russian():
    for name in ("Total", "TOTAL", "Grand Total", "Итого", "Всего", "Сумма"):
        assert is_total_row(name), name


def test_a_real_task_name_is_not_a_total_row():
    for name in ("Total redesign", "Sum of parts", "Discovery"):
        assert not is_total_row(name), name


def test_is_total_row_handles_blank_and_missing_names():
    assert not is_total_row(None)
    assert not is_total_row(float("nan"))
    assert not is_total_row("")


def test_strip_total_rows_removes_the_matching_rows_only():
    df = frame([["Discovery", 8, 12, 20], ["Total", 0, 0, 0], ["Build", 18, 25, 45]])
    cleaned, dropped = strip_total_rows(df, MAPPING)
    assert list(cleaned["Task"]) == ["Discovery", "Build"]
    assert dropped == ["Total"]


def test_strip_total_rows_is_a_no_op_without_a_task_column():
    df = frame([["Discovery", 8, 12, 20]])
    cleaned, dropped = strip_total_rows(df, dict(MAPPING, task=None))
    assert len(cleaned) == 1
    assert dropped == []


def test_strip_total_rows_leaves_a_clean_table_untouched():
    df = frame([["Discovery", 8, 12, 20], ["Build", 18, 25, 45]])
    cleaned, dropped = strip_total_rows(df, MAPPING)
    assert len(cleaned) == 2
    assert dropped == []


def test_strip_total_rows_catches_an_unlabelled_sum_row():
    """A blank-named row whose numbers equal the column sums above it is a
    spreadsheet SUM() total, even with no 'Total' label at all."""
    df = frame([
        ["Discovery", 8, 12, 20],
        ["Build", 18, 25, 45],
        [None, 26, 37, 65],  # 8+18=26, 12+25=37, 20+45=65
    ])
    cleaned, dropped = strip_total_rows(df, MAPPING)
    assert list(cleaned["Task"]) == ["Discovery", "Build"]
    assert dropped == ["Row 3"]


def test_strip_total_rows_keeps_a_blank_named_task_that_is_not_a_sum():
    df = frame([["Discovery", 8, 12, 20], [None, 5, 9, 14]])
    cleaned, dropped = strip_total_rows(df, MAPPING)
    assert len(cleaned) == 2
    assert dropped == []
