"""Shared fixtures: small Excel files written to a temporary directory."""
import pandas as pd
import pytest


@pytest.fixture
def messy_workbook(tmp_path):
    """A workbook with realistic, non-standard headers and two sheets."""
    path = tmp_path / "plan.xlsx"
    estimates = pd.DataFrame(
        {
            "Work package": ["Discovery", "Backend", "Frontend", None],
            "Track": ["Core", "Core", "UI", None],
            "Best case (d)": [8, 18, 15, None],
            "Expected": [12, 25, 22, None],
            "Worst case (d)": [20, 45, 38, None],
            "Notes": ["", "", "", None],
        }
    )
    notes = pd.DataFrame({"Comment": ["not the sheet you want"]})
    with pd.ExcelWriter(path) as writer:
        estimates.to_excel(writer, sheet_name="Estimates", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    return path


@pytest.fixture
def workbook_with_preamble(tmp_path):
    """A workbook with two title rows above the real header row."""
    path = tmp_path / "preamble.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            [
                ["Project Delivery Plan", None, None, None],
                ["Prepared 20 Aug 2026", None, None, None],
                ["Task", "Optimistic", "Realistic", "Pessimistic"],
                ["Discovery", 8, 12, 20],
                ["Build", 18, 25, 45],
            ]
        ).to_excel(writer, sheet_name="Plan", index=False, header=False)
    return path
