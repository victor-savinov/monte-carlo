"""Tests for the Excel summary export."""
import io

import numpy as np
import pandas as pd

from montecarlo.core.export import summary_workbook
from montecarlo.core.validate import Prepared

PREPARED = Prepared(
    names=["Discovery", "Build"],
    o=np.array([8.0, 18.0]),
    m=np.array([12.0, 25.0]),
    p=np.array([20.0, 45.0]),
    streams=None,
)
SETTINGS = {"unit": "days", "days_per_week": 5, "rho": 0.3,
            "iterations": 10000, "seed": 20260820, "start": "01 Jun 2026"}


def workbook():
    return summary_workbook(
        PREPARED,
        pctls={50: 175.0, 85: 201.0, 95: 216.0},
        date_labels={50: "15 Jan 2027", 85: "22 Feb 2027", 95: "15 Mar 2027"},
        settings=SETTINGS,
        baseline=149.0,
        baseline_probability=12.1,
    )


def test_export_returns_a_readable_workbook():
    sheets = pd.read_excel(io.BytesIO(workbook()), sheet_name=None)
    assert set(sheets) == {"Result", "Settings", "Tasks"}


def test_the_result_sheet_carries_every_percentile_and_the_plan():
    sheets = pd.read_excel(io.BytesIO(workbook()), sheet_name=None)
    values = sheets["Result"].astype(str).to_numpy().ravel().tolist()
    joined = " ".join(values)
    for expected in ("175", "201", "216", "149", "22 Feb 2027"):
        assert expected in joined


def test_the_tasks_sheet_lists_every_task():
    sheets = pd.read_excel(io.BytesIO(workbook()), sheet_name=None)
    assert list(sheets["Tasks"]["Task"]) == ["Discovery", "Build"]


def test_the_settings_sheet_records_the_seed():
    sheets = pd.read_excel(io.BytesIO(workbook()), sheet_name=None)
    assert "20260820" in " ".join(sheets["Settings"].astype(str)
                                  .to_numpy().ravel().tolist())
