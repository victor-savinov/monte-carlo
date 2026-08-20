"""Checking the input table and turning it into arrays.

Nothing here raises on user data. Problems come back as a list so the screen
can show all of them at once, rather than one exception at a time.
"""
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from montecarlo.core.mapping import REQUIRED_ROLES

ERROR = "error"
WARNING = "warning"
DURATION_ROLES = ("optimistic", "realistic", "pessimistic")
BLANK_STREAM = "(no stream)"

# A task name matching one of these (after normalising) is a spreadsheet
# total row, not a task, and gets dropped before validation ever sees it.
TOTAL_KEYWORDS = (
    "total", "totals", "grand total", "sum", "sum total", "subtotal",
    "overall total",
    "итого", "итог", "всего", "сумма", "общий итог", "итого по проекту",
)


def is_total_row(name: Any) -> bool:
    """Whether a task name looks like a spreadsheet-computed total row."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return False
    text = str(name).strip().lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text in TOTAL_KEYWORDS


def _is_blank_name(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def strip_total_rows(
    df: pd.DataFrame, mapping: Dict[str, Optional[str]]
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop rows that are spreadsheet-computed totals, not real tasks.

    Two independent signals catch this:

      - the task name itself reads like a total ("Total", "Итого", ...);
      - the task name is blank and every duration column's value equals
        the running sum of that column's rows above it — the classic
        unlabelled totals row a spreadsheet SUM() formula produces.

    Args:
        df: the sheet as read.
        mapping: role to column name. Only "task" is needed for the
            name-based check; the duration roles enable the sum-based one.

    Returns:
        The filtered table, and a label for each row that was dropped.
    """
    task_column = mapping.get("task")
    if not task_column or task_column not in df.columns:
        return df, []

    named_mask = df[task_column].apply(is_total_row)

    duration_columns = [
        mapping.get(role) for role in DURATION_ROLES
        if mapping.get(role) and mapping.get(role) in df.columns
    ]
    sum_mask = pd.Series(False, index=df.index)
    if duration_columns:
        blank_name = df[task_column].apply(_is_blank_name)
        candidate = blank_name.copy()
        for column in duration_columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            running_sum = numeric.cumsum().shift(fill_value=0.0)
            matches = np.isclose(numeric, running_sum, atol=1e-6, equal_nan=False)
            candidate &= pd.Series(matches, index=df.index)
        if duration_columns:
            first_running_sum = pd.to_numeric(df[duration_columns[0]], errors="coerce") \
                .cumsum().shift(fill_value=0.0)
            candidate &= first_running_sum > 0
        sum_mask = candidate

    mask = named_mask | sum_mask
    if not mask.any():
        return df, []

    dropped = [
        str(name) if not _is_blank_name(name) else "Row {0}".format(position + 1)
        for position, name in enumerate(df[task_column]) if mask.iloc[position]
    ]
    return df.loc[~mask].reset_index(drop=True), dropped


class Issue(NamedTuple):
    """One problem found in the input, addressed to the person who wrote it."""

    severity: str
    row: Optional[int]
    column: Optional[str]
    message: str


class Prepared(NamedTuple):
    """The input reduced to what the simulation needs."""

    names: List[str]
    o: np.ndarray
    m: np.ndarray
    p: np.ndarray
    streams: Optional[List[str]]


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def validate(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> List[Issue]:
    """Return every problem in the table, worst first is not guaranteed.

    Errors block the simulation. Warnings do not.

    Args:
        df: the sheet as read.
        mapping: role to column name; unmapped roles hold None.
    """
    issues: List[Issue] = []

    for role in REQUIRED_ROLES:
        if not mapping.get(role):
            issues.append(
                Issue(ERROR, None, None,
                      "No column is mapped to '{0}'.".format(role))
            )
    if issues:
        return issues

    if len(df) == 0:
        return [Issue(ERROR, None, None, "The sheet has no rows.")]

    numeric = {role: _numeric(df, mapping[role]) for role in DURATION_ROLES}

    for role in DURATION_ROLES:
        column = mapping[role]
        values = numeric[role]
        for position, value in enumerate(values):
            row = position + 1
            if pd.isna(value):
                issues.append(
                    Issue(ERROR, row, column,
                          "'{0}' is empty or not a number.".format(column))
                )
            elif value < 0:
                issues.append(
                    Issue(ERROR, row, column,
                          "'{0}' is negative.".format(column))
                )

    o, m, p = numeric["optimistic"], numeric["realistic"], numeric["pessimistic"]
    for position in range(len(df)):
        row = position + 1
        a, b, c = o.iloc[position], m.iloc[position], p.iloc[position]
        if pd.isna(a) or pd.isna(b) or pd.isna(c):
            continue
        if a > b or b > c:
            issues.append(
                Issue(WARNING, row, None,
                      "Estimates are out of order (optimistic {0}, realistic "
                      "{1}, pessimistic {2}).".format(a, b, c))
            )
        elif a == c:
            issues.append(
                Issue(WARNING, row, None,
                      "This task has no uncertainty; all three estimates "
                      "are {0}.".format(a))
            )

    names = df[mapping["task"]]
    for position, name in enumerate(names):
        if pd.isna(name) or str(name).strip() == "":
            issues.append(
                Issue(WARNING, position + 1, mapping["task"],
                      "The task has no name; it will be labelled by row.")
            )

    filled = names.dropna().astype(str).str.strip()
    for duplicate in filled[filled.duplicated()].unique():
        issues.append(
            Issue(WARNING, None, mapping["task"],
                  "'{0}' appears more than once.".format(duplicate))
        )

    return issues


def prepare(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    sort_three_point: bool = False,
) -> Prepared:
    """Extract the arrays the simulation needs.

    Call this only after ``validate`` returns no errors.

    Args:
        df: the sheet as read.
        mapping: role to column name.
        sort_three_point: when True, sort each row's three estimates into
            ascending order, repairing rows flagged as out of order.
    """
    o = _numeric(df, mapping["optimistic"]).to_numpy(dtype=float)
    m = _numeric(df, mapping["realistic"]).to_numpy(dtype=float)
    p = _numeric(df, mapping["pessimistic"]).to_numpy(dtype=float)

    if sort_three_point:
        stacked = np.sort(np.column_stack([o, m, p]), axis=1)
        o, m, p = stacked[:, 0], stacked[:, 1], stacked[:, 2]

    names = []
    for position, value in enumerate(df[mapping["task"]]):
        text = "" if pd.isna(value) else str(value).strip()
        names.append(text if text else "Row {0}".format(position + 1))

    streams = None
    if mapping.get("stream"):
        streams = [
            BLANK_STREAM if pd.isna(v) or str(v).strip() == "" else str(v).strip()
            for v in df[mapping["stream"]]
        ]

    return Prepared(names=names, o=o, m=m, p=p, streams=streams)
