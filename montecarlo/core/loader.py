"""Reading an estimate table out of an Excel workbook."""
from typing import Any, List, Optional

import pandas as pd


class LoaderError(Exception):
    """The file could not be opened as a workbook."""


def sheet_names(source: Any) -> List[str]:
    """List the sheets in the workbook, in workbook order."""
    try:
        with pd.ExcelFile(source) as workbook:
            return list(workbook.sheet_names)
    except Exception as error:  # openpyxl raises a wide range of types
        raise LoaderError(str(error))


def read_sheet(
    source: Any, sheet_name: Optional[str] = None, header_row: int = 0
) -> pd.DataFrame:
    """Read one sheet into a table, dropping empty rows and columns.

    Args:
        source: a path or a file-like object.
        sheet_name: the sheet to read; the first sheet when omitted.
        header_row: the zero-based row that holds the column titles. Use
            this when the sheet has title rows or a logo above the real
            table.

    Returns:
        A DataFrame with stripped column names and no all-blank rows or
        columns.
    """
    try:
        df = pd.read_excel(source, sheet_name=sheet_name or 0, header=header_row)
    except Exception as error:
        raise LoaderError(str(error))

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(name).strip() for name in df.columns]
    return df.reset_index(drop=True)


def raw_preview(
    source: Any, sheet_name: Optional[str] = None, n_rows: int = 12
) -> pd.DataFrame:
    """Read the top of a sheet with no header interpretation.

    Used to let a person see the sheet exactly as laid out, so they can
    pick which row actually holds the column titles.

    Args:
        source: a path or a file-like object.
        sheet_name: the sheet to read; the first sheet when omitted.
        n_rows: how many rows to preview.
    """
    try:
        df = pd.read_excel(source, sheet_name=sheet_name or 0, header=None,
                           nrows=n_rows)
    except Exception as error:
        raise LoaderError(str(error))

    df.columns = ["Column {0}".format(i + 1) for i in range(df.shape[1])]
    return df
