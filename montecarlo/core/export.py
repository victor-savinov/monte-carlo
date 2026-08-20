"""A one-file record of a run, so a result can be reopened months later."""
import io
from typing import Any, Dict, Optional

import pandas as pd

from montecarlo.core.validate import Prepared


def summary_workbook(
    prepared: Prepared,
    pctls: Dict[int, float],
    date_labels: Dict[int, str],
    settings: Dict[str, Any],
    baseline: float,
    baseline_probability: Optional[float] = None,
) -> bytes:
    """Build the summary workbook as bytes for a download button.

    Three sheets: the answer, the settings that produced it, and the inputs.
    """
    result_rows = [
        {"Measure": "Plan (sum of realistic)",
         "Duration, working days": round(baseline, 1),
         "Finish date": "",
         "Probability, %": ("" if baseline_probability is None
                            else round(baseline_probability, 1))}
    ]
    for level in sorted(pctls):
        result_rows.append({
            "Measure": "P{0}".format(level),
            "Duration, working days": round(pctls[level], 1),
            "Finish date": date_labels.get(level, ""),
            "Probability, %": level,
        })

    settings_rows = [{"Setting": key, "Value": str(value)}
                     for key, value in settings.items()]

    tasks = pd.DataFrame({
        "Task": prepared.names,
        "Optimistic": prepared.o,
        "Realistic": prepared.m,
        "Pessimistic": prepared.p,
    })
    if prepared.streams is not None:
        tasks.insert(1, "Stream", prepared.streams)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(result_rows).to_excel(writer, sheet_name="Result", index=False)
        pd.DataFrame(settings_rows).to_excel(writer, sheet_name="Settings", index=False)
        tasks.to_excel(writer, sheet_name="Tasks", index=False)
    return buffer.getvalue()
