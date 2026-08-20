"""Turning a cloud of simulated totals into numbers a manager can use."""
from typing import Any, Dict, Optional, Sequence

import numpy as np

from montecarlo.core.simulate import aggregate

DEFAULT_LEVELS = (50, 85, 95)


def percentiles(
    totals: np.ndarray, levels: Sequence[int] = DEFAULT_LEVELS
) -> Dict[int, float]:
    """Return the duration met or beaten in each given percentage of runs.

    Args:
        totals: simulated project totals, shape (n_iterations,).
        levels: confidence levels as whole percentages.

    Returns:
        A mapping from level to duration.
    """
    values = np.percentile(np.asarray(totals, dtype=float), list(levels))
    return {int(level): float(value) for level, value in zip(levels, values)}


def deterministic_baseline(
    m: np.ndarray, streams: Optional[Sequence[Any]] = None
) -> float:
    """The duration the plan claims: the realistic estimates, aggregated once.

    This is the number the project would have reported without simulating.
    Showing it beside P85 is the point of the tool.
    """
    m = np.asarray(m, dtype=float).reshape(1, -1)
    return float(aggregate(m, streams)[0])


def probability_of(totals: np.ndarray, target: float) -> float:
    """Percentage of runs that finish on or before ``target``."""
    totals = np.asarray(totals, dtype=float)
    return float((totals <= target).sum() / totals.size * 100.0)
