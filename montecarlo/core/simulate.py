"""Monte Carlo sampling of project durations."""
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import norm

DEFAULT_SEED = 20260820
DEFAULT_ITERATIONS = 10000


def triangular_from_uniform(
    u: np.ndarray, o: np.ndarray, m: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Map uniform draws onto a triangular distribution per task.

    We invert the CDF by hand rather than calling ``np.random.triangular``
    because the uniforms are generated elsewhere and may be correlated.

    Args:
        u: uniforms in [0, 1], shape (n_iterations, n_tasks).
        o, m, p: optimistic, most likely and pessimistic values,
            each of shape (n_tasks,), broadcast across iterations.

    Returns:
        Durations of shape (n_iterations, n_tasks).
    """
    o = np.asarray(o, dtype=float)
    m = np.asarray(m, dtype=float)
    p = np.asarray(p, dtype=float)

    span = p - o
    # A zero span means the task has no uncertainty. Substitute 1.0 so the
    # divisions below stay finite; the result is overwritten at the end.
    degenerate = span <= 0
    safe_span = np.where(degenerate, 1.0, span)

    f_mode = (m - o) / safe_span
    lower = o + np.sqrt(u * safe_span * (m - o))
    upper = p - np.sqrt((1.0 - u) * safe_span * (p - m))
    result = np.where(u < f_mode, lower, upper)

    return np.where(degenerate, o, result)


def correlated_uniforms(
    n_iterations: int, n_tasks: int, rho: float, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Draw uniforms whose pairwise correlation is ``rho``.

    A one-factor Gaussian copula: every task shares one common shock per
    iteration, which is what makes real projects slip together.

    Args:
        n_iterations: number of simulation runs.
        n_tasks: number of tasks.
        rho: correlation in [0, 1). 0 means fully independent.
        seed: seed for the generator.

    Returns:
        Uniforms of shape (n_iterations, n_tasks).
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1), got {0!r}".format(rho))

    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n_iterations, 1))
    idiosyncratic = rng.standard_normal((n_iterations, n_tasks))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic
    return norm.cdf(z)


def aggregate(durations: np.ndarray, streams: Optional[Sequence[Any]]) -> np.ndarray:
    """Combine per-task durations into one project total per iteration.

    Tasks inside a stream run in sequence, so they add up. Streams run in
    parallel, so the project finishes with the longest one. Without streams
    the whole project is one sequence.

    Args:
        durations: shape (n_iterations, n_tasks).
        streams: one label per task, or None.

    Returns:
        Totals of shape (n_iterations,).
    """
    if streams is None:
        return durations.sum(axis=1)

    labels = np.asarray(streams, dtype=object)
    if labels.shape[0] != durations.shape[1]:
        raise ValueError("streams must have one label per task")

    unique = list(dict.fromkeys(labels.tolist()))
    stream_totals = np.column_stack(
        [durations[:, labels == label].sum(axis=1) for label in unique]
    )
    return stream_totals.max(axis=1)


def simulate(
    o: np.ndarray,
    m: np.ndarray,
    p: np.ndarray,
    streams: Optional[Sequence[Any]] = None,
    rho: float = 0.3,
    n_iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Run the full Monte Carlo simulation.

    Args:
        o, m, p: optimistic, most likely and pessimistic estimates,
            each of shape (n_tasks,).
        streams: optional stream label per task.
        rho: correlation between task durations, in [0, 1).
        n_iterations: number of runs.
        seed: seed for the generator.

    Returns:
        Project totals of shape (n_iterations,).
    """
    o = np.asarray(o, dtype=float)
    m = np.asarray(m, dtype=float)
    p = np.asarray(p, dtype=float)
    if not (o.shape == m.shape == p.shape):
        raise ValueError("o, m and p must have the same length")
    if o.size == 0:
        raise ValueError("no tasks to simulate")

    u = correlated_uniforms(n_iterations, o.size, rho, seed)
    durations = triangular_from_uniform(u, o, m, p)
    return aggregate(durations, streams)
