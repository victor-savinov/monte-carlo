# Monte Carlo Schedule Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app that reads three-point task estimates from an Excel file, runs 10,000 Monte Carlo simulations, and reports P50/P85/P95 schedules with presentation-ready histogram and S-curve charts.

**Architecture:** Thin UI over a pure-Python core. `app.py` holds every Streamlit widget and nothing else; each module under `core/` takes plain values (numpy arrays, DataFrames, dicts) and returns plain values, so the whole engine is unit-testable without a browser. Data flows one way: load → map → validate → simulate → stats → charts → export.

**Tech Stack:** Python 3.9, numpy, scipy, pandas, openpyxl, matplotlib, streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-monte-carlo-schedule-estimator-design.md`

## Global Constraints

- **Python 3.9.** No PEP 604 unions (`int | None`), no `match`, no `list[str]` builtins in annotations. Use `typing.Optional`, `typing.List`, `typing.Dict`, `typing.Tuple`, `typing.Sequence`, `typing.NamedTuple`.
- **Dependencies are exactly:** `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`, `pytest`. Adding any other package requires an explicit decision, not a convenience import.
- **Randomness:** `np.random.default_rng(seed)` only. Never `np.random.seed`, never the legacy global functions. Default seed is the module constant `DEFAULT_SEED = 20260820`.
- **The core never touches Streamlit, never prints, never writes files.** `import streamlit` may appear in `app.py` and nowhere else. `charts.py` returns `Figure` objects; the caller saves them.
- **Validation returns data, never raises.** Every user-facing problem is an `Issue` in a list.
- **Comments and identifiers are English.** Docstrings on every public function, one line minimum.
- **Simulation is vectorised.** No Python-level loop over iterations anywhere in `core/simulate.py`.
- **Language of the tool's own UI text:** English.

---

### Task 1: Project scaffolding and the triangular sampler

The inverse triangular CDF is the heart of the tool; everything else is plumbing around it. Scaffolding rides along with it because there is nowhere to put the test until the project exists.

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `montecarlo/__init__.py` (empty)
- Create: `montecarlo/core/__init__.py` (empty)
- Create: `montecarlo/core/simulate.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_simulate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `montecarlo.core.simulate.triangular_from_uniform(u, o, m, p) -> np.ndarray`, where `u` has shape `(n_iterations, n_tasks)` and `o`, `m`, `p` are float arrays of shape `(n_tasks,)` broadcast across rows. Also the constants `DEFAULT_SEED = 20260820` and `DEFAULT_ITERATIONS = 10000`.

- [ ] **Step 1: Create the repository skeleton**

The working directory is not yet a git repository. Run:

```bash
cd "/Users/viktorsavinov/Claude Directory/Monte-carlo"
git init
mkdir -p montecarlo/core tests sample_data
touch montecarlo/__init__.py montecarlo/core/__init__.py tests/__init__.py
```

Create `requirements.txt`:

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
openpyxl>=3.1
matplotlib>=3.7
streamlit>=1.30
pytest>=7.4
```

Create `.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.DS_Store
*.png
!docs/*.png
```

Then install: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

All later commands assume `.venv/bin/python` and `.venv/bin/pytest`.

- [ ] **Step 2: Write the failing test**

Create `tests/test_simulate.py`:

```python
"""Tests for the Monte Carlo sampling engine."""
import numpy as np
import pytest

from montecarlo.core.simulate import triangular_from_uniform


def test_uniform_zero_returns_lower_bound():
    o = np.array([10.0])
    m = np.array([15.0])
    p = np.array([30.0])
    u = np.array([[0.0]])
    assert triangular_from_uniform(u, o, m, p)[0, 0] == pytest.approx(10.0)


def test_uniform_one_returns_upper_bound():
    o = np.array([10.0])
    m = np.array([15.0])
    p = np.array([30.0])
    u = np.array([[1.0]])
    assert triangular_from_uniform(u, o, m, p)[0, 0] == pytest.approx(30.0)


def test_uniform_at_mode_cdf_returns_the_mode():
    """F(m) = (m - o) / (p - o); feeding it back must return m exactly."""
    o, m, p = np.array([10.0]), np.array([15.0]), np.array([30.0])
    f_mode = (m - o) / (p - o)
    result = triangular_from_uniform(f_mode.reshape(1, 1), o, m, p)
    assert result[0, 0] == pytest.approx(15.0)


def test_sample_moments_match_closed_form():
    """Mean = (o+m+p)/3, Var = (o^2+m^2+p^2-om-op-mp)/18."""
    o, m, p = np.array([10.0]), np.array([15.0]), np.array([30.0])
    rng = np.random.default_rng(20260820)
    u = rng.random((200_000, 1))
    x = triangular_from_uniform(u, o, m, p)

    expected_mean = (10 + 15 + 30) / 3
    expected_var = (100 + 225 + 900 - 150 - 300 - 450) / 18
    assert x.mean() == pytest.approx(expected_mean, rel=0.01)
    assert x.var() == pytest.approx(expected_var, rel=0.03)


def test_degenerate_task_returns_a_constant():
    """A task with no uncertainty must not divide by zero."""
    o = m = p = np.array([7.0])
    u = np.array([[0.0], [0.5], [1.0]]).reshape(3, 1)
    x = triangular_from_uniform(u, o, m, p)
    assert np.all(x == 7.0)


def test_samples_stay_inside_the_bounds():
    o = np.array([5.0, 20.0])
    m = np.array([8.0, 22.0])
    p = np.array([9.0, 60.0])
    rng = np.random.default_rng(1)
    x = triangular_from_uniform(rng.random((5000, 2)), o, m, p)
    assert np.all(x >= o - 1e-9)
    assert np.all(x <= p + 1e-9)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_simulate.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'montecarlo.core.simulate'`

- [ ] **Step 4: Write the implementation**

Create `montecarlo/core/simulate.py`:

```python
"""Monte Carlo sampling of project durations."""
from typing import Any, Optional, Sequence

import numpy as np

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_simulate.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt montecarlo tests
git commit -m "feat: triangular sampler with project scaffolding"
```

---

### Task 2: Correlated uniforms and stream aggregation

**Files:**
- Modify: `montecarlo/core/simulate.py`
- Modify: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `triangular_from_uniform` from Task 1.
- Produces:
  - `correlated_uniforms(n_iterations: int, n_tasks: int, rho: float, seed: int) -> np.ndarray` returning shape `(n_iterations, n_tasks)`.
  - `aggregate(durations: np.ndarray, streams: Optional[Sequence[Any]]) -> np.ndarray` returning shape `(n_iterations,)`.
  - `simulate(o, m, p, streams=None, rho=0.3, n_iterations=DEFAULT_ITERATIONS, seed=DEFAULT_SEED) -> np.ndarray` returning shape `(n_iterations,)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simulate.py`:

```python
from montecarlo.core.simulate import aggregate, correlated_uniforms, simulate


def test_uniforms_have_the_right_shape_and_range():
    u = correlated_uniforms(1000, 4, rho=0.3, seed=1)
    assert u.shape == (1000, 4)
    assert np.all(u > 0.0) and np.all(u < 1.0)


def test_zero_correlation_leaves_tasks_independent():
    u = correlated_uniforms(200_000, 2, rho=0.0, seed=7)
    assert np.corrcoef(u[:, 0], u[:, 1])[0, 1] == pytest.approx(0.0, abs=0.02)


def test_requested_correlation_is_delivered():
    """Rank correlation of the uniforms tracks the requested rho closely."""
    u = correlated_uniforms(200_000, 2, rho=0.6, seed=7)
    assert np.corrcoef(u[:, 0], u[:, 1])[0, 1] == pytest.approx(0.6, abs=0.03)


def test_uniforms_are_reproducible():
    assert np.array_equal(
        correlated_uniforms(500, 3, rho=0.3, seed=42),
        correlated_uniforms(500, 3, rho=0.3, seed=42),
    )


def test_aggregate_without_streams_sums_every_task():
    durations = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    assert np.array_equal(aggregate(durations, None), np.array([6.0, 60.0]))


def test_aggregate_with_streams_sums_within_and_takes_the_max_across():
    # Stream A = 1 + 2 = 3, stream B = 10. Longest path is 10.
    durations = np.array([[1.0, 2.0, 10.0], [5.0, 5.0, 4.0]])
    streams = ["A", "A", "B"]
    assert np.array_equal(aggregate(durations, streams), np.array([10.0, 10.0]))


def test_aggregate_treats_a_single_stream_as_a_plain_sum():
    durations = np.array([[1.0, 2.0, 3.0]])
    assert np.array_equal(aggregate(durations, ["X", "X", "X"]), np.array([6.0]))


def test_simulate_returns_one_total_per_iteration():
    o = np.array([5.0, 10.0])
    m = np.array([8.0, 14.0])
    p = np.array([20.0, 30.0])
    totals = simulate(o, m, p, n_iterations=1000, seed=3)
    assert totals.shape == (1000,)
    assert np.all(totals >= 15.0) and np.all(totals <= 50.0)


def test_correlation_widens_the_distribution():
    """Independent tasks cancel each other out; correlated ones do not."""
    o = np.full(10, 5.0)
    m = np.full(10, 10.0)
    p = np.full(10, 40.0)
    narrow = simulate(o, m, p, rho=0.0, n_iterations=40_000, seed=11)
    wide = simulate(o, m, p, rho=0.6, n_iterations=40_000, seed=11)
    assert wide.std() > narrow.std() * 1.5


def test_simulate_is_reproducible():
    o, m, p = np.array([5.0]), np.array([8.0]), np.array([20.0])
    assert np.array_equal(
        simulate(o, m, p, n_iterations=500, seed=99),
        simulate(o, m, p, n_iterations=500, seed=99),
    )


def test_simulate_rejects_an_out_of_range_correlation():
    o, m, p = np.array([5.0]), np.array([8.0]), np.array([20.0])
    with pytest.raises(ValueError):
        simulate(o, m, p, rho=1.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_simulate.py -v`
Expected: `ImportError: cannot import name 'aggregate'`

- [ ] **Step 3: Write the implementation**

Add to `montecarlo/core/simulate.py` (the `scipy` import goes at the top with the others):

```python
from scipy.stats import norm


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_simulate.py -v`
Expected: 17 passed. The two 200,000-iteration correlation tests take a few seconds; that is expected.

- [ ] **Step 5: Commit**

```bash
git add montecarlo/core/simulate.py tests/test_simulate.py
git commit -m "feat: correlated sampling and stream aggregation"
```

---

### Task 3: Statistics

**Files:**
- Create: `montecarlo/core/stats.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Consumes: nothing at runtime; mirrors the stream semantics of `aggregate` from Task 2.
- Produces:
  - `DEFAULT_LEVELS = (50, 85, 95)`
  - `percentiles(totals: np.ndarray, levels: Sequence[int] = DEFAULT_LEVELS) -> Dict[int, float]`
  - `deterministic_baseline(m: np.ndarray, streams: Optional[Sequence[Any]] = None) -> float`
  - `probability_of(totals: np.ndarray, target: float) -> float` returning a percentage in [0, 100].

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats.py`:

```python
"""Tests for percentile and baseline reporting."""
import numpy as np
import pytest

from montecarlo.core.stats import (
    DEFAULT_LEVELS,
    deterministic_baseline,
    percentiles,
    probability_of,
)


def test_percentiles_on_a_known_array():
    totals = np.arange(1.0, 101.0)  # 1..100
    result = percentiles(totals)
    assert set(result) == set(DEFAULT_LEVELS)
    assert result[50] == pytest.approx(50.5)
    assert result[95] == pytest.approx(95.05)


def test_percentiles_accept_custom_levels():
    totals = np.arange(1.0, 101.0)
    assert set(percentiles(totals, levels=(10, 90))) == {10, 90}


def test_percentiles_are_ordered():
    rng = np.random.default_rng(5)
    result = percentiles(rng.normal(100, 15, 10_000))
    assert result[50] < result[85] < result[95]


def test_baseline_without_streams_is_the_sum():
    assert deterministic_baseline(np.array([1.0, 2.0, 3.0])) == pytest.approx(6.0)


def test_baseline_with_streams_is_the_longest_stream():
    m = np.array([1.0, 2.0, 10.0])
    assert deterministic_baseline(m, ["A", "A", "B"]) == pytest.approx(10.0)


def test_probability_counts_runs_at_or_below_the_target():
    totals = np.array([10.0, 20.0, 30.0, 40.0])
    assert probability_of(totals, 20.0) == pytest.approx(50.0)


def test_probability_is_zero_below_every_run():
    assert probability_of(np.array([10.0, 20.0]), 5.0) == pytest.approx(0.0)


def test_probability_is_one_hundred_above_every_run():
    assert probability_of(np.array([10.0, 20.0]), 99.0) == pytest.approx(100.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.stats'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/stats.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stats.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add montecarlo/core/stats.py tests/test_stats.py
git commit -m "feat: percentiles, plan baseline and target probability"
```

---

### Task 4: Units and calendar dates

**Files:**
- Create: `montecarlo/core/dates.py`
- Create: `tests/test_dates.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `to_working_days(duration: float, unit: str, days_per_week: int) -> float` where `unit` is `"days"` or `"weeks"`.
  - `to_date(duration_days: float, start_date: date, days_per_week: int) -> date`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dates.py`:

```python
"""Tests for unit conversion and calendar arithmetic."""
from datetime import date

import pytest

from montecarlo.core.dates import to_date, to_working_days


def test_days_pass_through_unchanged():
    assert to_working_days(10.0, "days", 5) == pytest.approx(10.0)


def test_weeks_convert_using_the_working_week():
    assert to_working_days(2.0, "weeks", 5) == pytest.approx(10.0)
    assert to_working_days(2.0, "weeks", 7) == pytest.approx(14.0)


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        to_working_days(1.0, "fortnights", 5)


def test_seven_day_week_is_plain_calendar_arithmetic():
    # Monday 1 June 2026 + 10 days of work finishes on 10 June.
    assert to_date(10, date(2026, 6, 1), 7) == date(2026, 6, 10)


def test_five_day_week_skips_the_weekend():
    # Monday 1 June + 5 working days -> Friday 5 June.
    assert to_date(5, date(2026, 6, 1), 5) == date(2026, 6, 5)
    # 6 working days spills over the weekend to Monday 8 June.
    assert to_date(6, date(2026, 6, 1), 5) == date(2026, 6, 8)


def test_five_day_week_crosses_a_month_boundary():
    # Monday 1 June + 25 working days -> Friday 3 July.
    assert to_date(25, date(2026, 6, 1), 5) == date(2026, 7, 3)


def test_a_start_date_on_a_weekend_moves_to_monday():
    # Saturday 6 June 2026; the first working day is Monday 8 June.
    assert to_date(1, date(2026, 6, 6), 5) == date(2026, 6, 8)


def test_fractional_durations_round_up_to_a_whole_day():
    assert to_date(4.2, date(2026, 6, 1), 5) == date(2026, 6, 5)


def test_zero_duration_returns_the_start_date():
    assert to_date(0, date(2026, 6, 1), 5) == date(2026, 6, 1)


def test_unsupported_working_week_is_rejected():
    with pytest.raises(ValueError):
        to_date(5, date(2026, 6, 1), 6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dates.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.dates'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/dates.py`:

```python
"""Converting durations into units and calendar dates.

Public holidays are not modelled; the interface says so out loud.
"""
import math
from datetime import date, timedelta

DAYS = "days"
WEEKS = "weeks"
UNITS = (DAYS, WEEKS)


def to_working_days(duration: float, unit: str, days_per_week: int) -> float:
    """Express a duration in working days.

    The working week is shared with the calendar conversion so the two
    settings cannot drift apart.

    Args:
        duration: the estimate as written in the source file.
        unit: "days" or "weeks".
        days_per_week: 5 or 7.
    """
    if unit == DAYS:
        return float(duration)
    if unit == WEEKS:
        return float(duration) * days_per_week
    raise ValueError("unit must be one of {0}, got {1!r}".format(UNITS, unit))


def to_date(duration_days: float, start_date: date, days_per_week: int) -> date:
    """Return the finish date for a duration in working days.

    Args:
        duration_days: working days of effort. Fractions round up, because
            half a day of remaining work still occupies a day.
        start_date: the first day of work. A weekend start rolls forward to
            the next Monday when the working week is 5 days.
        days_per_week: 5 (Monday to Friday) or 7 (every day).
    """
    if days_per_week not in (5, 7):
        raise ValueError("days_per_week must be 5 or 7, got {0!r}".format(days_per_week))

    whole_days = int(math.ceil(duration_days))
    if whole_days <= 0:
        return start_date

    if days_per_week == 7:
        return start_date + timedelta(days=whole_days - 1)

    current = start_date
    while current.weekday() >= 5:  # Saturday = 5, Sunday = 6
        current += timedelta(days=1)

    remaining = whole_days - 1
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dates.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add montecarlo/core/dates.py tests/test_dates.py
git commit -m "feat: unit conversion and working-day calendar"
```

---

### Task 5: Reading the Excel file

**Files:**
- Create: `montecarlo/core/loader.py`
- Create: `tests/test_loader.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `sheet_names(source) -> List[str]`
  - `read_sheet(source, sheet_name: Optional[str] = None) -> pd.DataFrame`
  - `LoaderError` (subclass of `Exception`), raised only for a file that cannot be opened at all.
  - `source` is anything `pandas.read_excel` accepts: a path, or the file-like object Streamlit's uploader returns.

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
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
```

Create `tests/test_loader.py`:

```python
"""Tests for reading estimates out of a workbook."""
import pytest

from montecarlo.core.loader import LoaderError, read_sheet, sheet_names


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_loader.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.loader'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/loader.py`:

```python
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


def read_sheet(source: Any, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read one sheet into a table, dropping empty rows and columns.

    Args:
        source: a path or a file-like object.
        sheet_name: the sheet to read; the first sheet when omitted.

    Returns:
        A DataFrame with stripped column names and no all-blank rows or
        columns.
    """
    try:
        df = pd.read_excel(source, sheet_name=sheet_name or 0)
    except Exception as error:
        raise LoaderError(str(error))

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(name).strip() for name in df.columns]
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_loader.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add montecarlo/core/loader.py tests/test_loader.py tests/conftest.py
git commit -m "feat: workbook loader with sheet selection"
```

---

### Task 6: Column mapping

**Files:**
- Create: `montecarlo/core/mapping.py`
- Create: `tests/test_mapping.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ROLES = ("task", "optimistic", "realistic", "pessimistic", "stream")`
  - `REQUIRED_ROLES = ("task", "optimistic", "realistic", "pessimistic")`
  - `class RoleGuess(NamedTuple)` with fields `column: Optional[str]` and `confidence: str`, where confidence is `"exact"`, `"fuzzy"` or `"none"`.
  - `normalize_header(raw: str) -> str`
  - `guess_mapping(columns: Sequence[str]) -> Dict[str, RoleGuess]` returning a guess for every role in `ROLES`.
  - Later tasks pass a plain `Dict[str, Optional[str]]` (role to column name) rather than the guess objects.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mapping.py`:

```python
"""Tests for turning spreadsheet headers into roles."""
from montecarlo.core.mapping import (
    REQUIRED_ROLES,
    ROLES,
    guess_mapping,
    normalize_header,
)


def test_normalisation_lowercases_and_strips_punctuation():
    assert normalize_header("  Best Case (d) ") == "best case d"
    assert normalize_header("Most-Likely") == "most likely"


def test_canonical_english_headers_map_exactly():
    guess = guess_mapping(["Task", "Optimistic", "Realistic", "Pessimistic"])
    for role in REQUIRED_ROLES:
        assert guess[role].confidence == "exact"
    assert guess["optimistic"].column == "Optimistic"
    assert guess["stream"].column is None


def test_common_business_synonyms_map():
    guess = guess_mapping(
        ["Work package", "Best case", "Most likely", "Worst case", "Track"]
    )
    assert guess["task"].column == "Work package"
    assert guess["optimistic"].column == "Best case"
    assert guess["realistic"].column == "Most likely"
    assert guess["pessimistic"].column == "Worst case"
    assert guess["stream"].column == "Track"


def test_russian_headers_map():
    guess = guess_mapping(
        ["Задача", "Оптимистичная", "Реалистичная", "Пессимистичная", "Поток"]
    )
    assert guess["task"].column == "Задача"
    assert guess["realistic"].column == "Реалистичная"
    assert guess["stream"].column == "Поток"


def test_units_in_parentheses_are_ignored():
    guess = guess_mapping(["Activity", "Optimistic (days)", "Realistic (days)",
                           "Pessimistic (days)"])
    assert guess["optimistic"].column == "Optimistic (days)"


def test_a_typo_is_recovered_by_fuzzy_matching():
    guess = guess_mapping(["Task", "Optimisitc", "Realistic", "Pessimistic"])
    assert guess["optimistic"].column == "Optimisitc"
    assert guess["optimistic"].confidence == "fuzzy"


def test_unrecognisable_headers_produce_no_guess():
    guess = guess_mapping(["alpha", "beta", "gamma", "delta"])
    assert all(guess[role].column is None for role in ROLES)
    assert all(guess[role].confidence == "none" for role in ROLES)


def test_a_column_is_never_claimed_by_two_roles():
    guess = guess_mapping(["Estimate", "Estimate 2"])
    claimed = [g.column for g in guess.values() if g.column is not None]
    assert len(claimed) == len(set(claimed))


def test_a_missing_stream_column_is_not_an_error():
    guess = guess_mapping(["Task", "Optimistic", "Realistic", "Pessimistic"])
    assert guess["stream"].column is None
    assert guess["stream"].confidence == "none"


def test_every_role_is_present_in_the_result():
    assert set(guess_mapping(["anything"])) == set(ROLES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_mapping.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.mapping'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/mapping.py`:

```python
"""Guessing which spreadsheet column plays which role.

The guess is never final. It pre-fills the dropdowns on screen so a person
confirms it, because a wrong mapping is invisible in the result.
"""
import difflib
import re
from typing import Dict, NamedTuple, Optional, Sequence

ROLES = ("task", "optimistic", "realistic", "pessimistic", "stream")
REQUIRED_ROLES = ("task", "optimistic", "realistic", "pessimistic")

SYNONYMS = {
    "task": [
        "task", "task name", "name", "activity", "work item", "work package",
        "workpackage", "description", "deliverable", "wbs", "item", "step",
        "задача", "работа", "этап", "наименование", "название",
    ],
    "optimistic": [
        "optimistic", "optimistic days", "best", "best case", "best case d",
        "min", "minimum", "low", "lo", "shortest", "o",
        "оптимистичная", "оптимистичный", "оптимистично", "минимум", "лучший",
    ],
    "realistic": [
        "realistic", "realistic days", "most likely", "mostlikely", "likely",
        "expected", "expectation", "estimate", "mode", "normal", "ml", "m",
        "реалистичная", "реалистичный", "ожидаемая", "наиболее вероятная",
        "вероятная", "оценка",
    ],
    "pessimistic": [
        "pessimistic", "pessimistic days", "worst", "worst case",
        "worst case d", "max", "maximum", "high", "hi", "longest", "p",
        "пессимистичная", "пессимистичный", "максимум", "худший",
    ],
    "stream": [
        "stream", "workstream", "work stream", "track", "group",
        "parallel group", "parallel", "phase", "swimlane", "lane",
        "поток", "ветка", "направление", "группа", "фаза",
    ],
}

FUZZY_CUTOFF = 0.75
MIN_FUZZY_LENGTH = 4  # short synonyms like "o" or "max" fuzzy-match anything


class RoleGuess(NamedTuple):
    """One role's best candidate column and how it was found."""

    column: Optional[str]
    confidence: str  # "exact", "fuzzy" or "none"


def normalize_header(raw: str) -> str:
    """Reduce a header to lowercase words separated by single spaces."""
    text = str(raw).lower().strip()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _exact_pass(
    normalized: Dict[str, str], taken: set
) -> Dict[str, RoleGuess]:
    """Match headers whose normalised form is a synonym outright."""
    found = {}
    for role in ROLES:
        vocabulary = set(SYNONYMS[role])
        for column, norm in normalized.items():
            if column in taken:
                continue
            if norm in vocabulary:
                found[role] = RoleGuess(column, "exact")
                taken.add(column)
                break
    return found


def _fuzzy_pass(
    normalized: Dict[str, str], taken: set, unresolved: Sequence[str]
) -> Dict[str, RoleGuess]:
    """Recover typos and suffixes for roles the exact pass missed."""
    found = {}
    for role in unresolved:
        vocabulary = [s for s in SYNONYMS[role] if len(s) >= MIN_FUZZY_LENGTH]
        best_column = None
        best_score = 0.0
        for column, norm in normalized.items():
            if column in taken:
                continue
            for word in [norm] + norm.split():
                matches = difflib.get_close_matches(
                    word, vocabulary, n=1, cutoff=FUZZY_CUTOFF
                )
                if not matches:
                    continue
                score = difflib.SequenceMatcher(None, word, matches[0]).ratio()
                if score > best_score:
                    best_score, best_column = score, column
        if best_column is not None:
            found[role] = RoleGuess(best_column, "fuzzy")
            taken.add(best_column)
    return found


def guess_mapping(columns: Sequence[str]) -> Dict[str, RoleGuess]:
    """Propose a column for every role.

    Two passes: exact synonym match first, then fuzzy matching for whatever
    is left. A column is claimed by at most one role.

    Args:
        columns: the sheet's column names, as written.

    Returns:
        A guess for every role in ROLES; unmatched roles get
        ``RoleGuess(None, "none")``.
    """
    normalized = {column: normalize_header(column) for column in columns}
    taken = set()

    guesses = _exact_pass(normalized, taken)
    unresolved = [role for role in ROLES if role not in guesses]
    guesses.update(_fuzzy_pass(normalized, taken, unresolved))

    return {role: guesses.get(role, RoleGuess(None, "none")) for role in ROLES}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_mapping.py -v`
Expected: 10 passed. If `test_unrecognisable_headers_produce_no_guess` fails because a nonsense header fuzzy-matched, raise `FUZZY_CUTOFF` rather than special-casing the test — a mapper that matches anything is worse than one that asks.

- [ ] **Step 5: Commit**

```bash
git add montecarlo/core/mapping.py tests/test_mapping.py
git commit -m "feat: column role auto-detection"
```

---

### Task 7: Validation and array extraction

**Files:**
- Create: `montecarlo/core/validate.py`
- Create: `tests/test_validate.py`

**Interfaces:**
- Consumes: `REQUIRED_ROLES` from Task 6; the mapping arrives as `Dict[str, Optional[str]]` (role to column name).
- Produces:
  - `class Issue(NamedTuple)` with `severity: str` (`"error"` or `"warning"`), `row: Optional[int]` (1-based spreadsheet row, header excluded), `column: Optional[str]`, `message: str`.
  - `validate(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> List[Issue]`
  - `class Prepared(NamedTuple)` with `names: List[str]`, `o: np.ndarray`, `m: np.ndarray`, `p: np.ndarray`, `streams: Optional[List[str]]`.
  - `prepare(df, mapping, sort_three_point: bool = False) -> Prepared`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate.py`:

```python
"""Tests for input validation and array extraction."""
import numpy as np
import pandas as pd
import pytest

from montecarlo.core.validate import Issue, prepare, validate

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_validate.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.validate'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/validate.py`:

```python
"""Checking the input table and turning it into arrays.

Nothing here raises on user data. Problems come back as a list so the screen
can show all of them at once, rather than one exception at a time.
"""
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from montecarlo.core.mapping import REQUIRED_ROLES

ERROR = "error"
WARNING = "warning"
DURATION_ROLES = ("optimistic", "realistic", "pessimistic")
BLANK_STREAM = "(no stream)"


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_validate.py -v`
Expected: 16 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -v`
Expected: 68 passed.

- [ ] **Step 6: Commit**

```bash
git add montecarlo/core/validate.py tests/test_validate.py
git commit -m "feat: input validation and array extraction"
```

---

### Task 8: Presentation charts

**Files:**
- Create: `montecarlo/core/charts.py`
- Create: `tests/test_charts.py`

**Interfaces:**
- Consumes: `percentiles` output shape (`Dict[int, float]`) from Task 3.
- Produces:
  - `histogram(totals, pctls, baseline=None, baseline_probability=None, unit_label="working days") -> matplotlib.figure.Figure`
  - `s_curve(totals, pctls, date_labels=None, unit_label="working days") -> matplotlib.figure.Figure`
  - `date_labels` is `Optional[Dict[int, str]]`, mapping a percentile level to a pre-formatted date string.
  - `figure_to_png_bytes(figure) -> bytes`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_charts.py`:

```python
"""Tests for the chart builders.

We assert on structure, not on pixels: the point is that the figures carry
the numbers a manager needs and that nothing writes to disk.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from montecarlo.core.charts import figure_to_png_bytes, histogram, s_curve


@pytest.fixture
def totals():
    rng = np.random.default_rng(1)
    return rng.normal(175.0, 20.0, 10_000)


@pytest.fixture
def pctls():
    return {50: 175.0, 85: 201.0, 95: 216.0}


def test_histogram_returns_a_figure_with_one_axes(totals, pctls):
    figure = histogram(totals, pctls)
    assert len(figure.axes) == 1


def test_histogram_draws_a_line_for_every_percentile(totals, pctls):
    axes = histogram(totals, pctls).axes[0]
    xs = [line.get_xdata()[0] for line in axes.lines]
    for value in pctls.values():
        assert any(abs(x - value) < 1e-6 for x in xs)


def test_histogram_draws_the_baseline_when_given(totals, pctls):
    axes = histogram(totals, pctls, baseline=149.0).axes[0]
    xs = [line.get_xdata()[0] for line in axes.lines]
    assert any(abs(x - 149.0) < 1e-6 for x in xs)


def test_histogram_omits_the_baseline_when_not_given(totals, pctls):
    with_baseline = len(histogram(totals, pctls, baseline=149.0).axes[0].lines)
    without = len(histogram(totals, pctls).axes[0].lines)
    assert with_baseline == without + 1


def test_s_curve_rises_monotonically(totals, pctls):
    axes = s_curve(totals, pctls).axes[0]
    ys = axes.lines[0].get_ydata()
    assert np.all(np.diff(ys) >= -1e-9)


def test_s_curve_spans_zero_to_one_hundred_percent(totals, pctls):
    ys = s_curve(totals, pctls).axes[0].lines[0].get_ydata()
    assert ys[0] == pytest.approx(0.0, abs=0.5)
    assert ys[-1] == pytest.approx(100.0, abs=0.5)


def test_s_curve_annotates_dates_when_given(totals, pctls):
    figure = s_curve(totals, pctls, date_labels={50: "15 Jan 2027",
                                                 85: "22 Feb 2027",
                                                 95: "15 Mar 2027"})
    text = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "22 Feb 2027" in text


def test_the_unit_label_reaches_the_axis(totals, pctls):
    axes = histogram(totals, pctls, unit_label="weeks").axes[0]
    assert "weeks" in axes.get_xlabel()


def test_png_export_produces_bytes(totals, pctls):
    data = figure_to_png_bytes(histogram(totals, pctls))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_charts.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.charts'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/charts.py`:

```python
"""Charts built to be projected in a management meeting.

Large type, no gridline clutter, labels on the lines instead of a legend
the audience has to decode. The functions return figures; saving is the
caller's business.
"""
import io
from typing import Dict, Optional

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

FIGSIZE = (11.0, 5.5)
DPI = 300
BAR_COLOR = "#2E7D86"
BASELINE_COLOR = "#9A3A25"
P50_COLOR = "#6B7A83"
P85_COLOR = "#A2610F"
P95_COLOR = "#C9A227"

LEVEL_COLORS = {50: P50_COLOR, 85: P85_COLOR, 95: P95_COLOR}


def _style(axes) -> None:
    """Strip the chart down to what carries information."""
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.tick_params(labelsize=11, length=3, colors="#5A6B75")
    axes.grid(False)


def _level_color(level: int) -> str:
    return LEVEL_COLORS.get(level, P50_COLOR)


def histogram(
    totals: np.ndarray,
    pctls: Dict[int, float],
    baseline: Optional[float] = None,
    baseline_probability: Optional[float] = None,
    unit_label: str = "working days",
) -> Figure:
    """Where the project is likely to land.

    Args:
        totals: simulated project totals.
        pctls: level to duration, from ``stats.percentiles``.
        baseline: the plan's own duration, drawn for comparison.
        baseline_probability: the plan's chance of success, as a percentage.
        unit_label: the unit shown on the x axis.
    """
    figure = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = figure.subplots()

    axes.hist(totals, bins=60, color=BAR_COLOR, alpha=0.35, edgecolor="none")
    _style(axes)
    axes.set_yticks([])
    axes.set_xlabel("Total project duration, {0}".format(unit_label),
                    fontsize=11, color="#5A6B75", labelpad=10)

    top = axes.get_ylim()[1]

    if baseline is not None:
        axes.axvline(baseline, color=BASELINE_COLOR, linewidth=1.6,
                     linestyle=(0, (5, 3)))
        label = "PLAN {0:.0f}".format(baseline)
        if baseline_probability is not None:
            label += "\n{0:.0f}% likely".format(baseline_probability)
        axes.text(baseline, top * 1.02, label, color=BASELINE_COLOR,
                  fontsize=11, fontweight="bold", ha="center", va="bottom")

    for level in sorted(pctls):
        value = pctls[level]
        axes.axvline(value, color=_level_color(level), linewidth=1.6)
        axes.text(value, top * 1.02, "P{0} {1:.0f}".format(level, value),
                  color=_level_color(level), fontsize=11, fontweight="bold",
                  ha="center", va="bottom")

    axes.set_ylim(0, top * 1.25)
    figure.tight_layout()
    return figure


def s_curve(
    totals: np.ndarray,
    pctls: Dict[int, float],
    date_labels: Optional[Dict[int, str]] = None,
    unit_label: str = "working days",
) -> Figure:
    """Cumulative probability of finishing by a given duration.

    Args:
        totals: simulated project totals.
        pctls: level to duration, from ``stats.percentiles``.
        date_labels: optional level to formatted finish date.
        unit_label: the unit shown on the x axis.
    """
    figure = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = figure.subplots()

    ordered = np.sort(np.asarray(totals, dtype=float))
    probability = np.arange(1, ordered.size + 1) / ordered.size * 100.0
    axes.plot(ordered, probability, color=BAR_COLOR, linewidth=2.2)

    _style(axes)
    axes.set_ylim(0, 105)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes.set_xlabel("Total project duration, {0}".format(unit_label),
                    fontsize=11, color="#5A6B75", labelpad=10)

    for level in sorted(pctls):
        value = pctls[level]
        color = _level_color(level)
        axes.plot([ordered[0], value], [level, level], color=color,
                  linewidth=0.9, linestyle=(0, (2, 3)))
        axes.plot([value], [level], marker="o", markersize=7, color=color,
                  markerfacecolor="white", markeredgewidth=2)
        text = "P{0} · {1:.0f} {2}".format(level, value, unit_label)
        if date_labels and level in date_labels:
            text += " · {0}".format(date_labels[level])
        axes.annotate(text, (value, level), textcoords="offset points",
                      xytext=(12, -4), fontsize=11, color=color,
                      fontweight="bold")

    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure: Figure) -> bytes:
    """Render a figure to PNG bytes for a download button."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
    return buffer.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_charts.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add montecarlo/core/charts.py tests/test_charts.py
git commit -m "feat: histogram and S-curve charts"
```

---

### Task 9: The Streamlit screen

The UI has no unit tests; it is checked by running it. Every calculation it performs already has tests behind it.

**Files:**
- Create: `app.py`
- Create: `sample_data/tasks_sample.xlsx` (generated by a script committed alongside it)
- Create: `sample_data/make_sample.py`

**Interfaces:**
- Consumes: everything built in Tasks 1–8.
- Produces: nothing other modules depend on.

- [ ] **Step 1: Build the sample file**

Create `sample_data/make_sample.py`. The headers are deliberately non-standard, so the demo exercises the mapper rather than flattering it:

```python
"""Generate the demo workbook. Run: python sample_data/make_sample.py"""
import pandas as pd

ROWS = [
    ("Discovery & requirements", "Core", 8, 12, 20),
    ("Solution architecture", "Core", 5, 8, 15),
    ("Data model", "Core", 4, 6, 11),
    ("Backend: core services", "Core", 18, 25, 45),
    ("Backend: integrations", "Core", 10, 16, 34),
    ("Frontend: main flows", "UI", 15, 22, 38),
    ("Frontend: admin", "UI", 7, 11, 20),
    ("Migration scripts", "Data", 5, 9, 22),
    ("QA cycle 1", "Core", 8, 12, 20),
    ("QA cycle 2", "Core", 5, 8, 14),
    ("UAT & fixes", "Core", 10, 15, 30),
    ("Release & handover", "Core", 3, 5, 12),
]

df = pd.DataFrame(
    ROWS, columns=["Work package", "Track", "Best case (d)", "Expected",
                   "Worst case (d)"]
)
df.to_excel("sample_data/tasks_sample.xlsx", sheet_name="Estimates", index=False)
print("wrote sample_data/tasks_sample.xlsx")
```

Run: `.venv/bin/python sample_data/make_sample.py`

- [ ] **Step 2: Write the app**

Create `app.py`:

```python
"""Monte Carlo schedule estimator — the screen.

This file holds every widget and no arithmetic. Anything that computes
belongs in montecarlo/core and has tests.
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from montecarlo.core import charts, dates, loader, mapping, simulate, stats
from montecarlo.core.validate import ERROR, prepare, validate

st.set_page_config(page_title="Schedule Estimator", layout="wide")

ROLE_LABELS = {
    "task": "Task name",
    "optimistic": "Optimistic",
    "realistic": "Realistic",
    "pessimistic": "Pessimistic",
    "stream": "Stream (optional)",
}
NONE_OPTION = "— none —"


def sidebar_settings():
    """Collect every run setting. Returns a plain dict."""
    st.sidebar.header("Settings")
    unit = st.sidebar.selectbox("Estimates are in", dates.UNITS, index=0)
    days_per_week = st.sidebar.selectbox("Working week", [5, 7], index=0,
                                         format_func=lambda d: "{0} days".format(d))
    start = st.sidebar.date_input("Start date", value=date.today())
    rho = st.sidebar.slider(
        "Correlation between tasks", 0.0, 0.9, 0.3, 0.05,
        help="How much tasks slip together. 0 means fully independent, which "
             "makes any forecast look more precise than it is.",
    )
    target = st.sidebar.date_input("Target date (optional)", value=None)
    with st.sidebar.expander("Advanced"):
        iterations = st.number_input("Iterations", 1000, 200_000,
                                     simulate.DEFAULT_ITERATIONS, 1000)
        seed = st.number_input("Random seed", 0, 10**9, simulate.DEFAULT_SEED)
    return {
        "unit": unit, "days_per_week": days_per_week, "start": start,
        "rho": rho, "target": target, "iterations": int(iterations),
        "seed": int(seed),
    }


def mapping_controls(columns):
    """Draw one dropdown per role, pre-filled with the automatic guess."""
    guesses = mapping.guess_mapping(columns)
    chosen = {}
    grid = st.columns(len(mapping.ROLES))
    for column_box, role in zip(grid, mapping.ROLES):
        options = [NONE_OPTION] + list(columns)
        guess = guesses[role]
        index = options.index(guess.column) if guess.column in options else 0
        label = ROLE_LABELS[role]
        if guess.confidence == "fuzzy":
            label += " ⚠"
        selection = column_box.selectbox(label, options, index=index,
                                         key="map_" + role)
        chosen[role] = None if selection == NONE_OPTION else selection
    if any(g.confidence == "fuzzy" for g in guesses.values()):
        st.caption("⚠ marks a column matched by similarity. Please confirm it.")
    return chosen


def show_issues(issues):
    """Render the validation report. Returns True when the run is blocked."""
    if not issues:
        return False
    table = pd.DataFrame(
        [{"Severity": i.severity, "Row": i.row or "", "Column": i.column or "",
          "Problem": i.message} for i in issues]
    )
    blocking = any(issue.severity == ERROR for issue in issues)
    if blocking:
        st.error("{0} problem(s) must be fixed before running.".format(
            sum(1 for i in issues if i.severity == ERROR)))
    else:
        st.warning("{0} warning(s). You can still run.".format(len(issues)))
    st.dataframe(table, use_container_width=True, hide_index=True)
    return blocking


def show_results(totals, settings, baseline_days, unit_label):
    """Percentiles, dates, the plan comparison and the two charts."""
    pctls = stats.percentiles(totals)
    date_labels = {
        level: dates.to_date(value, settings["start"],
                             settings["days_per_week"]).strftime("%d %b %Y")
        for level, value in pctls.items()
    }

    columns = st.columns(3)
    captions = {50: "P50 — coin flip", 85: "P85 — commit here",
                95: "P95 — worst realistic"}
    for box, level in zip(columns, sorted(pctls)):
        box.metric(captions[level],
                   "{0:.0f} {1}".format(pctls[level], unit_label),
                   date_labels[level])

    plan_probability = stats.probability_of(totals, baseline_days)
    plan_date = dates.to_date(baseline_days, settings["start"],
                              settings["days_per_week"])
    st.info(
        "Your plan of {0:.0f} {1} lands on {2} and succeeds in {3:.0f}% of "
        "runs.".format(baseline_days, unit_label,
                       plan_date.strftime("%d %b %Y"), plan_probability)
    )

    if settings["target"]:
        target_days = _days_until(settings["target"], settings)
        st.info("Your target of {0} succeeds in {1:.0f}% of runs.".format(
            settings["target"].strftime("%d %b %Y"),
            stats.probability_of(totals, target_days)))

    distribution, curve = st.tabs(["Distribution", "S-curve"])
    with distribution:
        figure = charts.histogram(totals, pctls, baseline=baseline_days,
                                  baseline_probability=plan_probability,
                                  unit_label=unit_label)
        st.pyplot(figure)
        st.download_button("Download PNG", charts.figure_to_png_bytes(figure),
                           "distribution.png", "image/png")
    with curve:
        figure = charts.s_curve(totals, pctls, date_labels=date_labels,
                                unit_label=unit_label)
        st.pyplot(figure)
        st.download_button("Download PNG", charts.figure_to_png_bytes(figure),
                           "s_curve.png", "image/png")


def _days_until(target, settings):
    """Working days between the start date and a target date."""
    day = settings["start"]
    counted = 0
    while day <= target:
        if settings["days_per_week"] == 7 or day.weekday() < 5:
            counted += 1
        day += timedelta(days=1)
    return counted


st.title("Monte Carlo Schedule Estimator")
st.caption("Three-point estimates in, a date you can defend out.")

settings = sidebar_settings()
uploaded = st.file_uploader("Excel file with task estimates",
                            type=["xlsx", "xls"])

if uploaded is None:
    st.info("Upload a file to begin. A sample lives in sample_data/.")
    st.stop()

try:
    names = loader.sheet_names(uploaded)
    sheet = names[0] if len(names) == 1 else st.selectbox("Sheet", names)
    df = loader.read_sheet(uploaded, sheet)
except loader.LoaderError as error:
    st.error("That file could not be read as a workbook. Supported formats "
             "are .xlsx and .xls.")
    with st.expander("Details"):
        st.code(str(error))
    st.stop()

st.subheader("Map the columns")
chosen = mapping_controls(list(df.columns))
st.dataframe(df.head(6), use_container_width=True, hide_index=True)

issues = validate(df, chosen)
blocked = show_issues(issues)
sort_estimates = False
if any("out of order" in issue.message for issue in issues):
    sort_estimates = st.checkbox(
        "Sort the three estimates on out-of-order rows", value=True)

if blocked:
    st.stop()

if st.button("Run {0:,} simulations".format(settings["iterations"]),
             type="primary"):
    with st.spinner("Simulating…"):
        data = prepare(df, chosen, sort_three_point=sort_estimates)
        to_days = dates.to_working_days
        o = [to_days(v, settings["unit"], settings["days_per_week"]) for v in data.o]
        m = [to_days(v, settings["unit"], settings["days_per_week"]) for v in data.m]
        p = [to_days(v, settings["unit"], settings["days_per_week"]) for v in data.p]
        totals = simulate.simulate(o, m, p, streams=data.streams,
                                   rho=settings["rho"],
                                   n_iterations=settings["iterations"],
                                   seed=settings["seed"])
        baseline = stats.deterministic_baseline(m, data.streams)
    show_results(totals, settings, baseline, "working days")
```

- [ ] **Step 3: Catch anything unexpected at the screen boundary**

The spec requires that no traceback ever reaches the user. Wrap the run
block at the bottom of `app.py` — everything from `with st.spinner("Simulating…"):`
through the `show_results(...)` call — in this handler:

```python
    try:
        with st.spinner("Simulating…"):
            ...unchanged body...
        show_results(totals, settings, baseline, "working days")
    except Exception as error:  # the screen is the last line of defence
        st.error("The simulation could not finish. Check the column mapping "
                 "and the estimates, then try again.")
        with st.expander("Details"):
            st.exception(error)
```

- [ ] **Step 4: Run the app and check it by hand**

Run: `.venv/bin/streamlit run app.py`

Walk through this list, in order:

1. Upload `sample_data/tasks_sample.xlsx`. All five roles fill in automatically, including Stream from "Track".
2. Change Realistic to the wrong column, confirm the results change, change it back.
3. Run the simulation. P50 < P85 < P95, and each shows a date.
4. The plan comparison appears and its probability is well under 50%.
5. Both chart tabs render and both PNGs download and open.
6. Set the working week to 7 days; the dates move earlier, the durations do not.
7. Set the unit to weeks; every duration multiplies by the working week.
8. Set correlation to 0 and then 0.9; the histogram visibly narrows and widens.
9. Re-run with the same settings twice; the numbers are identical.
10. Upload a text file renamed to `.xlsx`; a plain message appears, not a traceback.

- [ ] **Step 5: Commit**

```bash
git add app.py sample_data
git commit -m "feat: streamlit screen and sample workbook"
```

---

### Task 10: Excel export and README

**Files:**
- Create: `montecarlo/core/export.py`
- Create: `tests/test_export.py`
- Modify: `app.py` (add the download button inside `show_results`)
- Create: `README.md`

**Interfaces:**
- Consumes: `Prepared` from Task 7, `percentiles` from Task 3.
- Produces: `summary_workbook(prepared, pctls, date_labels, settings, baseline, baseline_probability) -> bytes`

- [ ] **Step 1: Write the failing test**

Create `tests/test_export.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: `ModuleNotFoundError: No module named 'montecarlo.core.export'`

- [ ] **Step 3: Write the implementation**

Create `montecarlo/core/export.py`:

```python
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
        pd.DataFrame(result_rows).to_excel(writer, "Result", index=False)
        pd.DataFrame(settings_rows).to_excel(writer, "Settings", index=False)
        tasks.to_excel(writer, "Tasks", index=False)
    return buffer.getvalue()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: 4 passed.

- [ ] **Step 5: Wire the export into the screen**

In `app.py`, add the import:

```python
from montecarlo.core.export import summary_workbook
```

Change the first line of `show_results` from

```python
def show_results(totals, settings, baseline_days, unit_label):
```

to

```python
def show_results(totals, settings, baseline_days, unit_label, prepared):
```

Leave the rest of the body untouched and append this as its last statement,
at the same indentation as the `distribution, curve = st.tabs(...)` line:

```python
    st.download_button(
        "Export summary .xlsx",
        summary_workbook(
            prepared, pctls, date_labels,
            {"unit": settings["unit"],
             "days_per_week": settings["days_per_week"],
             "start": settings["start"].strftime("%d %b %Y"),
             "correlation": settings["rho"],
             "iterations": settings["iterations"],
             "seed": settings["seed"]},
            baseline_days, plan_probability,
        ),
        "schedule_summary.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
```

And update the call site at the bottom of the file:

```python
    show_results(totals, settings, baseline, "working days", data)
```

- [ ] **Step 6: Write the README**

Create `README.md`:

````markdown
# Monte Carlo Schedule Estimator

Turns three-point task estimates into a date you can defend.

A plan that adds up every "realistic" estimate produces a date with roughly
a coin-flip chance of being met — often much worse. This tool simulates the
project 10,000 times and reports the duration you would hit in 50%, 85% and
95% of those runs, with charts built for a management deck.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/streamlit run app.py
```

Then open the address it prints and upload your Excel file. A sample is in
`sample_data/tasks_sample.xlsx`.

## Your Excel file

Four columns are required and one is optional. The names do not have to
match — the tool recognises common variants in English and Russian, and you
confirm its guess on screen.

| Role | Recognised names include |
|------|--------------------------|
| Task name | Task, Activity, Work package, Задача |
| Optimistic | Optimistic, Best case, Min, Оптимистичная |
| Realistic | Realistic, Most likely, Expected, Реалистичная |
| Pessimistic | Pessimistic, Worst case, Max, Пессимистичная |
| Stream *(optional)* | Stream, Track, Workstream, Поток |

Tasks sharing a **Stream** run one after another, and streams run in
parallel with each other. Without that column every task is treated as
sequential.

## Settings that matter

- **Correlation** — how much tasks slip together. 0 assumes every task is
  independent, which makes the forecast look far more precise than it is.
  The default of 0.3 is a conservative middle.
- **Working week** — 5 or 7 days. Public holidays are not modelled.
- **Seed** — fixed, so the same file always gives the same numbers.

## Tests

```bash
.venv/bin/pytest
```

## Design documents

- Spec: `docs/superpowers/specs/2026-08-20-monte-carlo-schedule-estimator-design.md`
- Plan: `docs/superpowers/plans/2026-08-20-monte-carlo-schedule-estimator.md`
- One-pager: `docs/one-pager.html`
````

- [ ] **Step 7: Run the whole suite and the app once more**

Run: `.venv/bin/pytest -v`
Expected: 81 passed.

Run: `.venv/bin/streamlit run app.py`, upload the sample, run it, and download the Excel summary. Open it and confirm all three sheets are populated.

- [ ] **Step 8: Commit**

```bash
git add montecarlo/core/export.py tests/test_export.py app.py README.md
git commit -m "feat: excel summary export and README"
```
