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
