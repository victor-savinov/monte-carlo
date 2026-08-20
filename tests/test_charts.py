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
    # Labels render in the board's uppercase style; compare case-insensitively.
    text = " ".join(t.get_text() for t in figure.axes[0].texts).upper()
    assert "22 FEB 2027" in text


def test_the_unit_label_reaches_the_axis(totals, pctls):
    axes = histogram(totals, pctls, unit_label="weeks").axes[0]
    assert "weeks" in axes.get_xlabel().lower()


def test_close_labels_are_staggered_not_overlapping(totals):
    """Regression: PLAN and P50 landing 3 days apart used to print on top
    of each other."""
    close_pctls = {50: 193.0, 85: 207.0, 95: 215.0}
    figure = histogram(totals, close_pctls, baseline=190.0,
                       baseline_probability=40.0)
    axes = figure.axes[0]
    texts = [t for t in axes.texts if t.get_text()]
    ys_at_baseline_x = {round(t.get_position()[1], 3) for t in texts
                        if abs(t.get_position()[0] - 190.0) < 5
                        or abs(t.get_position()[0] - 193.0) < 5}
    # The two close labels must occupy different vertical rows.
    assert len(ys_at_baseline_x) >= 2


def test_png_export_produces_bytes(totals, pctls):
    data = figure_to_png_bytes(histogram(totals, pctls))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5000
