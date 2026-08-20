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
