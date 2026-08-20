"""Tests for unit conversion and calendar arithmetic."""
from datetime import date

import pytest

from montecarlo.core.dates import to_date, to_working_days


def test_days_pass_through_unchanged():
    assert to_working_days(10.0, "days", 5) == pytest.approx(10.0)


def test_weeks_convert_using_the_working_week():
    assert to_working_days(2.0, "weeks", 5) == pytest.approx(10.0)
    assert to_working_days(2.0, "weeks", 7) == pytest.approx(14.0)


def test_hours_convert_using_hours_per_day():
    assert to_working_days(16.0, "hours", 5, hours_per_day=8.0) == pytest.approx(2.0)
    assert to_working_days(10.0, "hours", 5, hours_per_day=5.0) == pytest.approx(2.0)


def test_hours_default_to_an_eight_hour_day():
    assert to_working_days(4.0, "hours", 5) == pytest.approx(0.5)


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
