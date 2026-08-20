"""Converting durations into units and calendar dates.

Public holidays are not modelled; the interface says so out loud.
"""
import math
from datetime import date, timedelta

DAYS = "days"
WEEKS = "weeks"
HOURS = "hours"
UNITS = (DAYS, WEEKS, HOURS)
DEFAULT_HOURS_PER_DAY = 8.0


def to_working_days(
    duration: float,
    unit: str,
    days_per_week: int,
    hours_per_day: float = DEFAULT_HOURS_PER_DAY,
) -> float:
    """Express a duration in working days.

    The working week is shared with the calendar conversion so the two
    settings cannot drift apart.

    Args:
        duration: the estimate as written in the source file.
        unit: "days", "weeks" or "hours".
        days_per_week: 5 or 7.
        hours_per_day: length of a working day, used only when unit is
            "hours".
    """
    if unit == DAYS:
        return float(duration)
    if unit == WEEKS:
        return float(duration) * days_per_week
    if unit == HOURS:
        return float(duration) / hours_per_day
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
