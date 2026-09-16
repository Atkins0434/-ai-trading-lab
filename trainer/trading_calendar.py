from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable


# Exchange-wide closures that are not described by the regular holiday rules.
EXTRAORDINARY_CLOSURES = {
    date(2018, 12, 5),  # National Day of Mourning for George H. W. Bush
    date(2025, 1, 9),  # National Day of Mourning for Jimmy Carter
}


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    current = date(year, month, 1)
    offset = (weekday - current.weekday()) % 7
    return current + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def exchange_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    # New Year's Day can be observed in the preceding calendar year.
    holidays.add(_observed(date(year + 1, 1, 1)))
    return holidays


def generate_trading_dates(
    start_date: str,
    end_date: str,
    *,
    extra_closures: Iterable[str] = (),
    extra_sessions: Iterable[str] = (),
) -> list[str]:
    """Generate deterministic full-session U.S. equity trading dates."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    closures = EXTRAORDINARY_CLOSURES | {
        date.fromisoformat(value) for value in extra_closures
    }
    sessions = {date.fromisoformat(value) for value in extra_sessions}
    holidays: set[date] = set()
    for year in range(start.year - 1, end.year + 2):
        holidays.update(exchange_holidays(year))

    current = start
    while current <= end:
        if current in sessions or (
            current.weekday() < 5
            and current not in holidays
            and current not in closures
        ):
            sessions.add(current)
        current += timedelta(days=1)
    return [value.isoformat() for value in sorted(sessions) if start <= value <= end]
