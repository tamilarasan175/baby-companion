"""Time zone helpers. Timestamps are stored as naive UTC in the database."""
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "America/Edmonton")


def get_tz(name: str = None) -> ZoneInfo:
    for candidate in (name, DEFAULT_TZ, "UTC"):
        if candidate:
            try:
                return ZoneInfo(candidate)
            except (ZoneInfoNotFoundError, ValueError):
                continue
    return ZoneInfo("UTC")


def to_local(ts: datetime, tz: ZoneInfo) -> datetime:
    return ts.replace(tzinfo=timezone.utc).astimezone(tz)


def utc_iso(ts: datetime) -> str:
    """ISO string with an explicit UTC offset, so browsers don't read it as local time."""
    return ts.replace(tzinfo=timezone.utc).isoformat()


def local_midnight_utc(tz: ZoneInfo, days_ago: int = 0) -> datetime:
    """Naive-UTC datetime for midnight (local) `days_ago` days before today."""
    local_today = datetime.now(tz).date() - timedelta(days=days_ago)
    local_midnight = datetime(local_today.year, local_today.month, local_today.day, tzinfo=tz)
    return local_midnight.astimezone(timezone.utc).replace(tzinfo=None)
