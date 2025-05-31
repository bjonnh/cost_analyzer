#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["rich>=13.0.0"]
# ///

"""Check timezone handling."""

from datetime import datetime
from zoneinfo import ZoneInfo
from rich.console import Console

console = Console()

# Sample timestamp from the data
sample_timestamp = "2025-05-31T20:56:56Z"

# Parse it
dt = datetime.fromisoformat(sample_timestamp.replace('Z', '+00:00'))
console.print(f"Parsed datetime: {dt}")
console.print(f"Timezone info: {dt.tzinfo}")

# Convert to Eastern Time
eastern = ZoneInfo('America/New_York')
dt_eastern = dt.astimezone(eastern)
console.print(f"Eastern Time: {dt_eastern}")
console.print(f"Eastern Time (time only): {dt_eastern.strftime('%H:%M:%S')}")

# Check what 16:56:56 Eastern would be in UTC
dt_eastern_1656 = datetime(2025, 5, 31, 16, 56, 56, tzinfo=eastern)
dt_utc_from_eastern = dt_eastern_1656.astimezone(ZoneInfo('UTC'))
console.print(f"\n16:56:56 Eastern is {dt_utc_from_eastern.strftime('%H:%M:%S')} UTC")