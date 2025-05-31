#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["rich>=13.0.0"]
# ///

"""Debug window detection to see why windows don't start at the right time."""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from rich.console import Console
from rich.table import Table

console = Console()

def debug_windows():
    """Debug window detection logic."""
    db_path = Path.home() / '.claude' / 'usage_data.db'
    
    if not db_path.exists():
        console.print("[red]Database not found[/red]")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all records ordered by timestamp
    cursor.execute("""
        SELECT timestamp, model, costUSD, uuid 
        FROM usage_records 
        WHERE costUSD > 0
        ORDER BY timestamp
    """)
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        console.print("[yellow]No records found[/yellow]")
        return
    
    # Convert to local timezone
    local_tz = ZoneInfo('America/New_York')  # Adjust to your timezone
    utc_tz = ZoneInfo('UTC')
    
    console.print(f"\n[bold]Analyzing {len(records)} records[/bold]")
    
    # Process records and look for gaps
    prev_time = None
    gap_count = 0
    
    table = Table(title="Messages Around Gaps (showing first 10)")
    table.add_column("Time (Local)", style="cyan")
    table.add_column("Gap from Previous", style="yellow")
    table.add_column("Model", style="green")
    table.add_column("Cost", style="magenta")
    
    shown = 0
    for i, (timestamp_str, model, cost, uuid) in enumerate(records):
        # Parse timestamp
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=utc_tz)
        local_time = timestamp.astimezone(local_tz)
        
        if prev_time:
            gap = timestamp - prev_time
            if gap >= timedelta(hours=5) and shown < 10:
                # Show context around gap
                if i > 0:
                    prev_rec = records[i-1]
                    prev_ts = datetime.fromisoformat(prev_rec[0].replace('Z', '+00:00'))
                    prev_ts = prev_ts.replace(tzinfo=utc_tz).astimezone(local_tz)
                    table.add_row(
                        prev_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "Before gap",
                        prev_rec[1],
                        f"${prev_rec[2]:.2f}"
                    )
                
                table.add_row(
                    local_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{gap.total_seconds()/3600:.1f} hours",
                    model,
                    f"${cost:.2f}"
                )
                
                # Show next message too
                if i < len(records) - 1:
                    next_rec = records[i+1]
                    next_ts = datetime.fromisoformat(next_rec[0].replace('Z', '+00:00'))
                    next_ts = next_ts.replace(tzinfo=utc_tz).astimezone(local_tz)
                    next_gap = next_ts - timestamp
                    table.add_row(
                        next_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{next_gap.total_seconds()/60:.1f} min",
                        next_rec[1],
                        f"${next_rec[2]:.2f}"
                    )
                
                table.add_row("---", "---", "---", "---")
                gap_count += 1
                shown += 1
        
        prev_time = timestamp
    
    console.print(table)
    console.print(f"\n[bold]Found {gap_count} gaps of 5+ hours[/bold]")
    
    # Now let's trace through the window detection logic for a specific gap
    console.print("\n[bold]Simulating window detection for the most recent gap:[/bold]")
    
    # Find the most recent gap
    prev_time = None
    recent_gap_index = None
    
    for i, (timestamp_str, model, cost, uuid) in enumerate(records):
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=utc_tz)
        
        if prev_time:
            gap = timestamp - prev_time
            if gap >= timedelta(hours=5):
                recent_gap_index = i
        
        prev_time = timestamp
    
    if recent_gap_index:
        # Show what happens around this gap
        console.print(f"\n[yellow]Gap found at index {recent_gap_index}[/yellow]")
        
        # Show messages around the gap
        for j in range(max(0, recent_gap_index - 2), min(len(records), recent_gap_index + 3)):
            timestamp_str, model, cost, uuid = records[j]
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=utc_tz)
            local_time = timestamp.astimezone(local_tz)
            
            marker = " <-- Gap starts here" if j == recent_gap_index else ""
            console.print(f"  [{j}] {local_time.strftime('%H:%M:%S')} - {model} - ${cost:.2f}{marker}")

if __name__ == "__main__":
    debug_windows()