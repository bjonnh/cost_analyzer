#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "dash>=2.0.0",
#   "scikit-learn>=1.0.0",
#   "numpy>=1.21.0",
#   "dash-bootstrap-components>=1.0.0",
#   "rich>=13.0.0",
# ]
# ///

"""Trace window detection logic step by step."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from rich.console import Console

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from cost_analyzer.data_processor import load_raw_records_from_database
from cost_analyzer.window_detector import detect_windows

console = Console()

def trace_detection():
    """Trace window detection for debugging."""
    # Load records
    records = load_raw_records_from_database()
    
    if not records:
        console.print("[red]No records found[/red]")
        return
    
    # Get timezone
    local_tz = ZoneInfo('America/New_York')
    utc_tz = ZoneInfo('UTC')
    
    # Sort records
    sorted_records = sorted(records, key=lambda x: x['timestamp'])
    
    # Find the most recent gap
    console.print("[bold]Looking for most recent gap...[/bold]")
    gap_start_idx = None
    prev_time = None
    
    for i, record in enumerate(sorted_records):
        timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        if prev_time and timestamp - prev_time >= timedelta(hours=5):
            gap_start_idx = i
        prev_time = timestamp
    
    if gap_start_idx is None:
        console.print("[yellow]No gaps found[/yellow]")
        return
    
    # Show records around the gap
    console.print(f"\n[bold]Found gap at index {gap_start_idx}[/bold]")
    console.print("\nRecords around gap:")
    
    for i in range(max(0, gap_start_idx - 2), min(len(sorted_records), gap_start_idx + 5)):
        record = sorted_records[i]
        timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        local_time = timestamp.astimezone(local_tz)
        
        marker = ""
        if i == gap_start_idx - 1:
            marker = " <- Last before gap"
        elif i == gap_start_idx:
            marker = " <- First after gap (should start window)"
        
        console.print(f"  [{i}] {local_time.strftime('%H:%M:%S')} - ${record.get('costUSD', 0):.2f}{marker}")
    
    # Now detect windows and see what happens
    console.print("\n[bold]Running window detection...[/bold]")
    windows = detect_windows(sorted_records)
    
    # Find the window that should contain the message after the gap
    target_time = datetime.fromisoformat(sorted_records[gap_start_idx]['timestamp'].replace('Z', '+00:00'))
    
    console.print(f"\nLooking for window containing {target_time.astimezone(local_tz).strftime('%H:%M:%S')}...")
    
    found_window = None
    for window in windows:
        if any(msg['timestamp'] == target_time for msg in window.messages):
            found_window = window
            break
    
    if found_window:
        console.print(f"\n[green]Found window:[/green]")
        console.print(f"  Start: {found_window.start_time.astimezone(local_tz).strftime('%H:%M:%S')}")
        console.print(f"  End: {found_window.end_time.astimezone(local_tz).strftime('%H:%M:%S')}")
        console.print(f"  Messages: {len(found_window.messages)}")
        
        console.print("\n  First 3 messages in window:")
        for i, msg in enumerate(found_window.messages[:3]):
            msg_time = msg['timestamp'].astimezone(local_tz).strftime('%H:%M:%S')
            console.print(f"    [{i}] {msg_time} - ${msg['cost']:.2f}")
            
        if found_window.start_time != target_time:
            console.print(f"\n[red]ERROR: Window starts at {found_window.start_time.astimezone(local_tz).strftime('%H:%M:%S')} but should start at {target_time.astimezone(local_tz).strftime('%H:%M:%S')}[/red]")
    else:
        console.print(f"\n[red]ERROR: No window found containing the target message[/red]")

if __name__ == "__main__":
    trace_detection()