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

"""Check the current window and its messages."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from rich.console import Console
from rich.table import Table

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from cost_analyzer.data_processor import get_window_analysis
from cost_analyzer.database import load_credit_windows

console = Console()

def check_current_window():
    """Check current window details."""
    # Get window analysis
    window_data = get_window_analysis()
    windows = window_data.get('windows', [])
    
    if not windows:
        console.print("[red]No windows found[/red]")
        return
    
    # Get timezone
    local_tz = ZoneInfo('America/New_York')
    utc_tz = ZoneInfo('UTC')
    now = datetime.now(tz=local_tz)
    
    # Find current window
    current_window = None
    for window in windows:
        # Convert timestamps
        if isinstance(window['start_time'], str):
            start = datetime.fromisoformat(window['start_time'].replace('Z', '+00:00'))
            if start.tzinfo is None:
                start = start.replace(tzinfo=utc_tz)
            start = start.astimezone(local_tz)
        else:
            start = window['start_time']
            
        if isinstance(window['end_time'], str):
            end = datetime.fromisoformat(window['end_time'].replace('Z', '+00:00'))
            if end.tzinfo is None:
                end = end.replace(tzinfo=utc_tz)
            end = end.astimezone(local_tz)
        else:
            end = window['end_time']
        
        if start <= now < end:
            current_window = window
            current_window['start_time'] = start
            current_window['end_time'] = end
            break
    
    if current_window:
        console.print(f"[bold green]Current Window Found[/bold green]")
        console.print(f"Start: {current_window['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"End: {current_window['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Total Cost: ${current_window.get('total_cost', 0):.2f}")
        console.print(f"Messages: {current_window.get('total_messages', 0)}")
        
        # Show first and last few messages
        messages = current_window.get('messages', [])
        if messages:
            console.print(f"\n[bold]First 5 messages:[/bold]")
            table = Table()
            table.add_column("Time", style="cyan")
            table.add_column("Model", style="green")
            table.add_column("Cost", style="yellow")
            
            for msg in messages[:5]:
                if isinstance(msg['timestamp'], str):
                    ts = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=utc_tz)
                    ts = ts.astimezone(local_tz)
                else:
                    ts = msg['timestamp']
                
                table.add_row(
                    ts.strftime("%H:%M:%S"),
                    msg.get('model', 'unknown'),
                    f"${msg.get('cost', 0):.2f}"
                )
            
            console.print(table)
            
            if len(messages) > 5:
                console.print(f"\n... {len(messages) - 10} more messages ...\n")
                
                console.print(f"[bold]Last 5 messages:[/bold]")
                table2 = Table()
                table2.add_column("Time", style="cyan")
                table2.add_column("Model", style="green")
                table2.add_column("Cost", style="yellow")
                
                for msg in messages[-5:]:
                    if isinstance(msg['timestamp'], str):
                        ts = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=utc_tz)
                        ts = ts.astimezone(local_tz)
                    else:
                        ts = msg['timestamp']
                    
                    table2.add_row(
                        ts.strftime("%H:%M:%S"),
                        msg.get('model', 'unknown'),
                        f"${msg.get('cost', 0):.2f}"
                    )
                
                console.print(table2)
    else:
        console.print("[yellow]No current window active[/yellow]")
        
        # Show the most recent window
        if windows:
            console.print("\n[bold]Most recent window:[/bold]")
            recent = windows[-1]
            
            if isinstance(recent['start_time'], str):
                start = datetime.fromisoformat(recent['start_time'].replace('Z', '+00:00'))
                if start.tzinfo is None:
                    start = start.replace(tzinfo=utc_tz)
                start = start.astimezone(local_tz)
            else:
                start = recent['start_time']
                
            console.print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"Messages: {recent.get('total_messages', 0)}")
            console.print(f"Cost: ${recent.get('total_cost', 0):.2f}")

if __name__ == "__main__":
    check_current_window()