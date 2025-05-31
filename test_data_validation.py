#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas>=2.0.0",
#     "pytz",
#     "rich",
#     "tabulate",
# ]
# ///

"""
Test script to validate JSONL data loading and check for timezone/time issues,
particularly around the "Half-credit was reached" calculations.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import pandas as pd
from tabulate import tabulate

console = Console()

class CreditWindow:
    """Represents a 5-hour credit window"""
    
    # Percentage of Sonnet messages that indicates we've reached half credit
    HALF_CREDIT_SONNET_PERCENTAGE = 0.80  # 80% of messages are Sonnet
    HALF_CREDIT_WINDOW_SIZE = 10  # Check the last N messages
    
    def __init__(self, start_time: datetime):
        self.start_time = start_time
        self.end_time = None
        self.messages = []
        self.opus_messages = []
        self.sonnet_messages = []
        self.half_credit_time = None
        self.half_credit_reached = False
        self.total_cost = 0.0
        self.opus_cost = 0.0
        self.sonnet_cost = 0.0
        
    def add_message(self, timestamp: datetime, model: str, cost: float, entry: dict):
        """Add a message to the window"""
        self.messages.append({
            'timestamp': timestamp,
            'model': model,
            'cost': cost,
            'entry': entry
        })
        self.total_cost += cost
        
        if model and 'opus' in model.lower():
            self.opus_messages.append(self.messages[-1])
            self.opus_cost += cost
        elif model and 'sonnet' in model.lower():
            self.sonnet_messages.append(self.messages[-1])
            self.sonnet_cost += cost
            
            # Check if Sonnet messages dominate in the last N messages (>80%)
            # This indicates we've switched from Opus to Sonnet due to credit limits
            if not self.half_credit_reached and len(self.messages) >= self.HALF_CREDIT_WINDOW_SIZE:
                # Get the last N messages
                recent_messages = self.messages[-self.HALF_CREDIT_WINDOW_SIZE:]
                sonnet_count = sum(1 for msg in recent_messages if 'sonnet' in msg.get('model', '').lower())
                sonnet_percentage = sonnet_count / self.HALF_CREDIT_WINDOW_SIZE
                
                if sonnet_percentage > self.HALF_CREDIT_SONNET_PERCENTAGE:
                    self.half_credit_reached = True
                    self.half_credit_time = timestamp
                    self.half_credit_cost = self.total_cost
                
    def close(self, end_time: datetime):
        """Close the window"""
        self.end_time = end_time
        self.duration = (end_time - self.start_time).total_seconds() / 3600  # hours

def load_jsonl_data(base_dir: str = "~/.claude/projects") -> List[dict]:
    """Load all JSONL files from the Claude projects directory"""
    base_path = Path(base_dir).expanduser()
    entries = []
    file_count = 0
    
    if not base_path.exists():
        console.print(f"[red]Directory not found: {base_path}[/red]")
        return entries
    
    console.print(f"[cyan]Scanning {base_path} for JSONL files...[/cyan]")
    
    for jsonl_file in base_path.rglob("*.jsonl"):
        file_count += 1
        try:
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        entry['_source_file'] = str(jsonl_file)
                        entry['_line_number'] = line_num
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]Skipping malformed JSON in {jsonl_file}:{line_num} - {e}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error reading {jsonl_file}: {e}[/red]")
    
    console.print(f"[green]Loaded {len(entries)} entries from {file_count} files[/green]")
    return entries

def parse_timestamp(timestamp_str: str) -> Tuple[datetime, datetime]:
    """Parse timestamp and return both UTC and local datetime objects"""
    # Handle ISO format with Z suffix
    if timestamp_str.endswith('Z'):
        utc_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    else:
        utc_dt = datetime.fromisoformat(timestamp_str)
    
    # Ensure UTC timezone
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    else:
        utc_dt = utc_dt.astimezone(timezone.utc)
    
    # Convert to local timezone
    local_tz = datetime.now().astimezone().tzinfo
    local_dt = utc_dt.astimezone(local_tz)
    
    return utc_dt, local_dt

def detect_windows(entries: List[dict]) -> List[CreditWindow]:
    """Detect 5-hour credit windows from entries"""
    # Sort entries by timestamp
    sorted_entries = sorted(entries, key=lambda x: x.get('timestamp', ''))
    
    windows = []
    current_window = None
    last_timestamp = None
    
    for entry in sorted_entries:
        timestamp_str = entry.get('timestamp')
        if not timestamp_str:
            continue
            
        utc_dt, local_dt = parse_timestamp(timestamp_str)
        model = entry.get('message', {}).get('model', '')
        cost = entry.get('costUSD', 0.0)
        
        # Check if we need to start a new window
        if last_timestamp is None or (local_dt - last_timestamp).total_seconds() > 5 * 3600:
            # Close previous window
            if current_window:
                current_window.close(last_timestamp)
                windows.append(current_window)
            # Start new window
            current_window = CreditWindow(local_dt)
        
        # Add message to current window
        if current_window:
            current_window.add_message(local_dt, model, cost, entry)
        
        last_timestamp = local_dt
    
    # Close final window
    if current_window and last_timestamp:
        current_window.close(last_timestamp)
        windows.append(current_window)
    
    return windows

def validate_timestamps(entries: List[dict]) -> Dict[str, any]:
    """Validate timestamps and check for timezone issues"""
    issues = {
        'missing_timestamps': 0,
        'invalid_timestamps': 0,
        'timezone_conversions': [],
        'suspicious_times': [],
        'timestamp_examples': []
    }
    
    for entry in entries[:10]:  # Check first 10 entries
        timestamp_str = entry.get('timestamp')
        if not timestamp_str:
            issues['missing_timestamps'] += 1
            continue
            
        try:
            utc_dt, local_dt = parse_timestamp(timestamp_str)
            
            example = {
                'raw': timestamp_str,
                'utc': utc_dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'local': local_dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'timezone': str(local_dt.tzinfo)
            }
            issues['timestamp_examples'].append(example)
            
            # Check for suspicious times (e.g., exact midnight)
            if local_dt.hour == 0 and local_dt.minute < 5:
                issues['suspicious_times'].append({
                    'timestamp': timestamp_str,
                    'local_time': local_dt.strftime('%H:%M:%S'),
                    'file': entry.get('_source_file', 'unknown')
                })
                
        except Exception as e:
            issues['invalid_timestamps'] += 1
            
    return issues

def analyze_half_credit_times(windows: List[CreditWindow]) -> Dict[str, any]:
    """Analyze half-credit reached times for issues"""
    analysis = {
        'total_windows': len(windows),
        'windows_with_half_credit': 0,
        'half_credit_times': [],
        'suspicious_patterns': [],
        'high_sonnet_percentage_windows': []
    }
    
    for i, window in enumerate(windows):
        # Check for high Sonnet percentage windows
        if len(window.messages) > 0:
            sonnet_percentage = len(window.sonnet_messages) / len(window.messages)
            if sonnet_percentage > 0.5:  # Windows with >50% Sonnet (but not necessarily half-credit)
                analysis['high_sonnet_percentage_windows'].append({
                    'window_index': i,
                    'window_start': window.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_messages': len(window.messages),
                    'sonnet_messages': len(window.sonnet_messages),
                    'sonnet_percentage': f"{sonnet_percentage*100:.1f}%",
                    'reached_half_credit': window.half_credit_reached
                })
        
        if window.half_credit_reached:
            analysis['windows_with_half_credit'] += 1
            
            time_info = {
                'window_index': i,
                'window_start': window.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'half_credit_time': window.half_credit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'half_credit_time_only': window.half_credit_time.strftime('%H:%M'),
                'hours_to_half_credit': (window.half_credit_time - window.start_time).total_seconds() / 3600,
                'opus_messages': len(window.opus_messages),
                'sonnet_messages': len(window.sonnet_messages),
                'total_cost_at_half': window.half_credit_cost
            }
            analysis['half_credit_times'].append(time_info)
            
            # Check for suspicious patterns
            if window.half_credit_time.strftime('%H:%M') == '00:37':
                analysis['suspicious_patterns'].append({
                    'pattern': 'Exact 00:37 time',
                    'window': time_info
                })
            
            # Check if half credit was reached very quickly
            if time_info['hours_to_half_credit'] < 0.1:  # Less than 6 minutes
                analysis['suspicious_patterns'].append({
                    'pattern': 'Half credit reached very quickly',
                    'window': time_info
                })
    
    return analysis

def main():
    """Main test function"""
    console.print(Panel.fit("🔍 Claude Code Data Validation Test", style="bold blue"))
    
    # Load data
    entries = load_jsonl_data()
    if not entries:
        console.print("[red]No data loaded. Exiting.[/red]")
        return
    
    # Validate timestamps
    console.print("\n[bold cyan]1. Timestamp Validation[/bold cyan]")
    timestamp_issues = validate_timestamps(entries)
    
    # Show timestamp examples
    if timestamp_issues['timestamp_examples']:
        table = Table(title="Timestamp Conversion Examples")
        table.add_column("Raw Timestamp", style="cyan")
        table.add_column("UTC Time", style="green")
        table.add_column("Local Time", style="yellow")
        table.add_column("Timezone", style="magenta")
        
        for ex in timestamp_issues['timestamp_examples'][:5]:
            table.add_row(ex['raw'], ex['utc'], ex['local'], ex['timezone'])
        
        console.print(table)
    
    # Report timestamp issues
    if timestamp_issues['missing_timestamps'] > 0:
        console.print(f"[yellow]⚠️  Missing timestamps: {timestamp_issues['missing_timestamps']}[/yellow]")
    if timestamp_issues['invalid_timestamps'] > 0:
        console.print(f"[red]❌ Invalid timestamps: {timestamp_issues['invalid_timestamps']}[/red]")
    if timestamp_issues['suspicious_times']:
        console.print(f"[yellow]⚠️  Suspicious times (near midnight): {len(timestamp_issues['suspicious_times'])}[/yellow]")
        for st in timestamp_issues['suspicious_times'][:3]:
            console.print(f"   - {st['local_time']} in {st['file']}")
    
    # Detect windows
    console.print("\n[bold cyan]2. Credit Window Detection[/bold cyan]")
    windows = detect_windows(entries)
    console.print(f"[green]✓ Detected {len(windows)} credit windows[/green]")
    
    # Analyze half-credit times
    console.print("\n[bold cyan]3. Half-Credit Time Analysis[/bold cyan]")
    half_credit_analysis = analyze_half_credit_times(windows)
    
    console.print(f"Windows with half-credit reached: {half_credit_analysis['windows_with_half_credit']}/{half_credit_analysis['total_windows']}")
    
    # Debug: Show windows with significant Sonnet usage
    windows_with_sonnet = sum(1 for w in windows if w.sonnet_messages)
    console.print(f"Windows with any Sonnet usage: {windows_with_sonnet}/{len(windows)}")
    
    # Debug: Show Sonnet percentages for all windows
    console.print("\n[bold]Window Sonnet Percentages:[/bold]")
    for i, w in enumerate(windows):
        if len(w.messages) > 0:
            pct = len(w.sonnet_messages) / len(w.messages) * 100
            console.print(f"  Window {i+1}: {pct:.1f}% Sonnet ({len(w.sonnet_messages)}/{len(w.messages)} messages)")
            
            # Check rolling window at end and find max
            if len(w.messages) >= 10:
                last_10 = w.messages[-10:]
                sonnet_in_last_10 = sum(1 for msg in last_10 if 'sonnet' in msg.get('model', '').lower())
                console.print(f"    Last 10 messages: {sonnet_in_last_10}/10 = {sonnet_in_last_10/10*100:.0f}% Sonnet")
                
                # Find max rolling window percentage
                max_sonnet_pct = 0
                max_at_idx = 0
                for j in range(10, len(w.messages) + 1):
                    window_msgs = w.messages[j-10:j]
                    sonnet_count = sum(1 for msg in window_msgs if 'sonnet' in msg.get('model', '').lower())
                    pct = sonnet_count / 10
                    if pct > max_sonnet_pct:
                        max_sonnet_pct = pct
                        max_at_idx = j
                if max_sonnet_pct > 0.5:
                    console.print(f"    Max 10-msg window: {max_sonnet_pct*100:.0f}% at message {max_at_idx}")
    
    # Show half-credit times
    if half_credit_analysis['half_credit_times']:
        console.print("\n[bold]Recent Half-Credit Times:[/bold]")
        recent_times = half_credit_analysis['half_credit_times'][-5:]  # Last 5
        
        table = Table(title="Half-Credit Reached Times")
        table.add_column("Window Start", style="cyan")
        table.add_column("Half-Credit Time", style="yellow")
        table.add_column("Time Only", style="magenta")
        table.add_column("Hours to Half", style="green")
        table.add_column("Opus → Sonnet", style="blue")
        
        for ht in recent_times:
            table.add_row(
                ht['window_start'],
                ht['half_credit_time'],
                ht['half_credit_time_only'],
                f"{ht['hours_to_half_credit']:.2f}h",
                f"{ht['opus_messages']} → {ht['sonnet_messages']}"
            )
        
        console.print(table)
    
    # Report suspicious patterns
    if half_credit_analysis['suspicious_patterns']:
        console.print("\n[bold red]⚠️  Suspicious Patterns Detected:[/bold red]")
        for pattern in half_credit_analysis['suspicious_patterns']:
            console.print(f"\n[yellow]Pattern: {pattern['pattern']}[/yellow]")
            window = pattern['window']
            console.print(f"  Window start: {window['window_start']}")
            console.print(f"  Half-credit: {window['half_credit_time']} ({window['half_credit_time_only']})")
            console.print(f"  Duration: {window['hours_to_half_credit']:.2f} hours")
    
    # Report high Sonnet percentage windows
    if half_credit_analysis['high_sonnet_percentage_windows']:
        console.print(f"\n[bold yellow]Windows with High Sonnet Usage:[/bold yellow]")
        console.print(f"Found {len(half_credit_analysis['high_sonnet_percentage_windows'])} windows with >50% Sonnet messages")
        console.print(f"Half-credit is triggered when Sonnet exceeds {CreditWindow.HALF_CREDIT_SONNET_PERCENTAGE*100:.0f}% of messages")
        
        table = Table(title="High Sonnet Usage Windows")
        table.add_column("Window Start", style="cyan")
        table.add_column("Total Messages", style="yellow")
        table.add_column("Sonnet Messages", style="magenta")
        table.add_column("Sonnet %", style="red")
        table.add_column("Half-Credit?", style="green")
        
        for window in half_credit_analysis['high_sonnet_percentage_windows'][:10]:  # Show first 10
            table.add_row(
                window['window_start'],
                str(window['total_messages']),
                str(window['sonnet_messages']),
                window['sonnet_percentage'],
                "Yes" if window['reached_half_credit'] else "No"
            )
        
        console.print(table)
    
    # Additional diagnostics
    console.print("\n[bold cyan]4. Data Summary[/bold cyan]")
    
    # Get date range
    dates = []
    for entry in entries:
        ts = entry.get('timestamp')
        if ts:
            _, local_dt = parse_timestamp(ts)
            dates.append(local_dt)
    
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        console.print(f"Date range: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")
        console.print(f"Total days: {(max_date - min_date).days + 1}")
    
    # Model distribution
    model_counts = {}
    for entry in entries:
        model = entry.get('message', {}).get('model', 'unknown')
        model_counts[model] = model_counts.get(model, 0) + 1
    
    console.print("\n[bold]Model Distribution:[/bold]")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  {model}: {count} messages")
    
    console.print("\n[green]✅ Validation complete![/green]")

if __name__ == "__main__":
    main()