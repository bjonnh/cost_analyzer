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
#   "pytest>=7.0.0",
# ]
# ///

"""Test window analysis functionality with $10 minimum threshold."""

import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytest
from rich.console import Console
from rich.table import Table

# Add the project directory to Python path
sys.path.insert(0, '/home/bjo/Software/Anthropic/cost_analyzer')

from cost_analyzer.window_detector import CreditWindow, detect_windows, analyze_windows

console = Console()


def create_test_records(base_time: datetime) -> List[Dict[str, Any]]:
    """Create test records for window detection."""
    records = []
    
    # Window 1: $5 total (below threshold)
    for i in range(5):
        records.append({
            'timestamp': (base_time + timedelta(minutes=i*10)).isoformat() + 'Z',
            'model': 'claude-3.5-opus',
            'costUSD': 1.0,
            'total_tokens': 1000,
            'uuid': f'uuid-1-{i}'
        })
    
    # Gap of 6 hours - new window
    base_time = base_time + timedelta(hours=6)
    
    # Window 2: $18 total (above threshold)
    # Need at least 10 messages total for half-credit detection
    # First 5 opus, then 7 sonnet (last 10 will have at least 5 sonnet = 50%, not enough)
    # Let's add more messages to ensure we get >80% sonnet in last 10
    for i in range(12):
        # First 3 opus, then rest sonnet
        # Last 10 messages will be: 1 opus + 9 sonnet = 90% sonnet (>80%)
        model = 'claude-3.5-opus' if i < 3 else 'claude-3.5-sonnet'
        records.append({
            'timestamp': (base_time + timedelta(minutes=i*10)).isoformat() + 'Z',
            'model': model,
            'costUSD': 1.5,
            'total_tokens': 1500,
            'uuid': f'uuid-2-{i}'
        })
    
    # Gap of 6 hours - new window
    base_time = base_time + timedelta(hours=6)
    
    # Window 3: $50 total (well above threshold)
    for i in range(20):
        model = 'claude-3.5-opus' if i < 8 else 'claude-3.5-sonnet'
        records.append({
            'timestamp': (base_time + timedelta(minutes=i*10)).isoformat() + 'Z',
            'model': model,
            'costUSD': 2.5,
            'total_tokens': 2500,
            'uuid': f'uuid-3-{i}'
        })
    
    return records


def test_window_detection():
    """Test window detection functionality."""
    console.print("\n[bold blue]Testing Window Detection[/bold blue]")
    
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = create_test_records(base_time)
    
    # Detect windows
    windows = detect_windows(records)
    
    # Verify we have 3 windows
    assert len(windows) == 3, f"Expected 3 windows, got {len(windows)}"
    console.print(f"✅ Detected {len(windows)} windows correctly")
    
    # Verify window costs
    assert abs(windows[0].total_cost - 5.0) < 0.01, f"Window 1 cost: {windows[0].total_cost}"
    assert abs(windows[1].total_cost - 18.0) < 0.01, f"Window 2 cost: {windows[1].total_cost}"  # 12 * 1.5 = 18
    assert abs(windows[2].total_cost - 50.0) < 0.01, f"Window 3 cost: {windows[2].total_cost}"
    console.print("✅ Window costs calculated correctly")
    
    # Verify half-credit detection
    console.print(f"\nWindow 1: {windows[0].opus_messages} opus, {windows[0].sonnet_messages} sonnet, half_credit={windows[0].reached_half_credit}")
    console.print(f"Window 2: {len(windows[1].opus_messages)} opus, {len(windows[1].sonnet_messages)} sonnet, half_credit={windows[1].reached_half_credit}")
    console.print(f"Window 3: {len(windows[2].opus_messages)} opus, {len(windows[2].sonnet_messages)} sonnet, half_credit={windows[2].reached_half_credit}")
    
    # Debug: check last 10 messages of window 2
    if len(windows[1].messages) >= 10:
        last_10 = windows[1].messages[-10:]
        sonnet_count = sum(1 for msg in last_10 if 'sonnet' in msg.get('model', '').lower())
        console.print(f"Window 2 last 10 messages: {sonnet_count} sonnet out of 10")
    
    assert not windows[0].reached_half_credit, "Window 1 should not reach half credit"
    assert windows[1].reached_half_credit, "Window 2 should reach half credit"
    assert windows[2].reached_half_credit, "Window 3 should reach half credit"
    console.print("✅ Half-credit detection working correctly")


def test_window_analysis_filtering():
    """Test window analysis with $10 minimum threshold."""
    console.print("\n[bold blue]Testing Window Analysis Filtering[/bold blue]")
    
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    records = create_test_records(base_time)
    windows = detect_windows(records)
    
    # Test without filtering
    analysis_no_filter = analyze_windows(windows, min_window_cost=0.0)
    
    console.print("\n[yellow]Analysis without filtering:[/yellow]")
    table = Table(title="All Windows")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Windows", str(analysis_no_filter['total_windows']))
    table.add_row("Filtered Windows", str(analysis_no_filter['filtered_windows']))
    table.add_row("Windows Reached Half", str(analysis_no_filter['windows_reached_half_credit']))
    table.add_row("Avg Cost/Window", f"${analysis_no_filter['avg_cost_per_window']:.2f}")
    table.add_row("Avg Hours to Half", f"{analysis_no_filter['avg_hours_to_half_credit']:.2f}" if analysis_no_filter['avg_hours_to_half_credit'] else "N/A")
    
    console.print(table)
    
    # Test with $10 filtering
    analysis_filtered = analyze_windows(windows, min_window_cost=10.0)
    
    console.print("\n[yellow]Analysis with $10 minimum:[/yellow]")
    table2 = Table(title="Windows > $10")
    table2.add_column("Metric", style="cyan")
    table2.add_column("Value", style="green")
    
    table2.add_row("Total Windows", str(analysis_filtered['total_windows']))
    table2.add_row("Filtered Windows", str(analysis_filtered['filtered_windows']))
    table2.add_row("Windows Reached Half", str(analysis_filtered['windows_reached_half_credit']))
    table2.add_row("Avg Cost/Window", f"${analysis_filtered['avg_cost_per_window']:.2f}")
    table2.add_row("Avg Hours to Half", f"{analysis_filtered['avg_hours_to_half_credit']:.2f}" if analysis_filtered['avg_hours_to_half_credit'] else "N/A")
    
    console.print(table2)
    
    # Verify filtering
    assert analysis_no_filter['total_windows'] == 3, "Should have 3 total windows"
    assert analysis_no_filter['filtered_windows'] == 3, "Should have 3 filtered windows without threshold"
    assert analysis_filtered['filtered_windows'] == 2, "Should have 2 filtered windows with $10 threshold"
    
    # Verify statistics are calculated only on filtered windows
    expected_avg_cost = (18.0 + 50.0) / 2  # Only windows 2 and 3
    assert abs(analysis_filtered['avg_cost_per_window'] - expected_avg_cost) < 0.01, \
        f"Expected avg cost {expected_avg_cost}, got {analysis_filtered['avg_cost_per_window']}"
    
    console.print("\n✅ Window filtering working correctly")
    console.print("✅ Statistics calculated only on windows > $10")


def test_window_time_calculations():
    """Test time-based calculations in window analysis."""
    console.print("\n[bold blue]Testing Window Time Calculations[/bold blue]")
    
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    # Create a window that reaches half credit after 2 hours
    window = CreditWindow(base_time)
    
    # Add Opus messages for first 2 hours
    for i in range(12):  # 12 messages, 10 minutes apart = 2 hours
        window.add_message(
            timestamp=base_time + timedelta(minutes=i*10),
            model='claude-3.5-opus',
            cost=5.0,
            tokens=1000,
            uuid=f'uuid-{i}'
        )
    
    # Add Sonnet messages (should trigger half credit)
    for i in range(10):
        window.add_message(
            timestamp=base_time + timedelta(hours=2, minutes=i*5),
            model='claude-3.5-sonnet',
            cost=2.0,
            tokens=500,
            uuid=f'uuid-sonnet-{i}'
        )
    
    # Verify half credit detection
    assert window.reached_half_credit, "Window should reach half credit"
    
    # Calculate time to half credit
    time_to_half = (window.half_credit_time - window.start_time).total_seconds() / 3600
    console.print(f"Time to half credit: {time_to_half:.2f} hours")
    
    # Should be around 2-3 hours (need 10 messages before half-credit detection kicks in)
    assert 2.0 <= time_to_half <= 3.0, f"Expected 2-3 hours to half credit, got {time_to_half:.2f}"
    
    console.print("✅ Time calculations working correctly")


def test_edge_cases():
    """Test edge cases in window analysis."""
    console.print("\n[bold blue]Testing Edge Cases[/bold blue]")
    
    # Test empty windows
    analysis = analyze_windows([], min_window_cost=10.0)
    assert analysis['total_windows'] == 0
    assert analysis['filtered_windows'] == 0
    console.print("✅ Empty windows handled correctly")
    
    # Test windows with no messages reaching half credit
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    window = CreditWindow(base_time)
    
    # Add only Opus messages
    for i in range(20):
        window.add_message(
            timestamp=base_time + timedelta(minutes=i*5),
            model='claude-3.5-opus',
            cost=1.0,
            tokens=1000,
            uuid=f'uuid-{i}'
        )
    
    analysis = analyze_windows([window], min_window_cost=10.0)
    assert analysis['avg_hours_to_half_credit'] is None, "Should have no average time to half credit"
    console.print("✅ Windows without half credit handled correctly")
    
    # Test window exactly at threshold
    window2 = CreditWindow(base_time)
    for i in range(10):
        window2.add_message(
            timestamp=base_time + timedelta(minutes=i*5),
            model='claude-3.5-opus',
            cost=1.0,
            tokens=1000,
            uuid=f'uuid-{i}'
        )
    
    analysis = analyze_windows([window2], min_window_cost=10.0)
    assert analysis['filtered_windows'] == 1, "Window at exactly $10 should be included"
    console.print("✅ Window at exact threshold handled correctly")


if __name__ == "__main__":
    console.print("[bold magenta]Window Analysis Test Suite[/bold magenta]")
    console.print("=" * 50)
    
    try:
        test_window_detection()
        test_window_analysis_filtering()
        test_window_time_calculations()
        test_edge_cases()
        
        console.print("\n[bold green]All tests passed! ✅[/bold green]")
    except AssertionError as e:
        console.print(f"\n[bold red]Test failed: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)