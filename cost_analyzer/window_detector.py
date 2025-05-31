"""Window detection module for Claude Code 5-hour credit windows."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict


class CreditWindow:
    """Represents a 5-hour credit window."""
    
    # Percentage of Sonnet messages that indicates we've reached half credit
    HALF_CREDIT_SONNET_PERCENTAGE = 0.80  # 80% of messages are Sonnet
    HALF_CREDIT_WINDOW_SIZE = 10  # Check the last N messages
    
    def __init__(self, start_time: datetime):
        self.start_time = start_time
        self.end_time = start_time + timedelta(hours=5)
        self.messages = []
        self.opus_messages = []
        self.sonnet_messages = []
        self.total_cost = 0.0
        self.opus_cost = 0.0
        self.sonnet_cost = 0.0
        self.total_tokens = 0
        self.opus_tokens = 0
        self.sonnet_tokens = 0
        self.reached_half_credit = False
        self.half_credit_time = None
        self.half_credit_cost = None
        
    def add_message(self, timestamp: datetime, model: str, cost: float, tokens: int, uuid: str = None):
        """Add a message to the window."""
        self.messages.append({
            'timestamp': timestamp,
            'model': model,
            'cost': cost,
            'tokens': tokens,
            'uuid': uuid
        })
        
        self.total_cost += cost
        self.total_tokens += tokens
        
        if model and 'opus' in model.lower():
            self.opus_messages.append(self.messages[-1])
            self.opus_cost += cost
            self.opus_tokens += tokens
        elif model and 'sonnet' in model.lower():
            self.sonnet_messages.append(self.messages[-1])
            self.sonnet_cost += cost
            self.sonnet_tokens += tokens
            
            # Check if Sonnet messages dominate in the last N messages (>80%)
            # This indicates we've switched from Opus to Sonnet due to credit limits
            if not self.reached_half_credit and len(self.messages) >= self.HALF_CREDIT_WINDOW_SIZE:
                # Get the last N messages
                recent_messages = self.messages[-self.HALF_CREDIT_WINDOW_SIZE:]
                sonnet_count = sum(1 for msg in recent_messages if 'sonnet' in msg.get('model', '').lower())
                sonnet_percentage = sonnet_count / self.HALF_CREDIT_WINDOW_SIZE
                
                if sonnet_percentage > self.HALF_CREDIT_SONNET_PERCENTAGE:
                    self.reached_half_credit = True
                    self.half_credit_time = timestamp
                    self.half_credit_cost = self.total_cost
    
    def contains_time(self, timestamp: datetime) -> bool:
        """Check if a timestamp falls within this window."""
        return self.start_time <= timestamp < self.end_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert window to dictionary for analysis."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_hours': 5,
            'total_messages': len(self.messages),
            'opus_messages': len(self.opus_messages),
            'sonnet_messages': len(self.sonnet_messages),
            'total_cost': self.total_cost,
            'opus_cost': self.opus_cost,
            'sonnet_cost': self.sonnet_cost,
            'total_tokens': self.total_tokens,
            'opus_tokens': self.opus_tokens,
            'sonnet_tokens': self.sonnet_tokens,
            'reached_half_credit': self.reached_half_credit,
            'half_credit_time': self.half_credit_time,
            'half_credit_cost': self.half_credit_cost,
            'messages': self.messages
        }


def detect_windows(records: List[Dict[str, Any]]) -> List[CreditWindow]:
    """
    Detect 5-hour credit windows from usage records.
    
    Rules:
    1. A new window starts when there's been no activity for 5+ hours
    2. A window switch is detected when model changes from opus to sonnet (half credit reached)
    3. A window ends after 5 hours or when opus becomes available again
    
    Args:
        records: List of usage records with timestamp, model, cost, etc.
        
    Returns:
        List of detected CreditWindow objects
    """
    if not records:
        return []
    
    # Sort records by timestamp
    sorted_records = sorted(records, key=lambda x: x['timestamp'])
    
    windows = []
    current_window = None
    
    for record in sorted_records:
        timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        model = record.get('model', '')
        cost = record.get('costUSD', 0)
        tokens = record.get('total_tokens', 0)
        
        # Skip records without cost
        if cost <= 0:
            continue
        
        # Check if we need to start a new window
        if current_window is None:
            # First window
            current_window = CreditWindow(timestamp)
        else:
            # Check if this is a new window period
            # A new window starts if:
            # 1. We're outside the current 5-hour window period
            # 2. OR there's been a long gap (>5 hours) since any activity
            time_since_window_start = timestamp - current_window.start_time
            time_since_last_activity = timestamp - current_window.messages[-1]['timestamp'] if current_window.messages else timedelta(0)
            
            if time_since_window_start >= timedelta(hours=5):
                # We're past the 5-hour window period
                windows.append(current_window)
                current_window = CreditWindow(timestamp)
            elif time_since_last_activity >= timedelta(hours=5):
                # Long inactivity gap - credit should have reset
                windows.append(current_window)
                current_window = CreditWindow(timestamp)
            elif (current_window.reached_half_credit and 
                  model and 'opus' in model.lower() and 
                  timestamp > current_window.half_credit_time + timedelta(minutes=1)):
                # Opus is available again after using sonnet - new window
                windows.append(current_window)
                current_window = CreditWindow(timestamp)
        
        # Add message to current window
        uuid = record.get('uuid')
        current_window.add_message(timestamp, model, cost, tokens, uuid)
    
    # Don't forget the last window
    if current_window and current_window.messages:
        windows.append(current_window)
    
    return windows


def analyze_windows(windows: List[CreditWindow], min_window_cost: float = 0.0) -> Dict[str, Any]:
    """
    Analyze detected windows to provide statistics.
    
    Args:
        windows: List of CreditWindow objects
        min_window_cost: Minimum cost threshold for including windows in statistics
        
    Returns:
        Dictionary with window statistics
    """
    if not windows:
        return {
            'total_windows': 0,
            'filtered_windows': 0,
            'windows_data': []
        }
    
    windows_data = []
    for window in windows:
        window_dict = window.to_dict()
        
        # Calculate additional metrics
        if window.reached_half_credit and window.half_credit_time:
            time_to_half = (window.half_credit_time - window.start_time).total_seconds() / 3600
            window_dict['hours_to_half_credit'] = round(time_to_half, 2)
        else:
            window_dict['hours_to_half_credit'] = None
        
        # Calculate average cost per hour
        actual_duration = (window.messages[-1]['timestamp'] - window.start_time).total_seconds() / 3600
        window_dict['actual_duration_hours'] = round(actual_duration, 2)
        window_dict['avg_cost_per_hour'] = round(window.total_cost / max(actual_duration, 0.1), 2)
        
        windows_data.append(window_dict)
    
    # Filter windows for statistics based on minimum cost
    stats_windows = [w for w in windows if w.total_cost >= min_window_cost]
    
    # Overall statistics (using filtered windows)
    if stats_windows:
        total_cost = sum(w.total_cost for w in stats_windows)
        total_opus_cost = sum(w.opus_cost for w in stats_windows)
        total_sonnet_cost = sum(w.sonnet_cost for w in stats_windows)
        windows_reached_half = sum(1 for w in stats_windows if w.reached_half_credit)
        
        avg_cost_per_window = total_cost / len(stats_windows)
        avg_hours_to_half = []
        for w in stats_windows:
            if w.reached_half_credit and w.half_credit_time:
                hours = (w.half_credit_time - w.start_time).total_seconds() / 3600
                avg_hours_to_half.append(hours)
    else:
        # No windows meet the minimum cost threshold
        total_cost = 0
        total_opus_cost = 0
        total_sonnet_cost = 0
        windows_reached_half = 0
        avg_cost_per_window = 0
        avg_hours_to_half = []
    
    return {
        'total_windows': len(windows),
        'filtered_windows': len(stats_windows),
        'total_cost_all_windows': round(total_cost, 2),
        'avg_cost_per_window': round(avg_cost_per_window, 2),
        'windows_reached_half_credit': windows_reached_half,
        'percent_windows_reached_half': round(100 * windows_reached_half / len(stats_windows), 1) if stats_windows else 0,
        'total_opus_cost': round(total_opus_cost, 2),
        'total_sonnet_cost': round(total_sonnet_cost, 2),
        'opus_cost_percentage': round(100 * total_opus_cost / total_cost, 1) if total_cost > 0 else 0,
        'avg_hours_to_half_credit': round(sum(avg_hours_to_half) / len(avg_hours_to_half), 2) if avg_hours_to_half else None,
        'windows_data': windows_data,
        'min_window_cost': min_window_cost
    }


def create_windows_timeline_data(windows: List[CreditWindow]) -> Tuple[List[Dict], List[Dict]]:
    """
    Create timeline data for visualization.
    
    Returns:
        Tuple of (bars_data, annotations_data) for plotly visualization
    """
    if not windows:
        return [], []
    
    bars_data = []
    annotations_data = []
    
    for i, window in enumerate(windows):
        # Create bar for window duration
        bars_data.append({
            'x': [window.start_time, window.end_time],
            'y': [i, i],
            'mode': 'lines',
            'line': {'width': 20, 'color': 'rgba(100, 149, 237, 0.6)'},
            'hovertemplate': (
                f'Window {i+1}<br>'
                f'Start: %{{x[0]|%Y-%m-%d %H:%M}}<br>'
                f'End: %{{x[1]|%Y-%m-%d %H:%M}}<br>'
                f'Cost: ${window.total_cost:.2f}<br>'
                f'Messages: {len(window.messages)}<extra></extra>'
            ),
            'showlegend': False
        })
        
        # Mark half-credit point if reached
        if window.reached_half_credit and window.half_credit_time:
            annotations_data.append({
                'x': window.half_credit_time,
                'y': i,
                'text': '½',
                'showarrow': True,
                'arrowhead': 2,
                'arrowsize': 1,
                'arrowwidth': 2,
                'arrowcolor': 'orange',
                'ax': 0,
                'ay': -30,
                'font': {'size': 14, 'color': 'orange'}
            })
    
    return bars_data, annotations_data