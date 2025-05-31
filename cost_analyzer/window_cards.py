"""Window cards component for displaying detected credit windows."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List
import dash_bootstrap_components as dbc
from dash import html


def create_window_cards(windows_data: Dict[str, Any]) -> html.Div:
    """Create cards showing detected credit windows.
    
    Args:
        windows_data: Window analysis data
        
    Returns:
        html.Div containing window cards
    """
    windows = windows_data.get('windows', [])
    if not windows:
        return html.Div([
            dbc.Alert("No credit windows detected yet", color="info")
        ])
    
    # Import timezone function
    from .data_processor import get_local_timezone
    
    # Get local timezone
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    
    # Convert string timestamps to datetime objects with proper timezone
    for window in windows:
        if isinstance(window['start_time'], str):
            window['start_time'] = datetime.fromisoformat(window['start_time'].replace('Z', '+00:00'))
            if window['start_time'].tzinfo is None:
                window['start_time'] = window['start_time'].replace(tzinfo=utc_tz)
            window['start_time'] = window['start_time'].astimezone(local_tz)
        if isinstance(window['end_time'], str):
            window['end_time'] = datetime.fromisoformat(window['end_time'].replace('Z', '+00:00'))
            if window['end_time'].tzinfo is None:
                window['end_time'] = window['end_time'].replace(tzinfo=utc_tz)
            window['end_time'] = window['end_time'].astimezone(local_tz)
        for msg in window.get('messages', []):
            if isinstance(msg['timestamp'], str):
                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                if msg['timestamp'].tzinfo is None:
                    msg['timestamp'] = msg['timestamp'].replace(tzinfo=utc_tz)
                msg['timestamp'] = msg['timestamp'].astimezone(local_tz)
    
    # Get current time in local timezone
    now = datetime.now(tz=local_tz)
    
    # Sort windows by start time (most recent first)
    sorted_windows = sorted(windows, key=lambda w: w['start_time'], reverse=True)
    
    # Only show recent windows (last 48 hours)
    recent_windows = [w for w in sorted_windows if w['start_time'] >= now - timedelta(hours=48)]
    
    cards = []
    for window in recent_windows[:5]:  # Show max 5 windows
        # Determine if this is the current window
        is_current = window['start_time'] <= now < window['end_time']
        
        # Format window time range
        start_str = window['start_time'].strftime("%b %d %H:%M")
        end_str = window['end_time'].strftime("%H:%M")
        time_range = f"{start_str} - {end_str}"
        
        # Calculate window duration and message count
        messages = window.get('messages', [])
        actual_duration = (messages[-1]['timestamp'] - window['start_time']).total_seconds() / 3600 if messages else 0
        
        # Create status badge
        if is_current:
            status_badge = dbc.Badge("Current", color="success", className="ms-2")
        else:
            status_badge = None
        
        # Create model breakdown
        model_info = []
        opus_cost = window.get('opus_cost', 0)
        sonnet_cost = window.get('sonnet_cost', 0)
        opus_messages = window.get('opus_messages', 0)
        sonnet_messages = window.get('sonnet_messages', 0)
        
        if opus_messages > 0:
            model_info.append(
                html.Div([
                    html.Span("● ", style={'color': '#e91e63'}),
                    html.Span(f"Opus 4 • ${opus_cost:.2f}", style={'color': '#e91e63'}),
                    html.Div([
                        html.Small(f"Messages: {opus_messages}", className="text-muted d-block"),
                        html.Small(f"Tokens: {window.get('opus_tokens', 0):,}", className="text-muted d-block"),
                    ])
                ])
            )
        
        if sonnet_messages > 0:
            model_info.append(
                html.Div([
                    html.Span("● ", style={'color': '#2196f3'}),
                    html.Span(f"Sonnet 4 • ${sonnet_cost:.2f}", style={'color': '#2196f3'}),
                    html.Div([
                        html.Small(f"Messages: {sonnet_messages}", className="text-muted d-block"),
                        html.Small(f"Tokens: {window.get('sonnet_tokens', 0):,}", className="text-muted d-block"),
                    ])
                ])
            )
        
        # Create window card
        card_color = "success" if is_current else "secondary"
        card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    time_range,
                    status_badge
                ], className="mb-0"),
                html.Div([
                    html.H3(f"${window.get('total_cost', 0):.2f}", className="text-end mb-0"),
                    html.Small(f"{window.get('total_messages', 0)} messages", className="text-muted")
                ], className="text-end")
            ], className="d-flex justify-content-between align-items-center"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(model_info, width=12)
                ])
            ])
        ], color=card_color, outline=True, className="mb-3")
        
        cards.append(card)
    
    return html.Div([
        html.H4([
            html.I(className="bi bi-clock-history me-2"),
            "Detected 5-Hour Windows"
        ], className="mb-3"),
        html.Div(cards)
    ])