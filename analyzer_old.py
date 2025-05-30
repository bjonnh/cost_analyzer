#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas>=2.0.0",
#   "plotly>=6.0.0",
#   "dash>=2.0.0",
#   "dash-bootstrap-components>=1.0.0",
#   "scikit-learn>=1.0.0",
#   "numpy>=1.20.0",
# ]
# ///
"""
Claude Code Cost Dashboard

This script analyzes Claude Code usage costs from .jsonl files in the Claude projects directory.
It creates an interactive Dash dashboard with two charts:
- Daily costs by project (stacked area chart)
- Cumulative costs over time (line/bar chart)

The dashboard runs on a local server and updates automatically when refreshed.

Usage:
  ./analyzer.py
  (Dependencies will be automatically installed by uv)
"""

import os
import json
import glob
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import webbrowser
from threading import Timer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def init_database():
    """Initialize SQLite database for storing Claude Code usage data."""
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_records (
            uuid TEXT PRIMARY KEY,
            costUSD REAL NOT NULL,
            durationMs INTEGER,
            model TEXT,
            timestamp TEXT,
            project TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_creation_input_tokens INTEGER,
            cache_read_input_tokens INTEGER,
            total_tokens INTEGER,
            service_tier TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on timestamp for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_records(timestamp)
    """)
    
    # Create index on project for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_project ON usage_records(project)
    """)
    
    conn.commit()
    conn.close()
    
    return db_path


def insert_or_update_record(conn, record_data):
    """Insert or update a usage record in the database."""
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO usage_records 
        (uuid, costUSD, durationMs, model, timestamp, project,
         input_tokens, output_tokens, cache_creation_input_tokens,
         cache_read_input_tokens, total_tokens, service_tier, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        record_data['uuid'],
        record_data['costUSD'],
        record_data.get('durationMs'),
        record_data.get('model'),
        record_data.get('timestamp'),
        record_data.get('project'),
        record_data.get('input_tokens'),
        record_data.get('output_tokens'),
        record_data.get('cache_creation_input_tokens'),
        record_data.get('cache_read_input_tokens'),
        record_data.get('total_tokens'),
        record_data.get('service_tier')
    ))
    
    conn.commit()


def get_local_timezone():
    """Get the system's local timezone."""
    # Try to get timezone from environment or system
    try:
        # First try to get from /etc/localtime symlink (Linux/Mac)
        if os.path.exists('/etc/localtime'):
            # Read the symlink to get timezone name
            localtime_path = os.path.realpath('/etc/localtime')
            # Extract timezone from path (e.g., /usr/share/zoneinfo/America/New_York)
            if 'zoneinfo' in localtime_path:
                tz_name = localtime_path.split('zoneinfo/')[-1]
                return ZoneInfo(tz_name)
    except:
        pass
    
    # Fallback: use system local timezone
    # This works on most systems including Windows
    return ZoneInfo('localtime')


def load_cost_data():
    """Load and process Claude Code cost data from JSONL files."""
    # Initialize database
    db_path = init_database()
    conn = sqlite3.connect(db_path)
    
    # Find all JSONL files in the Claude projects directory
    project_dir = os.path.expanduser("~/.claude/projects/")
    jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
    
    # Get local timezone
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    
    # Dictionary to store data: {date: {project: cost}}
    data = defaultdict(lambda: defaultdict(float))
    project_names = set()
    total_cost = 0
    
    # Process each JSONL file
    for file_path in jsonl_files:
        try:
            # Extract project name from path
            parts = file_path.split(os.sep) # Use os.sep for platform-independent path splitting
            try:
                # Find the '.claude' part and get the project name after 'projects'
                claude_idx = -1
                for i, part in enumerate(parts):
                    if part == '.claude':
                        claude_idx = i
                        break
                
                if claude_idx != -1 and claude_idx + 2 < len(parts): # Ensure 'projects' and project name exist
                    # Get the project name from the path
                    project_name = parts[claude_idx + 2]
                    
                    # Extract just the last portion after any path separators
                    if '/' in project_name:
                        project_name = project_name.split('/')[-1]
                    if '-' in project_name and project_name.startswith('-'):
                        # Handle cases like '-Users-username-project'
                        project_name = project_name.split('-')[-1]
                else:
                    project_name = "unknown" # Fallback if structure not as expected
            except ValueError:
                # Fallback if '.claude' not in path
                project_name = os.path.basename(os.path.dirname(file_path))
            
            project_names.add(project_name)
            
            # Process each line in the JSONL file
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # Check if costUSD exists
                        cost = entry.get('costUSD', 0)
                        if cost:
                            # Extract date from timestamp
                            timestamp = entry.get('timestamp')
                            if timestamp:
                                # Parse UTC timestamp and convert to local timezone
                                # Timestamps in JSONL are in format: 2024-12-01T00:00:00.000Z
                                utc_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                utc_dt = utc_dt.replace(tzinfo=utc_tz)
                                
                                # Convert to local timezone
                                local_dt = utc_dt.astimezone(local_tz)
                                
                                # Get the date in local timezone
                                date = local_dt.strftime('%Y-%m-%d')
                                
                                # Add cost to the appropriate date and project
                                data[date][project_name] += cost
                                total_cost += cost
                                
                                # Extract additional fields for database
                                uuid = entry.get('id') or entry.get('uuid')  # Try both 'id' and 'uuid' fields
                                duration_ms = entry.get('durationMs')
                                
                                # Extract model from message if it exists
                                model = None
                                message = entry.get('message')
                                if message and isinstance(message, dict):
                                    model = message.get('model')
                                
                                # Extract token usage if available
                                usage = entry.get('usage', {})
                                input_tokens = usage.get('input_tokens')
                                output_tokens = usage.get('output_tokens')
                                cache_creation_tokens = usage.get('cache_creation_input_tokens')
                                cache_read_tokens = usage.get('cache_read_input_tokens')
                                service_tier = usage.get('service_tier')
                                
                                # Calculate total tokens
                                total_tokens = 0
                                if input_tokens:
                                    total_tokens += input_tokens
                                if output_tokens:
                                    total_tokens += output_tokens
                                if cache_creation_tokens:
                                    total_tokens += cache_creation_tokens
                                if cache_read_tokens:
                                    total_tokens += cache_read_tokens
                                
                                # Store in database if we have a UUID
                                if uuid:
                                    record_data = {
                                        'uuid': uuid,
                                        'costUSD': cost,
                                        'durationMs': duration_ms,
                                        'model': model,
                                        'timestamp': timestamp,
                                        'project': project_name,
                                        'input_tokens': input_tokens,
                                        'output_tokens': output_tokens,
                                        'cache_creation_input_tokens': cache_creation_tokens,
                                        'cache_read_input_tokens': cache_read_tokens,
                                        'total_tokens': total_tokens,
                                        'service_tier': service_tier
                                    }
                                    insert_or_update_record(conn, record_data)
                    except (json.JSONDecodeError, KeyError) as e:
                        continue  # Skip invalid lines
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    # Convert to DataFrame for easier plotting
    dates = sorted(data.keys())
    projects = sorted(list(project_names)) # Convert set to list for sorting

    # Create DataFrame
    df = pd.DataFrame(index=dates, columns=projects)
    for date in dates:
        for project in projects:
            df.loc[date, project] = data[date].get(project, 0)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    df = df.infer_objects(copy=False)  # Fix pandas FutureWarning
    
    # Calculate daily totals and filter out days with zero cost
    df['daily_total'] = df.sum(axis=1)
    df = df[df['daily_total'] > 0]  # Keep only days with cost > 0
    df = df.drop(columns=['daily_total'])  # Remove the temporary column
    
    # Close database connection
    conn.close()
    
    return df, projects, total_cost


def load_data_from_database():
    """Load usage data from the SQLite database."""
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    
    if not os.path.exists(db_path):
        # If database doesn't exist, create it by loading from JSONL first
        return load_cost_data()
    
    conn = sqlite3.connect(db_path)
    
    # Get local timezone
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    
    # Query all data from database
    query = """
        SELECT 
            timestamp, project, costUSD, model,
            input_tokens, output_tokens, 
            cache_creation_input_tokens, cache_read_input_tokens,
            total_tokens, service_tier, durationMs
        FROM usage_records
        WHERE costUSD > 0
        ORDER BY timestamp
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    records = cursor.fetchall()
    
    # Process records into DataFrames
    data = defaultdict(lambda: defaultdict(float))
    token_data = defaultdict(lambda: defaultdict(lambda: {'input': 0, 'output': 0, 'cache_creation': 0, 'cache_read': 0, 'total': 0}))
    project_names = set()
    total_cost = 0
    
    for record in records:
        timestamp, project, cost, model, input_tokens, output_tokens, cache_creation, cache_read, total_tokens, service_tier, duration = record
        
        if timestamp and project:
            # Convert UTC timestamp to local timezone
            utc_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            utc_dt = utc_dt.replace(tzinfo=utc_tz)
            local_dt = utc_dt.astimezone(local_tz)
            date = local_dt.strftime('%Y-%m-%d')
            
            # Aggregate cost data
            data[date][project] += cost
            total_cost += cost
            project_names.add(project)
            
            # Aggregate token data
            if input_tokens:
                token_data[date][project]['input'] += input_tokens
            if output_tokens:
                token_data[date][project]['output'] += output_tokens
            if cache_creation:
                token_data[date][project]['cache_creation'] += cache_creation
            if cache_read:
                token_data[date][project]['cache_read'] += cache_read
            if total_tokens:
                token_data[date][project]['total'] += total_tokens
    
    conn.close()
    
    # Convert to DataFrames
    dates = sorted(data.keys())
    projects = sorted(list(project_names))
    
    # Cost DataFrame
    df = pd.DataFrame(index=dates, columns=projects)
    for date in dates:
        for project in projects:
            df.loc[date, project] = data[date].get(project, 0)
    
    df = df.fillna(0)
    df = df.infer_objects(copy=False)
    
    # Filter out days with zero cost
    df['daily_total'] = df.sum(axis=1)
    df = df[df['daily_total'] > 0]
    df = df.drop(columns=['daily_total'])
    
    # Token DataFrame
    token_df = pd.DataFrame(index=dates)
    for date in dates:
        for project in projects:
            for token_type in ['input', 'output', 'cache_creation', 'cache_read', 'total']:
                col_name = f"{project}_{token_type}"
                token_df.loc[date, col_name] = token_data[date][project][token_type]
    
    token_df = token_df.fillna(0)
    token_df = token_df.infer_objects(copy=False)
    
    return df, projects, total_cost, token_df


def calculate_statistics(df, projects, token_df=None):
    """Calculate various statistics from the cost data."""
    stats = {}
    
    # Overall statistics
    total_cost = df.sum().sum()
    daily_costs = df.sum(axis=1)
    
    stats['total_cost'] = total_cost
    stats['avg_daily_cost'] = daily_costs.mean()
    stats['median_daily_cost'] = daily_costs.median()
    stats['max_daily_cost'] = daily_costs.max()
    stats['max_daily_date'] = daily_costs.idxmax()
    stats['active_days'] = len(df)
    
    # Calculate hourly statistics (assuming 8-hour workday)
    stats['avg_hourly_cost'] = stats['avg_daily_cost'] / 8
    stats['max_hourly_cost'] = stats['max_daily_cost'] / 8
    
    # Project statistics
    project_totals = df[projects].sum().sort_values(ascending=False)
    stats['top_project'] = project_totals.index[0] if len(project_totals) > 0 else "N/A"
    stats['top_project_cost'] = project_totals.iloc[0] if len(project_totals) > 0 else 0
    
    # Day of week analysis
    df_with_dow = df.copy()
    df_with_dow['dow'] = pd.to_datetime(df_with_dow.index).day_name()
    dow_costs = df_with_dow.groupby('dow').sum().sum(axis=1)
    stats['most_expensive_dow'] = dow_costs.idxmax() if len(dow_costs) > 0 else "N/A"
    
    # Token statistics if available
    if token_df is not None and len(token_df) > 0:
        # Calculate total tokens
        total_cols = [col for col in token_df.columns if col.endswith('_total')]
        if total_cols:
            daily_tokens = token_df[total_cols].sum(axis=1)
            stats['total_tokens'] = int(token_df[total_cols].sum().sum())
            stats['avg_daily_tokens'] = int(daily_tokens.mean())
            stats['max_daily_tokens'] = int(daily_tokens.max())
            
            # Input vs Output breakdown
            input_cols = [col for col in token_df.columns if col.endswith('_input')]
            output_cols = [col for col in token_df.columns if col.endswith('_output')]
            cache_creation_cols = [col for col in token_df.columns if col.endswith('_cache_creation')]
            cache_read_cols = [col for col in token_df.columns if col.endswith('_cache_read')]
            
            if input_cols:
                stats['total_input_tokens'] = int(token_df[input_cols].sum().sum())
            if output_cols:
                stats['total_output_tokens'] = int(token_df[output_cols].sum().sum())
            if cache_creation_cols:
                stats['total_cache_creation_tokens'] = int(token_df[cache_creation_cols].sum().sum())
            if cache_read_cols:
                stats['total_cache_read_tokens'] = int(token_df[cache_read_cols].sum().sum())
            
            # Calculate cost per 1K tokens
            if stats.get('total_tokens', 0) > 0:
                stats['cost_per_1k_tokens'] = (total_cost / stats['total_tokens']) * 1000
    
    return stats


def predict_future_costs(df, days_ahead=30, model_types=None):
    """Predict future costs using multiple regression models."""
    if len(df) < 6:  # Need at least 6 days of data
        return None
    
    # Default to all model types if not specified
    if model_types is None:
        model_types = ['survival', 'random']
    
    # Prepare data
    daily_totals = df.sum(axis=1)
    dates = pd.to_datetime(daily_totals.index)
    
    # Convert dates to numerical values (days since first date)
    X = (dates - dates[0]).days.values.reshape(-1, 1)
    y = daily_totals.values
    
    # Create future dates
    last_day = X[-1][0]
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    # Calculate physical limit based on max hourly rate and 8-hour constraint
    max_hourly_rate = np.max(y) / 8  # Maximum hourly rate observed
    daily_cap = max_hourly_rate * 8  # Physical limit per day
    
    results = {}
    
    # Survival analysis model
    if 'survival' in model_types:
        # Calculate the "hazard" of high costs
        # We'll model the probability of costs staying above different thresholds
        
        # Define cost thresholds (percentiles of historical costs)
        percentiles = [50, 75, 90]
        threshold_values = np.percentile(y, percentiles)
        
        # Kaplan-Meier style analysis
        # Calculate survival probability for each threshold
        survival_data = {}
        
        for i, threshold in enumerate(threshold_values):
            # Count consecutive days above threshold
            above_threshold = y > threshold
            
            # Find runs of days above threshold
            runs = []
            current_run = 0
            for val in above_threshold:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            
            # Calculate mean run length (expected duration above threshold)
            mean_duration = np.mean(runs) if runs else 1
            
            # Exponential decay model for survival probability
            # P(survive t days) = exp(-t/mean_duration)
            survival_data[f'p{percentiles[i]}'] = {
                'threshold': threshold,
                'mean_duration': mean_duration,
                'current_streak': current_run
            }
        
        # Current cost level and trend
        recent_costs = y[-7:] if len(y) >= 7 else y  # Last week
        current_level = y[-1]
        recent_trend = (recent_costs[-1] - recent_costs[0]) / len(recent_costs) if len(recent_costs) > 1 else 0
        
        # Make predictions based on survival model
        predictions_survival = []
        historical_fit_survival = []
        
        # For historical fit, use a smoothed version
        window = min(7, len(y) // 3)
        for i in range(len(y)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(y), i + window // 2 + 1)
            historical_fit_survival.append(np.mean(y[start_idx:end_idx]))
        
        # For future predictions, use exponential decay from current level
        # toward the median cost
        median_cost = np.median(y)
        
        for day_idx in range(days_ahead):
            # Exponential decay toward median
            decay_rate = 1 / survival_data['p75']['mean_duration']  # Use 75th percentile duration
            
            # Add some randomness based on historical volatility
            volatility = np.std(np.diff(y)) if len(y) > 1 else 0
            
            # Prediction with decay toward median
            days_ahead_current = day_idx + 1
            predicted = median_cost + (current_level - median_cost) * np.exp(-decay_rate * days_ahead_current)
            
            # Add trend component
            predicted += recent_trend * days_ahead_current * 0.5  # Damped trend
            
            # Ensure non-negative and apply daily cap
            predicted = max(0, min(predicted, daily_cap))
            
            predictions_survival.append(predicted)
        
        # Calculate R²
        ss_res = np.sum((y - historical_fit_survival) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Create description
        desc_parts = []
        for pct, data in survival_data.items():
            desc_parts.append(f"{pct}: ${data['threshold']:.0f} ({data['mean_duration']:.1f}d)")
        
        results['survival'] = {
            'name': 'Survival Analysis',
            'dates': future_dates,
            'predictions': np.array(predictions_survival),
            'historical_dates': dates,
            'historical_fit': np.array(historical_fit_survival),
            'r_squared': r_squared,
            'equation': f'Decay to median ${median_cost:.2f} | Thresholds: {", ".join(desc_parts)}'
        }
    
    # Random distribution model
    if 'random' in model_types:
        # Calculate statistics of the daily costs
        mean_cost = np.mean(y)
        std_cost = np.std(y)
        
        # Generate random predictions from normal distribution
        np.random.seed(42)  # For reproducibility
        predictions_random = []
        
        for _ in range(days_ahead):
            # Sample from normal distribution
            sample = np.random.normal(mean_cost, std_cost)
            
            # Apply constraints: non-negative and daily cap
            sample = max(0, min(sample, daily_cap))
            
            predictions_random.append(sample)
        
        # For historical fit, just use the actual mean for each day
        # This gives a flat line at the mean level
        historical_fit_random = np.full_like(y, mean_cost)
        
        # Apply daily cap to historical fit
        historical_fit_random = np.minimum(historical_fit_random, daily_cap)
        
        # Calculate R² (will be low since we're fitting with the mean)
        ss_res = np.sum((y - historical_fit_random) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results['random'] = {
            'name': 'Random Distribution',
            'dates': future_dates,
            'predictions': np.array(predictions_random),
            'historical_dates': dates,
            'historical_fit': historical_fit_random,
            'r_squared': r_squared,
            'equation': f'N(μ=${mean_cost:.2f}, σ=${std_cost:.2f}) | Cap=${daily_cap:.2f}/day'
        }
    
    return results


def create_figure(df, projects, total_cost):
    """Create the Plotly figure with two subplots."""
    # Create a single figure with two subplots sharing the x-axis
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Daily Claude Code Usage Cost by Project', 'Cumulative Claude Code Usage Cost')
    )
    
    # 1. Daily stacked area chart (top subplot)
    for project in projects:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[project],
                mode='lines',
                name=project,
                stackgroup='one',
                hoverinfo='x+y+name'
            ),
            row=1, col=1
        )
    
    # Add total cost line to daily chart
    df['Total'] = df.sum(axis=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Total'],
            mode='lines',
            name='Total Cost',
            line=dict(width=2, color='black', dash='dash'),
        ),
        row=1, col=1
    )
    
    # 2. Cumulative cost chart (bottom subplot)
    df['Cumulative'] = df['Total'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Cumulative'],
            mode='lines',
            name='Cumulative Cost',
            line=dict(width=3, color='red'),
            fill='tozeroy',
        ),
        row=2, col=1
    )
    
    # Add daily cost bars to the cumulative chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Total'],
            name='Daily Cost',
            marker_color='rgba(55, 83, 109, 0.7)',
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # Update layout with dark theme
    fig.update_layout(
        title=f'Claude Code Usage Cost Analysis (Total: ${total_cost:.2f})',
        hovermode='x unified',
        legend_title='Projects/Costs',
        height=900,
        margin=dict(l=50, r=50, t=100, b=100),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="Daily Cost (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Cost (USD)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Add physical limit line to daily chart
    max_hourly_rate = df.sum(axis=1).max() / 8
    daily_cap = max_hourly_rate * 8
    
    fig.add_hline(
        y=daily_cap, 
        line_dash="dot", 
        line_color="red",
        annotation_text=f"Physical Limit: ${daily_cap:.2f}/day",
        annotation_position="top right",
        row=1, col=1
    )
    
    return fig


def create_additional_charts(df, projects, stats, predictions):
    """Create additional analysis charts."""
    charts = {}
    
    # 1. Project cost breakdown horizontal bar chart
    project_totals = df[projects].sum().sort_values(ascending=True)
    
    # Calculate percentages
    total_sum = project_totals.sum()
    percentages = (project_totals / total_sum * 100).round(1)
    
    # Create horizontal bar chart
    bar_fig = go.Figure()
    
    bar_fig.add_trace(go.Bar(
        y=project_totals.index,
        x=project_totals.values,
        orientation='h',
        text=[f'${val:.2f} ({pct}%)' for val, pct in zip(project_totals.values, percentages)],
        textposition='auto',
        marker=dict(
            color=project_totals.values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cost (USD)")
        )
    ))
    
    bar_fig.update_layout(
        title="Project Cost Breakdown",
        xaxis_title="Total Cost (USD)",
        yaxis_title="Project",
        height=max(400, len(projects) * 40),  # Dynamic height based on number of projects
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        showlegend=False
    )
    
    charts['project_breakdown'] = bar_fig
    
    # 2. Treemap visualization for project costs
    if len(projects) > 1:
        treemap_data = []
        for project in project_totals.index:
            cost = project_totals[project]
            treemap_data.append({
                'name': project,
                'value': cost,
                'label': f'{project}<br>${cost:.2f}'
            })
        
        treemap_fig = px.treemap(
            treemap_data,
            path=['name'],
            values='value',
            title='Project Cost Hierarchy',
            custom_data=['label']
        )
        
        treemap_fig.update_traces(
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{customdata[0]}</b><br>%{percentParent}<extra></extra>'
        )
        
        treemap_fig.update_layout(
            height=500,
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font=dict(color='#fff')
        )
        
        charts['treemap'] = treemap_fig
    
    # 3. Day of week analysis
    df_dow = df.copy()
    df_dow['dow'] = pd.to_datetime(df_dow.index).day_name()
    df_dow['dow_num'] = pd.to_datetime(df_dow.index).dayofweek
    dow_totals = df_dow.groupby(['dow', 'dow_num']).sum().sum(axis=1).reset_index()
    dow_totals.columns = ['dow', 'dow_num', 'cost']
    dow_totals = dow_totals.sort_values('dow_num')
    
    dow_fig = px.bar(
        dow_totals,
        x='dow',
        y='cost',
        title="Total Costs by Day of Week",
        labels={'cost': 'Total Cost (USD)', 'dow': 'Day of Week'}
    )
    dow_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    charts['dow'] = dow_fig
    
    # 4. Prediction charts - Individual models
    if predictions and isinstance(predictions, dict):
        # Create individual prediction charts
        daily_totals = df.sum(axis=1)
        
        for model_key, model_data in predictions.items():
            pred_fig = go.Figure()
            
            # Historical data
            pred_fig.add_trace(go.Scatter(
                x=pd.to_datetime(daily_totals.index),
                y=daily_totals.values,
                mode='markers',
                name='Actual Costs',
                marker=dict(size=8, color='lightblue')
            ))
            
            # Historical fit
            pred_fig.add_trace(go.Scatter(
                x=model_data['historical_dates'],
                y=model_data['historical_fit'],
                mode='lines',
                name=f'{model_data["name"]} Fit',
                line=dict(color='green', dash='dash', width=2)
            ))
            
            # Future predictions
            pred_fig.add_trace(go.Scatter(
                x=model_data['dates'],
                y=model_data['predictions'],
                mode='lines+markers',
                name='Predicted Costs',
                line=dict(color='red', width=3),
                marker=dict(size=6)
            ))
            
            pred_fig.update_layout(
                title=f"{model_data['name']} Model (R² = {model_data['r_squared']:.3f})<br><sub>{model_data['equation']}</sub>",
                xaxis_title="Date",
                yaxis_title="Daily Cost (USD)",
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#1a1a1a',
                font=dict(color='#fff')
            )
            
            charts[f'prediction_{model_key}'] = pred_fig
        
        # Create comparison chart
        comp_fig = go.Figure()
        
        # Historical data
        comp_fig.add_trace(go.Scatter(
            x=pd.to_datetime(daily_totals.index),
            y=daily_totals.values,
            mode='markers',
            name='Actual Costs',
            marker=dict(size=8, color='white'),
            showlegend=True
        ))
        
        # Add all model fits and predictions
        colors = {
            'survival': 'green',
            'random': 'orange'
        }
        
        for model_key, model_data in predictions.items():
            color = colors.get(model_key, 'gray')
            
            # Historical fit
            comp_fig.add_trace(go.Scatter(
                x=model_data['historical_dates'],
                y=model_data['historical_fit'],
                mode='lines',
                name=f'{model_data["name"]} (R²={model_data["r_squared"]:.2f})',
                line=dict(color=color, dash='dash', width=2),
                legendgroup=model_key
            ))
            
            # Future predictions
            comp_fig.add_trace(go.Scatter(
                x=model_data['dates'],
                y=model_data['predictions'],
                mode='lines',
                name=f'{model_data["name"]} Prediction',
                line=dict(color=color, width=3),
                legendgroup=model_key,
                showlegend=False
            ))
        
        comp_fig.update_layout(
            title="Model Comparison - All Prediction Models",
            xaxis_title="Date",
            yaxis_title="Daily Cost (USD)",
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font=dict(color='#fff'),
            height=600
        )
        
        charts['prediction_comparison'] = comp_fig
    
    # 5. Hourly cost analysis
    hourly_fig = go.Figure()
    
    # Calculate hourly costs for each day (assuming 8-hour workday)
    daily_totals = df.sum(axis=1)
    hourly_costs = daily_totals / 8
    
    # Create box plot for hourly costs
    hourly_fig.add_trace(go.Box(
        y=hourly_costs.values,
        name='Hourly Cost Distribution',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color='lightblue'),
        line=dict(color='blue')
    ))
    
    # Add average and max lines
    avg_hourly = hourly_costs.mean()
    max_hourly = hourly_costs.max()
    
    hourly_fig.add_hline(
        y=avg_hourly, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"Average: ${avg_hourly:.2f}/hr"
    )
    
    hourly_fig.add_hline(
        y=max_hourly, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Physical Limit: ${max_hourly:.2f}/hr"
    )
    
    hourly_fig.update_layout(
        title="Hourly Cost Analysis (8-hour workday)",
        yaxis_title="Cost per Hour (USD)",
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=400
    )
    
    charts['hourly_analysis'] = hourly_fig
    
    # 6. Moving average chart
    ma_fig = go.Figure()
    daily_totals = df.sum(axis=1)
    
    ma_fig.add_trace(go.Scatter(
        x=daily_totals.index,
        y=daily_totals.values,
        mode='lines',
        name='Daily Cost',
        line=dict(color='lightblue')
    ))
    
    # Add 7-day moving average
    ma_7 = daily_totals.rolling(window=7, min_periods=1).mean()
    ma_fig.add_trace(go.Scatter(
        x=ma_7.index,
        y=ma_7.values,
        mode='lines',
        name='7-Day MA',
        line=dict(color='blue', width=2)
    ))
    
    # Add 30-day moving average if enough data
    if len(daily_totals) >= 30:
        ma_30 = daily_totals.rolling(window=30, min_periods=1).mean()
        ma_fig.add_trace(go.Scatter(
            x=ma_30.index,
            y=ma_30.values,
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=2)
        ))
    
    ma_fig.update_layout(
        title="Cost Trends with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Cost (USD)",
        hovermode='x unified'
    )
    ma_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    charts['moving_avg'] = ma_fig
    
    return charts


def create_token_charts(token_df, projects, stats):
    """Create token usage analysis charts."""
    charts = {}
    
    # 1. Token usage over time (stacked area chart)
    token_fig = go.Figure()
    
    # Get daily totals for each token type
    dates = token_df.index
    
    # Calculate totals for each type
    input_total = pd.Series(0, index=dates)
    output_total = pd.Series(0, index=dates) 
    cache_creation_total = pd.Series(0, index=dates)
    cache_read_total = pd.Series(0, index=dates)
    
    for project in projects:
        if f"{project}_input" in token_df.columns:
            input_total += token_df[f"{project}_input"]
        if f"{project}_output" in token_df.columns:
            output_total += token_df[f"{project}_output"]
        if f"{project}_cache_creation" in token_df.columns:
            cache_creation_total += token_df[f"{project}_cache_creation"]
        if f"{project}_cache_read" in token_df.columns:
            cache_read_total += token_df[f"{project}_cache_read"]
    
    # Add traces for each token type
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=input_total,
        mode='lines',
        name='Input Tokens',
        stackgroup='one',
        fillcolor='rgba(65, 105, 225, 0.6)'
    ))
    
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=output_total,
        mode='lines',
        name='Output Tokens',
        stackgroup='one',
        fillcolor='rgba(50, 205, 50, 0.6)'
    ))
    
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=cache_creation_total,
        mode='lines',
        name='Cache Creation Tokens',
        stackgroup='one',
        fillcolor='rgba(255, 165, 0, 0.6)'
    ))
    
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=cache_read_total,
        mode='lines',
        name='Cache Read Tokens',
        stackgroup='one',
        fillcolor='rgba(138, 43, 226, 0.6)'
    ))
    
    token_fig.update_layout(
        title="Daily Token Usage by Type",
        xaxis_title="Date",
        yaxis_title="Number of Tokens",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=600
    )
    
    charts['token_usage_time'] = token_fig
    
    # 2. Token type breakdown (pie chart)
    if stats and 'total_tokens' in stats and stats['total_tokens'] > 0:
        labels = []
        values = []
        colors = []
        
        if 'total_input_tokens' in stats and stats['total_input_tokens'] > 0:
            labels.append('Input')
            values.append(stats['total_input_tokens'])
            colors.append('#4169E1')
        
        if 'total_output_tokens' in stats and stats['total_output_tokens'] > 0:
            labels.append('Output')
            values.append(stats['total_output_tokens'])
            colors.append('#32CD32')
            
        if 'total_cache_creation_tokens' in stats and stats['total_cache_creation_tokens'] > 0:
            labels.append('Cache Creation')
            values.append(stats['total_cache_creation_tokens'])
            colors.append('#FFA500')
            
        if 'total_cache_read_tokens' in stats and stats['total_cache_read_tokens'] > 0:
            labels.append('Cache Read')
            values.append(stats['total_cache_read_tokens'])
            colors.append('#8A2BE2')
        
        if labels:
            pie_fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='auto',
            )])
            
            pie_fig.update_layout(
                title="Token Usage Breakdown",
                template='plotly_dark',
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#1a1a1a',
                font=dict(color='#fff'),
                height=500,
                annotations=[dict(
                    text=f'{stats["total_tokens"]:,}<br>Total Tokens',
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )]
            )
            
            charts['token_breakdown'] = pie_fig
    
    # 3. Token efficiency chart (tokens per dollar)
    if stats and stats.get('total_cost', 0) > 0:
        efficiency_fig = go.Figure()
        
        # Calculate daily token totals and costs
        daily_tokens = input_total + output_total + cache_creation_total + cache_read_total
        
        # Get daily costs from the main dataframe (we'll need to pass this)
        # For now, use average cost per token
        cost_per_token = stats['total_cost'] / stats.get('total_tokens', 1) if stats.get('total_tokens', 0) > 0 else 0
        tokens_per_dollar = 1 / cost_per_token if cost_per_token > 0 else 0
        
        efficiency_fig.add_trace(go.Scatter(
            x=dates,
            y=daily_tokens / 1000,  # Show in K tokens
            mode='lines+markers',
            name='Daily Tokens (K)',
            line=dict(color='cyan', width=2)
        ))
        
        efficiency_fig.update_layout(
            title=f"Token Usage Trend (Avg: {tokens_per_dollar:.0f} tokens/$)",
            xaxis_title="Date",
            yaxis_title="Tokens (thousands)",
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font=dict(color='#fff'),
            height=400
        )
        
        charts['token_efficiency'] = efficiency_fig
    
    # 4. Project token usage comparison
    project_tokens = {}
    for project in projects:
        total = 0
        for token_type in ['input', 'output', 'cache_creation', 'cache_read']:
            col_name = f"{project}_{token_type}"
            if col_name in token_df.columns:
                total += token_df[col_name].sum()
        if total > 0:
            project_tokens[project] = total
    
    if project_tokens:
        sorted_projects = sorted(project_tokens.items(), key=lambda x: x[1], reverse=True)
        
        bar_fig = go.Figure(data=[
            go.Bar(
                x=[p[0] for p in sorted_projects],
                y=[p[1] / 1e6 for p in sorted_projects],  # Show in millions
                text=[f'{p[1]/1e6:.1f}M' for p in sorted_projects],
                textposition='auto',
                marker=dict(
                    color=[p[1] for p in sorted_projects],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Tokens (M)")
                )
            )
        ])
        
        bar_fig.update_layout(
            title="Token Usage by Project",
            xaxis_title="Project",
            yaxis_title="Total Tokens (millions)",
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font=dict(color='#fff'),
            height=400
        )
        
        charts['token_by_project'] = bar_fig
    
    return charts


def open_browser(port):
    """Open the browser after a short delay."""
    webbrowser.open_new(f'http://localhost:{port}/')


# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0a0a0a;
            }
            .card {
                background-color: #1a1a1a;
                border: 1px solid #333;
            }
            .stats-card {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s;
            }
            .stats-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            }
            .prediction-alert {
                background-color: #2d1b69;
                border: 1px solid #6c4ba8;
            }
            .DateRangePickerInput {
                background-color: #1a1a1a !important;
                border: 1px solid #444 !important;
            }
            .DateRangePickerInput__withBorder {
                border-color: #444 !important;
            }
            .DateInput_input {
                background-color: #1a1a1a !important;
                color: #fff !important;
            }
            .DateRangePickerInput_arrow {
                background-color: #1a1a1a !important;
            }
            .btn-group-sm > .btn {
                padding: 0.25rem 0.5rem;
                font-size: 0.875rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Claude Code Cost Analytics", 
                   className="text-center mb-4 mt-3",
                   style={'color': '#fff', 'fontWeight': 'bold'}),
            html.Hr(style={'borderColor': '#444'}),
        ])
    ]),
    
    # Control panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🔄 Refresh Data", 
                                      id='refresh-button', 
                                      color="primary", 
                                      size="lg",
                                      n_clicks=0),
                        ], width=2),
                        dbc.Col([
                            dcc.DatePickerRange(
                                id='date-range-picker',
                                display_format='YYYY-MM-DD',
                                style={'width': '100%'},
                                className="dash-bootstrap"
                            )
                        ], width=5),
                        dbc.Col([
                            html.Div(id='last-update', className="text-center mt-2")
                        ], width=3),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("1W", id="range-1w", color="secondary", size="sm"),
                                dbc.Button("1M", id="range-1m", color="secondary", size="sm"),
                                dbc.Button("3M", id="range-3m", color="secondary", size="sm"),
                                dbc.Button("All", id="range-all", color="secondary", size="sm"),
                            ], size="sm")
                        ], width=2),
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Date range info
    dbc.Row([
        dbc.Col([
            dbc.Alert(id="date-range-info", color="info", className="text-center mb-3", is_open=False)
        ])
    ]),
    
    # Stats cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Cost", className="card-title"),
                    html.H2(id="total-cost-stat", className="text-primary"),
                ])
            ], className="stats-card text-white")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Daily Average", className="card-title"),
                    html.H2(id="daily-avg-stat", className="text-info"),
                ])
            ], className="stats-card text-white")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Top Project", className="card-title"),
                    html.H3(id="top-project-stat", className="text-warning"),
                ])
            ], className="stats-card text-white")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Active Days", className="card-title"),
                    html.H2(id="active-days-stat", className="text-success"),
                ])
            ], className="stats-card text-white")
        ], width=3),
    ], className="mb-4"),
    
    # Prediction alert
    dbc.Row([
        dbc.Col([
            dbc.Alert(
                id="prediction-alert",
                is_open=False,
                className="prediction-alert",
                color="info"
            )
        ])
    ], className="mb-4"),
    
    # Main content tabs
    dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="overview"),
        dbc.Tab(label="📈 Trends", tab_id="trends"),
        dbc.Tab(label="🎯 Analysis", tab_id="analysis"),
        dbc.Tab(label="🔢 Tokens", tab_id="tokens"),
        dbc.Tab(label="🔮 Predictions", tab_id="predictions"),
    ], id="tabs", active_tab="overview", className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Hidden divs to store data
    dcc.Store(id='data-store'),
    dcc.Store(id='stats-store'),
    dcc.Store(id='predictions-store'),
    dcc.Store(id='charts-store'),
    dcc.Store(id='full-data-store'),  # Store unfiltered data
    dcc.Store(id='token-data-store'),  # Store token data
    
], fluid=True, style={'backgroundColor': '#0a0a0a'})


# Callback for date range quick buttons
@app.callback(
    [Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date')],
    [Input('range-1w', 'n_clicks'),
     Input('range-1m', 'n_clicks'),
     Input('range-3m', 'n_clicks'),
     Input('range-all', 'n_clicks')],
    [State('full-data-store', 'data')]
)
def update_date_range(n1w, n1m, n3m, nall, full_data):
    """Update date range based on quick selection buttons."""
    if not full_data:
        return None, None
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Get the date range from full data
    df = pd.DataFrame.from_dict(full_data['df'])
    if len(df) == 0:
        return None, None
    
    end_date = pd.to_datetime(df.index).max()
    
    if button_id == 'range-1w':
        start_date = end_date - timedelta(days=7)
    elif button_id == 'range-1m':
        start_date = end_date - timedelta(days=30)
    elif button_id == 'range-3m':
        start_date = end_date - timedelta(days=90)
    else:  # range-all
        start_date = pd.to_datetime(df.index).min()
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# Callback to update all data stores when refresh is clicked or date range changes
@app.callback(
    [Output('data-store', 'data'),
     Output('stats-store', 'data'),
     Output('predictions-store', 'data'),
     Output('charts-store', 'data'),
     Output('last-update', 'children'),
     Output('full-data-store', 'data'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('token-data-store', 'data')],
    [Input('refresh-button', 'n_clicks'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_data_stores(n_clicks, start_date, end_date):
    """Load and process all data when refresh is clicked or date range changes."""
    # Load the full data from database
    result = load_data_from_database()
    
    if len(result) == 3:
        # Old format without token data, load from JSONL
        df_full, projects, _ = result
        token_df_full = None
    else:
        # New format with token data
        df_full, projects, _, token_df_full = result
    
    if len(df_full) == 0:
        return {}, {}, None, {}, "No data available", {}, None, None, {}
    
    # Store full data
    full_data = {
        'df': df_full.to_dict(),
        'projects': projects,
        'total_cost': df_full.sum().sum()
    }
    
    # Get date range bounds
    min_date = pd.to_datetime(df_full.index).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(df_full.index).max().strftime('%Y-%m-%d')
    
    # Apply date filtering if dates are provided
    df = df_full.copy()
    token_df = token_df_full.copy() if token_df_full is not None else None
    
    if start_date and end_date:
        mask = (pd.to_datetime(df.index) >= pd.to_datetime(start_date)) & \
               (pd.to_datetime(df.index) <= pd.to_datetime(end_date))
        df = df[mask]
        if token_df is not None:
            token_df = token_df[mask]
    
    # Recalculate total cost for filtered data
    total_cost = df.sum().sum()
    
    # Calculate statistics on filtered data (now includes token stats)
    stats = calculate_statistics(df, projects, token_df) if len(df) > 0 else {}
    
    # Generate predictions on filtered data
    predictions = predict_future_costs(df) if len(df) > 0 else None
    
    # Create additional charts with filtered data
    charts = create_additional_charts(df, projects, stats, predictions) if len(df) > 0 else {}
    
    # Create token charts if token data is available
    if token_df is not None and len(token_df) > 0:
        token_charts = create_token_charts(token_df, projects, stats)
        charts.update(token_charts)
    
    # Update timestamp
    date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else " (All time)"
    last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{date_range_text}"
    
    # Store filtered data
    data = {
        'df': df.to_dict(),
        'projects': projects,
        'total_cost': total_cost
    }
    
    # Store token data
    token_data = token_df.to_dict() if token_df is not None else {}
    
    # Convert charts to JSON
    charts_json = {k: v.to_json() for k, v in charts.items()}
    
    return data, stats, predictions, charts_json, last_update, full_data, min_date, max_date, token_data


# Callback to update date range info
@app.callback(
    [Output('date-range-info', 'children'),
     Output('date-range-info', 'is_open')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')],
    [State('full-data-store', 'data')]
)
def update_date_range_info(start_date, end_date, full_data):
    """Update the date range information alert."""
    if not start_date or not end_date:
        return "", False
    
    if full_data:
        df_full = pd.DataFrame.from_dict(full_data['df'])
        min_date = pd.to_datetime(df_full.index).min()
        max_date = pd.to_datetime(df_full.index).max()
        
        # Check if we're showing all data
        if (pd.to_datetime(start_date).date() == min_date.date() and 
            pd.to_datetime(end_date).date() == max_date.date()):
            return "Showing all available data", True
    
    # Calculate days in range
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    return f"Showing data from {start_date} to {end_date} ({days} days)", True


# Callback to update stats cards
@app.callback(
    [Output('total-cost-stat', 'children'),
     Output('daily-avg-stat', 'children'),
     Output('top-project-stat', 'children'),
     Output('active-days-stat', 'children')],
    [Input('stats-store', 'data')]
)
def update_stats_cards(stats):
    """Update the statistics cards."""
    if not stats:
        return "$0.00", "$0.00", "N/A", "0"
    
    total = f"${stats.get('total_cost', 0):.2f}"
    daily_avg = f"${stats.get('avg_daily_cost', 0):.2f}"
    top_project = stats.get('top_project', 'N/A')
    if len(top_project) > 15:
        top_project = top_project[:15] + "..."
    active_days = str(stats.get('active_days', 0))
    
    return total, daily_avg, top_project, active_days


# Callback to update prediction alert
@app.callback(
    [Output('prediction-alert', 'children'),
     Output('prediction-alert', 'is_open')],
    [Input('predictions-store', 'data'),
     Input('stats-store', 'data')]
)
def update_prediction_alert(predictions, stats):
    """Update the prediction alert."""
    if not predictions or not stats:
        return "", False
    
    # Find the best model based on R²
    best_model_key = None
    best_r2 = -1
    best_model_data = None
    
    for model_key, model_data in predictions.items():
        if model_data['r_squared'] > best_r2:
            best_r2 = model_data['r_squared']
            best_model_key = model_key
            best_model_data = model_data
    
    if not best_model_data:
        return "", False
    
    # Calculate 30-day prediction with best model
    next_30_days_cost = sum(best_model_data['predictions'])
    
    alert_text = [
        html.H5("📊 30-Day Forecast", className="alert-heading"),
        html.P([
            f"Based on the {best_model_data['name']} model (best fit with R²={best_r2:.2f}), ",
            f"your estimated cost for the next 30 days is ",
            html.Strong(f"${next_30_days_cost:.2f}"),
            f" (${next_30_days_cost/30:.2f}/day)."
        ]),
        html.P([
            "Other models predict: ",
            html.Ul([
                html.Li(f"{data['name']}: ${sum(data['predictions']):.2f} (R²={data['r_squared']:.2f})")
                for key, data in predictions.items()
                if key != best_model_key
            ])
        ], className="small")
    ]
    
    return alert_text, True


# Callback to render tab content
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('data-store', 'data'),
     Input('stats-store', 'data'),
     Input('charts-store', 'data'),
     Input('token-data-store', 'data')]
)
def render_tab_content(active_tab, data_store, stats, charts_json, token_data_store):
    """Render content based on active tab."""
    if not data_store:
        return dbc.Alert("No data available. Click 'Refresh Data' to load.", color="warning")
    
    # Reconstruct dataframe
    df = pd.DataFrame.from_dict(data_store['df'])
    projects = data_store['projects']
    total_cost = data_store['total_cost']
    
    if active_tab == "overview":
        # Create the main figure
        fig = create_figure(df, projects, total_cost)
        return dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '900px'})
            ])
        ])
    
    elif active_tab == "trends":
        if 'moving_avg' in charts_json:
            ma_fig = go.Figure(json.loads(charts_json['moving_avg']))
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=ma_fig, style={'height': '500px'})
                        ])
                    ])
                ], width=12),
            ])
        return dbc.Alert("Not enough data for trend analysis", color="info")
    
    elif active_tab == "analysis":
        charts_content = []
        
        if 'project_breakdown' in charts_json:
            breakdown_fig = go.Figure(json.loads(charts_json['project_breakdown']))
            charts_content.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=breakdown_fig, style={'height': 'auto'})
                        ])
                    ])
                ], width=12)
            )
        
        # Add treemap if available
        if 'treemap' in charts_json:
            treemap_fig = go.Figure(json.loads(charts_json['treemap']))
            charts_content.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=treemap_fig, style={'height': '500px'})
                        ])
                    ])
                ], width=12, className="mb-4")
            )
        
        if 'dow' in charts_json:
            dow_fig = go.Figure(json.loads(charts_json['dow']))
            charts_content.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=dow_fig, style={'height': '400px'})
                        ])
                    ])
                ], width=6)
            )
        
        # Add hourly analysis if available
        if 'hourly_analysis' in charts_json:
            hourly_fig = go.Figure(json.loads(charts_json['hourly_analysis']))
            charts_content.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=hourly_fig, style={'height': '400px'})
                        ])
                    ])
                ], width=6)
            )
        
        # Add detailed statistics
        if stats:
            stats_content = dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Detailed Statistics")),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Maximum Daily Cost:", width=8),
                                    dbc.Col(f"${stats.get('max_daily_cost', 0):.2f}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Date of Highest Cost:", width=8),
                                    dbc.Col(stats.get('max_daily_date', 'N/A'), width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Median Daily Cost:", width=8),
                                    dbc.Col(f"${stats.get('median_daily_cost', 0):.2f}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Most Expensive Day of Week:", width=8),
                                    dbc.Col(stats.get('most_expensive_dow', 'N/A'), width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Average Hourly Cost (8h/day):", width=8),
                                    dbc.Col(f"${stats.get('avg_hourly_cost', 0):.2f}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Maximum Hourly Cost (8h/day):", width=8),
                                    dbc.Col(f"${stats.get('max_hourly_cost', 0):.2f}", width=4, className="text-end")
                                ])
                            ]),
                        ])
                    ])
                ])
            ], width=12, className="mt-4")
            
            return dbc.Container([
                dbc.Row(charts_content),
                dbc.Row([stats_content])
            ])
        
        return dbc.Row(charts_content)
    
    elif active_tab == "predictions":
        prediction_content = []
        
        # Check if we have predictions
        has_predictions = any(key.startswith('prediction_') for key in charts_json)
        
        if not has_predictions:
            return dbc.Alert("Not enough data for predictions (need at least 6 days)", color="info")
        
        # Model selector
        model_options = []
        if 'prediction_survival' in charts_json:
            model_options.append({'label': 'Survival Analysis', 'value': 'survival'})
        if 'prediction_random' in charts_json:
            model_options.append({'label': 'Random Distribution', 'value': 'random'})
        
        # Add controls
        prediction_content.append(
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Prediction Model Selection", className="card-title"),
                            dcc.RadioItems(
                                id='model-selector',
                                options=[
                                    {'label': 'Show Individual Model', 'value': 'individual'},
                                    {'label': 'Show All Models Comparison', 'value': 'comparison'}
                                ],
                                value='comparison',
                                inline=True,
                                className="mb-3"
                            ),
                            dcc.Dropdown(
                                id='individual-model-dropdown',
                                options=model_options,
                                value='survival' if 'prediction_survival' in charts_json else model_options[0]['value'] if model_options else None,
                                style={'display': 'none'}
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4")
        )
        
        # Add model comparison if available
        if 'prediction_comparison' in charts_json:
            comp_fig = go.Figure(json.loads(charts_json['prediction_comparison']))
            prediction_content.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(
                                    id='prediction-graph',
                                    figure=comp_fig,
                                    style={'height': '600px'}
                                )
                            ])
                        ])
                    ], width=12)
                ])
            )
        
        # Add model statistics table
        if stats:
            prediction_content.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Model Fit Statistics")),
                            dbc.CardBody([
                                html.Div(id='model-stats-table')
                            ])
                        ])
                    ], width=12)
                ], className="mt-4")
            )
        
        return dbc.Container(prediction_content)
    
    elif active_tab == "tokens":
        if not token_data_store:
            return dbc.Alert("No token usage data available.", color="info")
        
        token_charts_content = []
        
        # Display token usage charts
        if 'token_usage_time' in charts_json:
            token_time_fig = go.Figure(json.loads(charts_json['token_usage_time']))
            token_charts_content.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(figure=token_time_fig, style={'height': '600px'})
                            ])
                        ])
                    ], width=12)
                ], className="mb-4")
            )
        
        # Token breakdown and efficiency in same row
        row_content = []
        if 'token_breakdown' in charts_json:
            token_breakdown_fig = go.Figure(json.loads(charts_json['token_breakdown']))
            row_content.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=token_breakdown_fig, style={'height': '500px'})
                        ])
                    ])
                ], width=6)
            )
        
        if 'token_efficiency' in charts_json:
            token_efficiency_fig = go.Figure(json.loads(charts_json['token_efficiency']))
            row_content.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=token_efficiency_fig, style={'height': '400px'})
                        ])
                    ])
                ], width=6)
            )
        
        if row_content:
            token_charts_content.append(dbc.Row(row_content, className="mb-4"))
        
        # Token usage by project
        if 'token_by_project' in charts_json:
            token_project_fig = go.Figure(json.loads(charts_json['token_by_project']))
            token_charts_content.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(figure=token_project_fig, style={'height': '400px'})
                            ])
                        ])
                    ], width=12)
                ])
            )
        
        # Add token statistics
        if stats and 'total_tokens' in stats:
            token_stats_content = dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Token Usage Statistics")),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Total Tokens Used:", width=8),
                                    dbc.Col(f"{stats.get('total_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Average Daily Tokens:", width=8),
                                    dbc.Col(f"{stats.get('avg_daily_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Maximum Daily Tokens:", width=8),
                                    dbc.Col(f"{stats.get('max_daily_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Cost per 1K Tokens:", width=8),
                                    dbc.Col(f"${stats.get('cost_per_1k_tokens', 0):.4f}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Input Tokens:", width=8),
                                    dbc.Col(f"{stats.get('total_input_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Output Tokens:", width=8),
                                    dbc.Col(f"{stats.get('total_output_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Cache Creation Tokens:", width=8),
                                    dbc.Col(f"{stats.get('total_cache_creation_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col("Cache Read Tokens:", width=8),
                                    dbc.Col(f"{stats.get('total_cache_read_tokens', 0):,}", width=4, className="text-end")
                                ])
                            ]),
                        ])
                    ])
                ])
            ], width=12, className="mt-4")
            
            token_charts_content.append(dbc.Row([token_stats_content]))
        
        return dbc.Container(token_charts_content)
    
    return html.Div()


# Callback for model selector visibility
@app.callback(
    Output('individual-model-dropdown', 'style'),
    [Input('model-selector', 'value')]
)
def toggle_model_dropdown(selector_value):
    """Show/hide the individual model dropdown based on selector."""
    if selector_value == 'individual':
        return {'display': 'block'}
    return {'display': 'none'}


# Callback for prediction graph update
@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('model-selector', 'value'),
     Input('individual-model-dropdown', 'value')],
    [State('charts-store', 'data')]
)
def update_prediction_graph(selector_value, individual_model, charts_json):
    """Update the prediction graph based on model selection."""
    if not charts_json:
        return go.Figure()
    
    if selector_value == 'comparison' and 'prediction_comparison' in charts_json:
        return go.Figure(json.loads(charts_json['prediction_comparison']))
    elif selector_value == 'individual' and f'prediction_{individual_model}' in charts_json:
        return go.Figure(json.loads(charts_json[f'prediction_{individual_model}']))
    
    return go.Figure()


# Callback for model stats table
@app.callback(
    Output('model-stats-table', 'children'),
    [Input('predictions-store', 'data')]
)
def update_model_stats(predictions):
    """Update the model statistics table."""
    if not predictions:
        return html.P("No model statistics available")
    
    # Create table rows
    rows = []
    for model_key, model_data in predictions.items():
        # Calculate 30-day cost
        cost_30d = sum(model_data['predictions'])
        
        rows.append(
            html.Tr([
                html.Td(model_data['name']),
                html.Td(f"{model_data['r_squared']:.4f}"),
                html.Td(f"${cost_30d:.2f}"),
                html.Td(f"${cost_30d/30:.2f}"),
                html.Td(model_data['equation'], style={'fontSize': '0.9em'})
            ])
        )
    
    # Sort by R² descending
    rows.sort(key=lambda x: float(x.children[1].children), reverse=True)
    
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Model"),
                html.Th("R² Score"),
                html.Th("30-Day Forecast"),
                html.Th("Daily Average"),
                html.Th("Equation")
            ])
        ]),
        html.Tbody(rows)
    ], striped=True, hover=True, responsive=True, className="text-white")


if __name__ == '__main__':
    port = 8050
    print(f"Starting Claude Code Cost Dashboard on http://localhost:{port}/")
    print("Press Ctrl+C to stop the server")
    
    # Run the app
    app.run(debug=False, port=port)
