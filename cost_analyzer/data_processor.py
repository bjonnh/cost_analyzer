"""Data processing module for Claude Code Cost Analyzer."""

import os
import json
import glob
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd
import numpy as np

from .database import (
    init_database, insert_or_update_record, fetch_all_records,
    get_database_path, get_connection
)


def get_local_timezone() -> ZoneInfo:
    """Get the system's local timezone.
    
    Returns:
        ZoneInfo: Local timezone
    """
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
    return ZoneInfo('localtime')


def extract_project_name(file_path: str) -> str:
    """Extract project name from file path.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        str: Extracted project name
    """
    parts = file_path.split(os.sep)
    try:
        # Find the '.claude' part and get the project name after 'projects'
        claude_idx = -1
        for i, part in enumerate(parts):
            if part == '.claude':
                claude_idx = i
                break
        
        if claude_idx != -1 and claude_idx + 2 < len(parts):
            # Get the project name from the path
            project_name = parts[claude_idx + 2]
            
            # Extract just the last portion after any path separators
            if '/' in project_name:
                project_name = project_name.split('/')[-1]
            if '-' in project_name and project_name.startswith('-'):
                # Handle cases like '-Users-username-project'
                project_name = project_name.split('-')[-1]
        else:
            project_name = "unknown"
    except ValueError:
        # Fallback if '.claude' not in path
        project_name = os.path.basename(os.path.dirname(file_path))
    
    return project_name


def process_jsonl_entry(entry: Dict[str, Any], project_name: str, 
                       local_tz: ZoneInfo, utc_tz: ZoneInfo) -> Optional[Dict[str, Any]]:
    """Process a single JSONL entry.
    
    Args:
        entry: JSON entry from JSONL file
        project_name: Name of the project
        local_tz: Local timezone
        utc_tz: UTC timezone
        
    Returns:
        Optional[Dict[str, Any]]: Processed record data or None
    """
    cost = entry.get('costUSD', 0)
    if not cost:
        return None
    
    timestamp = entry.get('timestamp')
    if not timestamp:
        return None
    
    # Parse UTC timestamp and convert to local timezone
    utc_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    utc_dt = utc_dt.replace(tzinfo=utc_tz)
    
    # Convert to local timezone
    local_dt = utc_dt.astimezone(local_tz)
    
    # Get the date in local timezone
    date = local_dt.strftime('%Y-%m-%d')
    
    # Extract additional fields
    uuid = entry.get('id') or entry.get('uuid')
    duration_ms = entry.get('durationMs')
    
    # Extract model from message if it exists
    model = None
    message = entry.get('message')
    if message and isinstance(message, dict):
        model = message.get('model')
    
    # Extract token usage if available
    # Token data is nested inside the message object
    usage = {}
    if 'message' in entry and isinstance(entry['message'], dict):
        usage = entry['message'].get('usage', {})
    
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
    
    return {
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
        'service_tier': service_tier,
        'date': date
    }


def load_cost_data() -> Tuple[pd.DataFrame, List[str], float, pd.DataFrame]:
    """Load and process Claude Code cost data from JSONL files.
    
    Returns:
        Tuple containing:
        - DataFrame with cost data
        - List of project names
        - Total cost
        - DataFrame with token data
    """
    # Initialize database
    db_path = init_database()
    
    # Find all JSONL files in the Claude projects directory
    project_dir = os.path.expanduser("~/.claude/projects/")
    jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
    
    # Get local timezone
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    
    # Dictionary to store data
    data = defaultdict(lambda: defaultdict(float))
    token_data = defaultdict(lambda: defaultdict(lambda: {'input': 0, 'output': 0, 'cache_creation': 0, 'cache_read': 0, 'total': 0}))
    project_names = set()
    total_cost = 0
    
    with get_connection(db_path) as conn:
        # Process each JSONL file
        for file_path in jsonl_files:
            try:
                project_name = extract_project_name(file_path)
                project_names.add(project_name)
                
                # Process each line in the JSONL file
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            record = process_jsonl_entry(entry, project_name, local_tz, utc_tz)
                            
                            if record:
                                date = record['date']
                                
                                # Add cost to the appropriate date and project
                                data[date][project_name] += record['costUSD']
                                total_cost += record['costUSD']
                                
                                # Aggregate token data
                                if record['input_tokens']:
                                    token_data[date][project_name]['input'] += record['input_tokens']
                                if record['output_tokens']:
                                    token_data[date][project_name]['output'] += record['output_tokens']
                                if record['cache_creation_input_tokens']:
                                    token_data[date][project_name]['cache_creation'] += record['cache_creation_input_tokens']
                                if record['cache_read_input_tokens']:
                                    token_data[date][project_name]['cache_read'] += record['cache_read_input_tokens']
                                if record['total_tokens']:
                                    token_data[date][project_name]['total'] += record['total_tokens']
                                
                                # Store in database if we have a UUID
                                if record['uuid']:
                                    insert_or_update_record(conn, record)
                        except (json.JSONDecodeError, KeyError) as e:
                            continue
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    # Convert to DataFrames
    dates = sorted(data.keys())
    projects = sorted(list(project_names))

    # Create cost DataFrame
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
    
    # Create token DataFrame
    token_df = pd.DataFrame(index=dates)
    for date in dates:
        for project in projects:
            for token_type in ['input', 'output', 'cache_creation', 'cache_read', 'total']:
                col_name = f"{project}_{token_type}"
                token_df.loc[date, col_name] = token_data[date][project][token_type]
    
    token_df = token_df.fillna(0)
    token_df = token_df.infer_objects(copy=False)
    
    # Filter token_df to match cost df dates
    if len(df) > 0:
        token_df = token_df.loc[df.index]
    
    return df, projects, total_cost, token_df


def refresh_and_load_data() -> Tuple[pd.DataFrame, List[str], float, pd.DataFrame]:
    """Refresh data by scanning JSONL files, updating database, then loading from database.
    
    This ensures:
    - New JSONL data is picked up and stored in database
    - Historical database data is preserved (even if JSONL files are deleted)
    - Database acts as single source of truth for display
    
    Returns:
        Tuple containing:
        - DataFrame with cost data
        - List of project names
        - Total cost
        - DataFrame with token data
    """
    # First, scan JSONL files and update database
    load_cost_data()
    
    # Then load from database to get complete dataset
    return load_data_from_database()


def load_data_from_database() -> Tuple[pd.DataFrame, List[str], float, pd.DataFrame]:
    """Load usage data from the SQLite database.
    
    Returns:
        Tuple containing:
        - DataFrame with cost data
        - List of project names
        - Total cost
        - DataFrame with token data
    """
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        # If database doesn't exist, create it by loading from JSONL first
        return load_cost_data()
    
    # Get local timezone
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    
    # Query all data from database
    records = fetch_all_records(db_path)
    
    # Process records into DataFrames
    data = defaultdict(lambda: defaultdict(float))
    token_data = defaultdict(lambda: defaultdict(lambda: {'input': 0, 'output': 0, 'cache_creation': 0, 'cache_read': 0, 'total': 0}))
    project_names = set()
    total_cost = 0
    
    for record in records:
        uuid, timestamp, project, cost, model, input_tokens, output_tokens, cache_creation, cache_read, total_tokens, service_tier, duration = record
        
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
    
    # Filter token_df to match cost df dates
    if len(df) > 0:
        token_df = token_df.loc[df.index]
    
    return df, projects, total_cost, token_df


def calculate_statistics(df: pd.DataFrame, projects: List[str], token_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calculate various statistics from the cost data.
    
    Args:
        df: DataFrame with cost data
        projects: List of project names
        token_df: Optional DataFrame with token data
        
    Returns:
        Dict[str, Any]: Dictionary of statistics
    """
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