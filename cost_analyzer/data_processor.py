"""Data processing module for Claude Code Cost Analyzer."""

import os
import json
import glob
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np

from .database import (
    init_database, insert_or_update_record, insert_records_batch, fetch_all_records,
    get_database_path, get_connection
)
from .window_detector import detect_windows, analyze_windows

# Set up performance logging
performance_logger = logging.getLogger('performance')
performance_logger.setLevel(logging.INFO)
if not performance_logger.handlers:
    handler = logging.FileHandler('performance.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    performance_logger.addHandler(handler)

def log_timing(operation_name: str, start_time: float, **kwargs) -> None:
    """Log timing information for performance analysis."""
    duration = time.time() - start_time
    extra_info = ' '.join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ''
    performance_logger.info(f"{operation_name}: {duration:.4f}s {extra_info}")


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


def process_jsonl_file_batch(file_path: str, project_name: str, 
                           local_tz: ZoneInfo, utc_tz: ZoneInfo) -> Tuple[List[Dict[str, Any]], int, int]:
    """Process an entire JSONL file in batch mode for better performance.
    
    Args:
        file_path: Path to the JSONL file
        project_name: Name of the project
        local_tz: Local timezone
        utc_tz: UTC timezone
        
    Returns:
        Tuple[List[Dict[str, Any]], int, int]: List of processed records, total lines, valid records
    """
    records = []
    total_lines = 0
    valid_records = 0
    
    try:
        # Read entire file content at once
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            return records, 0, 0
        
        # Split into lines and process in batch
        lines = content.split('\n')
        total_lines = len(lines)
        
        for line in lines:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                record = process_jsonl_entry(entry, project_name, local_tz, utc_tz)
                if record:
                    records.append(record)
                    valid_records += 1
            except (json.JSONDecodeError, KeyError):
                continue
                
    except Exception as e:
        performance_logger.warning(f"Error processing file {file_path}: {e}")
        
    return records, total_lines, valid_records


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
    start_total = time.time()
    performance_logger.info("Starting load_cost_data()")
    
    # Initialize database
    start_db = time.time()
    db_path = init_database()
    log_timing("Database initialization", start_db)
    
    # Find all JSONL files in the Claude projects directory
    start_scan = time.time()
    project_dir = os.path.expanduser("~/.claude/projects/")
    jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
    log_timing("File scanning", start_scan, files_found=len(jsonl_files))
    
    # Get local timezone
    start_tz = time.time()
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    log_timing("Timezone setup", start_tz)
    
    # Dictionary to store data
    data = defaultdict(lambda: defaultdict(float))
    token_data = defaultdict(lambda: defaultdict(lambda: {'input': 0, 'output': 0, 'cache_creation': 0, 'cache_read': 0, 'total': 0}))
    project_names = set()
    total_cost = 0
    
    start_processing = time.time()
    total_lines_processed = 0
    total_records_processed = 0
    
    # Process files in parallel for better performance
    max_workers = min(8, len(jsonl_files))  # Limit concurrent file operations
    lock = threading.Lock()  # For thread-safe data updates
    
    def process_single_file(file_info):
        """Process a single file and return results."""
        i, file_path = file_info
        start_file = time.time()
        
        try:
            project_name = extract_project_name(file_path)
            # Process entire file in batch
            records, file_lines, file_records = process_jsonl_file_batch(
                file_path, project_name, local_tz, utc_tz
            )
            
            # Thread-safe updates to shared data structures
            with lock:
                project_names.add(project_name)
                
                for record in records:
                    date = record['date']
                    
                    # Add cost to the appropriate date and project
                    data[date][project_name] += record['costUSD']
                    
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
            
            log_timing(f"File processing [{i+1}/{len(jsonl_files)}]", start_file, 
                      file=os.path.basename(file_path), lines=file_lines, records=file_records)
            
            return records, file_lines, file_records
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            log_timing(f"File processing ERROR [{i+1}/{len(jsonl_files)}]", start_file, 
                      file=os.path.basename(file_path), error=str(e))
            return [], 0, 0
    
    # Process files in parallel
    all_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file processing tasks
        future_to_file = {executor.submit(process_single_file, (i, file_path)): file_path 
                         for i, file_path in enumerate(jsonl_files)}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                records, file_lines, file_records = future.result()
                all_records.extend(records)
                total_lines_processed += file_lines
                total_records_processed += file_records
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    # Calculate total cost from all records
    total_cost = sum(record['costUSD'] for record in all_records)
    
    # Optimized database operations with efficient batch processing
    start_db = time.time()
    records_with_uuid = [record for record in all_records if record['uuid']]
    
    if records_with_uuid:
        # Use highly efficient batch insert with executemany
        with get_connection(db_path) as conn:
            insert_records_batch(conn, records_with_uuid)
    
    log_timing("Database batch insert (highly optimized)", start_db, record_count=len(records_with_uuid))
    
    log_timing("File processing (parallel)", start_processing, 
              total_lines=total_lines_processed, total_records=total_records_processed, 
              max_workers=max_workers, file_processing_only=True)
    
    # Convert to DataFrames
    start_dataframe = time.time()
    dates = sorted(data.keys())
    projects = sorted(list(project_names))
    log_timing("Data sorting", start_dataframe, dates=len(dates), projects=len(projects))

    # Create cost DataFrame
    start_cost_df = time.time()
    df = pd.DataFrame(index=dates, columns=projects)
    for date in dates:
        for project in projects:
            df.loc[date, project] = data[date].get(project, 0)
    
    df = df.fillna(0)
    df = df.infer_objects(copy=False)
    log_timing("Cost DataFrame creation", start_cost_df, shape=df.shape)
    
    # Filter out days with zero cost
    start_filter = time.time()
    df['daily_total'] = df.sum(axis=1)
    df = df[df['daily_total'] > 0]
    df = df.drop(columns=['daily_total'])
    log_timing("Cost DataFrame filtering", start_filter, final_shape=df.shape)
    
    # Create token DataFrame
    start_token_df = time.time()
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
    
    log_timing("Token DataFrame creation", start_token_df, shape=token_df.shape)
    
    log_timing("Total load_cost_data()", start_total, 
              final_dates=len(df), final_projects=len(projects), total_cost=f"${total_cost:.4f}")
    
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
    start_refresh = time.time()
    performance_logger.info("Starting refresh_and_load_data()")
    
    # First, scan JSONL files and update database
    start_load = time.time()
    load_cost_data()
    log_timing("JSONL scanning and database update", start_load)
    
    # Then load from database to get complete dataset
    start_db_load = time.time()
    result = load_data_from_database()
    log_timing("Database loading", start_db_load)
    
    log_timing("Total refresh_and_load_data()", start_refresh)
    return result


def load_data_from_database() -> Tuple[pd.DataFrame, List[str], float, pd.DataFrame]:
    """Load usage data from the SQLite database.
    
    Returns:
        Tuple containing:
        - DataFrame with cost data
        - List of project names
        - Total cost
        - DataFrame with token data
    """
    start_total = time.time()
    performance_logger.info("Starting load_data_from_database()")
    
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        # If database doesn't exist, create it by loading from JSONL first
        performance_logger.info("Database doesn't exist, falling back to load_cost_data()")
        return load_cost_data()
    
    # Get local timezone
    start_tz = time.time()
    local_tz = get_local_timezone()
    utc_tz = ZoneInfo('UTC')
    log_timing("Timezone setup (DB load)", start_tz)
    
    # Query all data from database
    start_query = time.time()
    records = fetch_all_records(db_path)
    log_timing("Database query", start_query, records_count=len(records))
    
    # Process records into DataFrames
    start_processing = time.time()
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
    
    log_timing("Record processing (DB load)", start_processing, processed_records=len(records))
    
    # Convert to DataFrames
    start_dataframe = time.time()
    dates = sorted(data.keys())
    projects = sorted(list(project_names))
    log_timing("Data sorting (DB load)", start_dataframe, dates=len(dates), projects=len(projects))
    
    # Cost DataFrame
    start_cost_df = time.time()
    df = pd.DataFrame(index=dates, columns=projects)
    for date in dates:
        for project in projects:
            df.loc[date, project] = data[date].get(project, 0)
    
    df = df.fillna(0)
    df = df.infer_objects(copy=False)
    log_timing("Cost DataFrame creation (DB load)", start_cost_df, shape=df.shape)
    
    # Filter out days with zero cost
    start_filter = time.time()
    df['daily_total'] = df.sum(axis=1)
    df = df[df['daily_total'] > 0]
    df = df.drop(columns=['daily_total'])
    log_timing("Cost DataFrame filtering (DB load)", start_filter, final_shape=df.shape)
    
    # Token DataFrame
    start_token_df = time.time()
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
    
    log_timing("Token DataFrame creation (DB load)", start_token_df, shape=token_df.shape)
    
    log_timing("Total load_data_from_database()", start_total, 
              final_dates=len(df), final_projects=len(projects), total_cost=f"${total_cost:.4f}")
    
    return df, projects, total_cost, token_df


def load_raw_records_from_database() -> List[Dict[str, Any]]:
    """Load raw usage records from the SQLite database for window detection.
    
    Returns:
        List of dictionaries containing raw record data
    """
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        return []
    
    records = fetch_all_records(db_path)
    raw_records = []
    
    for record in records:
        uuid, timestamp, project, cost, model, input_tokens, output_tokens, cache_creation, cache_read, total_tokens, service_tier, duration = record
        
        if timestamp and cost > 0:
            raw_records.append({
                'uuid': uuid,
                'timestamp': timestamp,
                'project': project,
                'costUSD': cost,
                'model': model,
                'input_tokens': input_tokens or 0,
                'output_tokens': output_tokens or 0,
                'cache_creation_input_tokens': cache_creation or 0,
                'cache_read_input_tokens': cache_read or 0,
                'total_tokens': total_tokens or 0,
                'service_tier': service_tier,
                'durationMs': duration
            })
    
    return raw_records


def get_window_analysis() -> Dict[str, Any]:
    """Get credit window analysis.
    
    Returns:
        Dictionary containing window analysis data
    """
    from .database import save_all_windows, load_credit_windows
    
    # First try to load windows from database
    saved_windows = load_credit_windows()
    
    # Load raw records to check if we need to update windows
    raw_records = load_raw_records_from_database()
    
    # If we have records but no saved windows, or if we have new records, recalculate
    need_recalculation = False
    
    if not saved_windows and raw_records:
        need_recalculation = True
    elif saved_windows and raw_records:
        # Check if we have newer records than the last window
        last_window_end = saved_windows[-1]['end_time']
        last_record_time = max(r['timestamp'] for r in raw_records)
        if last_record_time > last_window_end:
            need_recalculation = True
    
    if need_recalculation:
        # Detect windows from scratch
        windows = detect_windows(raw_records)
        
        # Convert to dictionaries and save to database
        window_dicts = [w.to_dict() for w in windows]
        save_all_windows(window_dicts)
        
        # Use the newly detected windows
        analysis = analyze_windows(windows)
        analysis['windows'] = window_dicts
    else:
        # Use saved windows
        # Convert saved windows back to CreditWindow objects for analysis
        from .window_detector import CreditWindow
        windows = []
        for w_dict in saved_windows:
            window = CreditWindow(datetime.fromisoformat(w_dict['start_time']))
            window.end_time = datetime.fromisoformat(w_dict['end_time'])
            window.total_cost = w_dict['total_cost']
            window.opus_cost = w_dict['opus_cost']
            window.sonnet_cost = w_dict['sonnet_cost']
            window.reached_half_credit = w_dict['reached_half_credit']
            window.half_credit_time = datetime.fromisoformat(w_dict['half_credit_time']) if w_dict.get('half_credit_time') else None
            window.half_credit_cost = w_dict.get('half_credit_cost')
            window.messages = w_dict.get('messages', [])
            window.opus_messages = [m for m in window.messages if m.get('model') and 'opus' in m['model'].lower()]
            window.sonnet_messages = [m for m in window.messages if m.get('model') and 'sonnet' in m['model'].lower()]
            windows.append(window)
        
        analysis = analyze_windows(windows)
        analysis['windows'] = saved_windows
    
    return analysis


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