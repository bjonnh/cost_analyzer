"""Database module for Claude Code Cost Analyzer."""

import os
import sqlite3
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


@contextmanager
def get_connection(db_path: str):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> str:
    """Initialize SQLite database for storing Claude Code usage data.
    
    Returns:
        str: Path to the initialized database
    """
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with get_connection(db_path) as conn:
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
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_records(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_project ON usage_records(project)
        """)
        
        # Create windows table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS credit_windows (
                window_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                total_cost REAL NOT NULL,
                opus_cost REAL NOT NULL,
                sonnet_cost REAL NOT NULL,
                total_messages INTEGER NOT NULL,
                opus_messages INTEGER NOT NULL,
                sonnet_messages INTEGER NOT NULL,
                reached_half_credit BOOLEAN NOT NULL,
                half_credit_time TEXT,
                half_credit_cost REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create window messages table (to track messages in each window)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS window_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id TEXT NOT NULL,
                message_uuid TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model TEXT,
                cost REAL NOT NULL,
                tokens INTEGER,
                FOREIGN KEY (window_id) REFERENCES credit_windows(window_id),
                FOREIGN KEY (message_uuid) REFERENCES usage_records(uuid)
            )
        """)
        
        # Create indexes for window tables
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_window_start_time ON credit_windows(start_time)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_window_messages_window_id ON window_messages(window_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_window_messages_uuid ON window_messages(message_uuid)
        """)
        
        conn.commit()
    
    return db_path


def insert_or_update_record(conn: sqlite3.Connection, record_data: Dict[str, Any]) -> None:
    """Insert or update a usage record in the database.
    
    Args:
        conn: SQLite connection
        record_data: Dictionary containing record data
    """
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
    
    # Note: commit is now handled by caller for better batching performance


def insert_records_batch(conn: sqlite3.Connection, records: list) -> None:
    """Insert or update multiple records efficiently in a single transaction.
    
    Args:
        conn: SQLite connection
        records: List of record dictionaries
    """
    if not records:
        return
        
    cursor = conn.cursor()
    
    # Prepare data for executemany
    record_tuples = []
    for record_data in records:
        record_tuples.append((
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
    
    # Use executemany for much better performance
    cursor.executemany("""
        INSERT OR REPLACE INTO usage_records 
        (uuid, costUSD, durationMs, model, timestamp, project,
         input_tokens, output_tokens, cache_creation_input_tokens,
         cache_read_input_tokens, total_tokens, service_tier, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, record_tuples)
    
    conn.commit()


def fetch_all_records(db_path: str) -> list:
    """Fetch all records from the database.
    
    Args:
        db_path: Path to the database
        
    Returns:
        list: List of all records
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT 
                uuid, timestamp, project, costUSD, model,
                input_tokens, output_tokens, 
                cache_creation_input_tokens, cache_read_input_tokens,
                total_tokens, service_tier, durationMs
            FROM usage_records
            WHERE costUSD > 0
            ORDER BY timestamp
        """
        
        cursor.execute(query)
        return cursor.fetchall()


def get_database_path() -> str:
    """Get the default database path.
    
    Returns:
        str: Path to the database
    """
    return os.path.expanduser("~/.claude/usage_data.db")


def save_credit_window(conn: sqlite3.Connection, window: Dict[str, Any]) -> None:
    """Save a credit window to the database.
    
    Args:
        conn: SQLite connection
        window: Window data dictionary from CreditWindow.to_dict()
    """
    cursor = conn.cursor()
    
    # Generate window ID from start time
    window_id = f"window_{window['start_time']}"
    
    # Insert or update window
    cursor.execute("""
        INSERT OR REPLACE INTO credit_windows
        (window_id, start_time, end_time, total_cost, opus_cost, sonnet_cost,
         total_messages, opus_messages, sonnet_messages, reached_half_credit,
         half_credit_time, half_credit_cost, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        window_id,
        window['start_time'],
        window['end_time'],
        window['total_cost'],
        window['opus_cost'],
        window['sonnet_cost'],
        window['total_messages'],
        window['opus_messages'],
        window['sonnet_messages'],
        window['reached_half_credit'],
        window.get('half_credit_time'),
        window.get('half_credit_cost')
    ))
    
    # Delete existing messages for this window (to handle updates)
    cursor.execute("DELETE FROM window_messages WHERE window_id = ?", (window_id,))
    
    # Insert window messages
    messages = window.get('messages', [])
    for msg in messages:
        cursor.execute("""
            INSERT INTO window_messages
            (window_id, message_uuid, timestamp, model, cost, tokens)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            window_id,
            msg.get('uuid', ''),
            msg.get('timestamp'),
            msg.get('model'),
            msg.get('cost', 0),
            msg.get('tokens', 0)
        ))


def save_all_windows(windows: List[Dict[str, Any]]) -> None:
    """Save all credit windows to the database.
    
    Args:
        windows: List of window dictionaries
    """
    db_path = get_database_path()
    
    with get_connection(db_path) as conn:
        for window in windows:
            save_credit_window(conn, window)
        conn.commit()


def load_credit_windows() -> List[Dict[str, Any]]:
    """Load all credit windows from the database.
    
    Returns:
        List of window dictionaries
    """
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        return []
    
    windows = []
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Load windows
        cursor.execute("""
            SELECT window_id, start_time, end_time, total_cost, opus_cost, sonnet_cost,
                   total_messages, opus_messages, sonnet_messages, reached_half_credit,
                   half_credit_time, half_credit_cost
            FROM credit_windows
            ORDER BY start_time
        """)
        
        window_rows = cursor.fetchall()
        
        for row in window_rows:
            window_id, start_time, end_time, total_cost, opus_cost, sonnet_cost, \
            total_messages, opus_messages, sonnet_messages, reached_half_credit, \
            half_credit_time, half_credit_cost = row
            
            # Load messages for this window
            cursor.execute("""
                SELECT message_uuid, timestamp, model, cost, tokens
                FROM window_messages
                WHERE window_id = ?
                ORDER BY timestamp
            """, (window_id,))
            
            messages = []
            for msg_row in cursor.fetchall():
                uuid, timestamp, model, cost, tokens = msg_row
                messages.append({
                    'uuid': uuid,
                    'timestamp': timestamp,
                    'model': model,
                    'cost': cost,
                    'tokens': tokens
                })
            
            window = {
                'start_time': start_time,
                'end_time': end_time,
                'total_cost': total_cost,
                'opus_cost': opus_cost,
                'sonnet_cost': sonnet_cost,
                'total_messages': total_messages,
                'opus_messages': opus_messages,
                'sonnet_messages': sonnet_messages,
                'reached_half_credit': bool(reached_half_credit),
                'half_credit_time': half_credit_time,
                'half_credit_cost': half_credit_cost,
                'messages': messages
            }
            
            windows.append(window)
    
    return windows