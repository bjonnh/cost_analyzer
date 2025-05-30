"""Database module for Claude Code Cost Analyzer."""

import os
import sqlite3
from typing import Dict, Any, Optional
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