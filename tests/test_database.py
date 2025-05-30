"""Tests for the database module."""

import os
import tempfile
import sqlite3
from pathlib import Path
import pytest

from cost_analyzer.database import (
    init_database, insert_or_update_record, fetch_all_records,
    get_database_path, get_connection
)


class TestDatabase:
    """Test database operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_init_database_creates_file(self, monkeypatch, tmp_path):
        """Test that init_database creates the database file."""
        # Mock the home directory
        mock_home = tmp_path / "home"
        mock_home.mkdir()
        monkeypatch.setattr(os.path, 'expanduser', lambda x: str(mock_home / x.replace('~/', '')))
        
        db_path = init_database()
        
        assert os.path.exists(db_path)
        assert db_path.endswith('usage_data.db')
        
        # Check tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            assert ('usage_records',) in tables
    
    def test_insert_or_update_record(self, temp_db_path):
        """Test inserting and updating records."""
        # Initialize database
        with get_connection(temp_db_path) as conn:
            cursor = conn.cursor()
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
            conn.commit()
        
        # Test data
        record_data = {
            'uuid': 'test-uuid-123',
            'costUSD': 0.05,
            'durationMs': 1500,
            'model': 'claude-3-sonnet',
            'timestamp': '2024-01-01T00:00:00.000Z',
            'project': 'test-project',
            'input_tokens': 100,
            'output_tokens': 50,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 5,
            'total_tokens': 165,
            'service_tier': 'standard'
        }
        
        # Insert record
        with get_connection(temp_db_path) as conn:
            insert_or_update_record(conn, record_data)
            
            # Verify insertion
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM usage_records WHERE uuid = ?", (record_data['uuid'],))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == 'test-uuid-123'  # uuid
            assert result[1] == 0.05  # costUSD
            assert result[2] == 1500  # durationMs
            assert result[3] == 'claude-3-sonnet'  # model
            assert result[6] == 100  # input_tokens
            assert result[7] == 50  # output_tokens
        
        # Update record
        record_data['costUSD'] = 0.10
        record_data['output_tokens'] = 75
        
        with get_connection(temp_db_path) as conn:
            insert_or_update_record(conn, record_data)
            
            # Verify update
            cursor = conn.cursor()
            cursor.execute("SELECT costUSD, output_tokens FROM usage_records WHERE uuid = ?", 
                         (record_data['uuid'],))
            result = cursor.fetchone()
            
            assert result[0] == 0.10
            assert result[1] == 75
    
    def test_fetch_all_records(self, temp_db_path):
        """Test fetching all records."""
        # Initialize and populate database
        with get_connection(temp_db_path) as conn:
            cursor = conn.cursor()
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
            
            # Insert test records
            test_records = [
                ('uuid1', '2024-01-01T00:00:00Z', 'project1', 0.05, 'claude-3', 100, 50, 10, 5, 165, 'standard', 1000),
                ('uuid2', '2024-01-02T00:00:00Z', 'project2', 0.10, 'claude-3', 200, 100, 20, 10, 330, 'standard', 2000),
                ('uuid3', '2024-01-03T00:00:00Z', 'project1', 0.00, 'claude-3', 0, 0, 0, 0, 0, 'standard', 500),
            ]
            
            for record in test_records:
                cursor.execute("""
                    INSERT INTO usage_records 
                    (uuid, timestamp, project, costUSD, model, input_tokens, output_tokens,
                     cache_creation_input_tokens, cache_read_input_tokens, total_tokens, 
                     service_tier, durationMs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, record)
            
            conn.commit()
        
        # Fetch records
        records = fetch_all_records(temp_db_path)
        
        # Should only return records with costUSD > 0
        assert len(records) == 2
        assert records[0][0] == 'uuid1'  # uuid
        assert records[1][0] == 'uuid2'  # uuid
        
        # Verify ordering by timestamp
        assert records[0][1] == '2024-01-01T00:00:00Z'
        assert records[1][1] == '2024-01-02T00:00:00Z'
    
    def test_get_database_path(self):
        """Test getting the default database path."""
        path = get_database_path()
        assert path.endswith('.claude/usage_data.db')
        assert '~' not in path  # Should be expanded
    
    def test_get_connection_context_manager(self, temp_db_path):
        """Test the connection context manager."""
        # Create a table using the context manager
        with get_connection(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test_table (id INTEGER)")
            conn.commit()
        
        # Verify table exists in a new connection
        with get_connection(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 'test_table'