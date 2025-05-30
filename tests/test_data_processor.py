"""Tests for the data processor module."""

import os
import json
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
import pandas as pd
import numpy as np

from cost_analyzer.data_processor import (
    get_local_timezone, extract_project_name, process_jsonl_entry,
    load_cost_data, load_data_from_database, calculate_statistics
)


class TestDataProcessor:
    """Test data processing functions."""
    
    def test_get_local_timezone(self):
        """Test getting local timezone."""
        tz = get_local_timezone()
        assert isinstance(tz, ZoneInfo)
    
    def test_extract_project_name(self):
        """Test project name extraction from various paths."""
        test_cases = [
            ("/home/user/.claude/projects/my-project/data.jsonl", "my-project"),
            ("/Users/john/.claude/projects/test-app/logs/file.jsonl", "test-app"),
            ("C:\\Users\\john\\.claude\\projects\\windows-app\\data.jsonl", "windows-app"),
            ("/home/user/.claude/projects/-Users-john-project/data.jsonl", "project"),
            ("/some/other/path/unknown/data.jsonl", "unknown"),
        ]
        
        for path, expected in test_cases:
            result = extract_project_name(path)
            assert result == expected
    
    def test_process_jsonl_entry(self):
        """Test processing a single JSONL entry."""
        local_tz = ZoneInfo('UTC')
        utc_tz = ZoneInfo('UTC')
        
        # Valid entry with all fields
        entry = {
            'id': 'test-uuid-123',
            'costUSD': 0.05,
            'timestamp': '2024-01-01T12:00:00.000Z',
            'durationMs': 1500,
            'message': {'model': 'claude-3-sonnet'},
            'usage': {
                'input_tokens': 100,
                'output_tokens': 50,
                'cache_creation_input_tokens': 10,
                'cache_read_input_tokens': 5,
                'service_tier': 'standard'
            }
        }
        
        result = process_jsonl_entry(entry, 'test-project', local_tz, utc_tz)
        
        assert result is not None
        assert result['uuid'] == 'test-uuid-123'
        assert result['costUSD'] == 0.05
        assert result['project'] == 'test-project'
        assert result['model'] == 'claude-3-sonnet'
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['total_tokens'] == 165
        assert result['date'] == '2024-01-01'
        
        # Entry with no cost
        entry_no_cost = {'id': 'test-uuid', 'timestamp': '2024-01-01T00:00:00Z'}
        result = process_jsonl_entry(entry_no_cost, 'test-project', local_tz, utc_tz)
        assert result is None
        
        # Entry with no timestamp
        entry_no_timestamp = {'id': 'test-uuid', 'costUSD': 0.05}
        result = process_jsonl_entry(entry_no_timestamp, 'test-project', local_tz, utc_tz)
        assert result is None
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        # Create test DataFrame
        dates = pd.date_range('2024-01-01', periods=7, freq='D').strftime('%Y-%m-%d')
        data = {
            'project1': [10, 20, 15, 0, 25, 30, 20],
            'project2': [5, 10, 20, 15, 0, 10, 5]
        }
        df = pd.DataFrame(data, index=dates)
        projects = ['project1', 'project2']
        
        # Create test token DataFrame
        token_data = {}
        for date in dates:
            for project in projects:
                token_data[f"{project}_total"] = [100, 200, 150, 50, 250, 300, 200]
                token_data[f"{project}_input"] = [60, 120, 90, 30, 150, 180, 120]
                token_data[f"{project}_output"] = [40, 80, 60, 20, 100, 120, 80]
        
        token_df = pd.DataFrame(token_data, index=dates)
        
        stats = calculate_statistics(df, projects, token_df)
        
        # Check cost statistics
        assert stats['total_cost'] == 185.0
        assert stats['active_days'] == 7
        assert stats['avg_daily_cost'] == pytest.approx(185.0 / 7)
        assert stats['median_daily_cost'] == 20.0
        assert stats['max_daily_cost'] == 40.0
        assert stats['max_daily_date'] == '2024-01-06'
        assert stats['top_project'] == 'project1'
        assert stats['top_project_cost'] == 120.0
        
        # Check hourly statistics
        assert stats['avg_hourly_cost'] == pytest.approx(stats['avg_daily_cost'] / 8)
        assert stats['max_hourly_cost'] == pytest.approx(40.0 / 8)
        
        # Check token statistics
        assert 'total_tokens' in stats
        assert 'avg_daily_tokens' in stats
        assert 'cost_per_1k_tokens' in stats
    
    def test_load_cost_data_integration(self, tmp_path, monkeypatch):
        """Test loading cost data from JSONL files."""
        # Create temporary project directory structure
        projects_dir = tmp_path / ".claude" / "projects"
        project1_dir = projects_dir / "test-project1"
        project2_dir = projects_dir / "test-project2" / "logs"
        project1_dir.mkdir(parents=True)
        project2_dir.mkdir(parents=True)
        
        # Mock expanduser
        monkeypatch.setattr(os.path, 'expanduser', lambda x: str(tmp_path / x.replace('~/', '')))
        
        # Create test JSONL files
        test_data1 = [
            {
                'id': 'uuid1',
                'costUSD': 0.05,
                'timestamp': '2024-01-01T12:00:00.000Z',
                'usage': {
                    'input_tokens': 100,
                    'output_tokens': 50,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0
                }
            },
            {
                'id': 'uuid2',
                'costUSD': 0.10,
                'timestamp': '2024-01-01T14:00:00.000Z',
                'usage': {
                    'input_tokens': 200,
                    'output_tokens': 100
                }
            },
            {
                'id': 'uuid3',
                'costUSD': 0.0,  # Should be ignored
                'timestamp': '2024-01-02T10:00:00.000Z'
            }
        ]
        
        test_data2 = [
            {
                'id': 'uuid4',
                'costUSD': 0.08,
                'timestamp': '2024-01-02T15:00:00.000Z',
                'usage': {
                    'input_tokens': 150,
                    'output_tokens': 80
                }
            }
        ]
        
        # Write JSONL files
        with open(project1_dir / "data.jsonl", 'w') as f:
            for entry in test_data1:
                f.write(json.dumps(entry) + '\n')
        
        with open(project2_dir / "logs.jsonl", 'w') as f:
            for entry in test_data2:
                f.write(json.dumps(entry) + '\n')
        
        # Load data
        df, projects, total_cost, token_df = load_cost_data()
        
        # Verify results
        assert len(projects) == 2
        assert 'test-project1' in projects
        assert 'test-project2' in projects
        
        assert len(df) == 2  # Should have 2 days of data
        assert total_cost == pytest.approx(0.23)  # 0.05 + 0.10 + 0.08
        
        # Check token data
        assert len(token_df) == 2
        assert 'test-project1_input' in token_df.columns
        assert 'test-project2_output' in token_df.columns
        
        # Check aggregated values
        assert df.loc['2024-01-01', 'test-project1'] == pytest.approx(0.15)
        assert df.loc['2024-01-02', 'test-project2'] == pytest.approx(0.08)
    
    def test_calculate_statistics_edge_cases(self):
        """Test statistics calculation with edge cases."""
        # Empty DataFrame
        df_empty = pd.DataFrame()
        stats = calculate_statistics(df_empty, [])
        assert stats['total_cost'] == 0
        assert stats['active_days'] == 0
        assert stats['top_project'] == 'N/A'
        
        # Single day, single project
        df_single = pd.DataFrame({'project1': [10.0]}, index=['2024-01-01'])
        stats = calculate_statistics(df_single, ['project1'])
        assert stats['total_cost'] == 10.0
        assert stats['active_days'] == 1
        assert stats['avg_daily_cost'] == 10.0
        assert stats['median_daily_cost'] == 10.0