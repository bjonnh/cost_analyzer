#!/usr/bin/env python
"""Test token extraction with a sample JSONL entry."""

import os
import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost_analyzer.data_processor import process_jsonl_entry

def test_extraction():
    """Test token extraction with various JSONL formats."""
    print("=== TESTING TOKEN EXTRACTION ===\n")
    
    local_tz = ZoneInfo('UTC')
    utc_tz = ZoneInfo('UTC')
    
    # Test case 1: Complete entry with all token fields
    print("Test 1: Complete entry with all fields")
    entry1 = {
        'id': 'test-123',
        'costUSD': 0.05,
        'timestamp': '2024-01-01T12:00:00.000Z',
        'usage': {
            'input_tokens': 100,
            'output_tokens': 50,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 5,
            'service_tier': 'standard'
        }
    }
    
    result1 = process_jsonl_entry(entry1, 'test-project', local_tz, utc_tz)
    print(f"Result: {json.dumps(result1, indent=2)}")
    
    # Test case 2: Entry with partial token data
    print("\n\nTest 2: Entry with partial token data")
    entry2 = {
        'id': 'test-456',
        'costUSD': 0.03,
        'timestamp': '2024-01-01T13:00:00.000Z',
        'usage': {
            'input_tokens': 200,
            'output_tokens': 100
            # Missing cache tokens
        }
    }
    
    result2 = process_jsonl_entry(entry2, 'test-project', local_tz, utc_tz)
    if result2:
        print(f"Total tokens calculated: {result2['total_tokens']}")
        print(f"Input tokens: {result2['input_tokens']}")
        print(f"Output tokens: {result2['output_tokens']}")
        print(f"Cache creation tokens: {result2['cache_creation_input_tokens']}")
    
    # Test case 3: Check actual JSONL file
    print("\n\nTest 3: Checking actual JSONL files")
    import glob
    
    project_dir = os.path.expanduser("~/.claude/projects/")
    jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
    
    if jsonl_files:
        print(f"Found {len(jsonl_files)} JSONL files")
        
        # Check first file with content
        for file_path in jsonl_files[:3]:
            print(f"\nChecking: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    found_usage = False
                    for i, line in enumerate(f):
                        if i >= 5:  # Check first 5 entries
                            break
                        try:
                            entry = json.loads(line)
                            if 'usage' in entry and entry.get('costUSD', 0) > 0:
                                found_usage = True
                                print(f"  Entry {i}: Found usage data")
                                usage = entry['usage']
                                print(f"    Usage fields: {list(usage.keys())}")
                                
                                # Process this entry
                                project_name = os.path.basename(os.path.dirname(file_path))
                                result = process_jsonl_entry(entry, project_name, local_tz, utc_tz)
                                if result:
                                    print(f"    Extracted tokens: {result.get('total_tokens', 0)}")
                                break
                        except json.JSONDecodeError:
                            pass
                    
                    if not found_usage:
                        print(f"  No usage data found in first 5 entries")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print("No JSONL files found!")


if __name__ == "__main__":
    test_extraction()