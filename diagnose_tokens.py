#!/usr/bin/env python
"""Diagnostic script to check token data extraction and storage."""

import os
import json
import glob
import sqlite3
from collections import defaultdict
import pandas as pd

def check_jsonl_files():
    """Check JSONL files for token data."""
    print("=== CHECKING JSONL FILES ===")
    project_dir = os.path.expanduser("~/.claude/projects/")
    jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    token_entries = 0
    sample_entries = []
    
    for file_path in jsonl_files[:5]:  # Check first 5 files
        print(f"\nChecking file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Only check first 3 entries per file
                        break
                    try:
                        entry = json.loads(line)
                        if 'usage' in entry:
                            token_entries += 1
                            print(f"  Entry {i}: Found usage data:")
                            usage = entry['usage']
                            print(f"    - input_tokens: {usage.get('input_tokens', 'N/A')}")
                            print(f"    - output_tokens: {usage.get('output_tokens', 'N/A')}")
                            print(f"    - cache_creation_input_tokens: {usage.get('cache_creation_input_tokens', 'N/A')}")
                            print(f"    - cache_read_input_tokens: {usage.get('cache_read_input_tokens', 'N/A')}")
                            print(f"    - service_tier: {usage.get('service_tier', 'N/A')}")
                            
                            if len(sample_entries) < 3:
                                sample_entries.append(entry)
                        else:
                            print(f"  Entry {i}: No 'usage' field found")
                            print(f"    Available fields: {list(entry.keys())}")
                    except json.JSONDecodeError:
                        print(f"  Entry {i}: Failed to parse JSON")
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    print(f"\nTotal entries with token data found: {token_entries}")
    return sample_entries


def check_database():
    """Check database for token data."""
    print("\n=== CHECKING DATABASE ===")
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    
    if not os.path.exists(db_path):
        print("Database does not exist!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check schema
    cursor.execute("PRAGMA table_info(usage_records)")
    columns = cursor.fetchall()
    print("Database columns:")
    token_columns = []
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
        if 'token' in col[1]:
            token_columns.append(col[1])
    
    print(f"\nToken-related columns: {token_columns}")
    
    # Check for records with token data
    cursor.execute("""
        SELECT COUNT(*) FROM usage_records 
        WHERE input_tokens IS NOT NULL 
           OR output_tokens IS NOT NULL
           OR cache_creation_input_tokens IS NOT NULL
           OR cache_read_input_tokens IS NOT NULL
    """)
    token_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM usage_records")
    total_records = cursor.fetchone()[0]
    
    print(f"\nTotal records: {total_records}")
    print(f"Records with token data: {token_records}")
    
    # Get sample records with tokens
    cursor.execute("""
        SELECT project, timestamp, 
               input_tokens, output_tokens, 
               cache_creation_input_tokens, cache_read_input_tokens,
               total_tokens
        FROM usage_records 
        WHERE total_tokens > 0
        LIMIT 5
    """)
    samples = cursor.fetchall()
    
    if samples:
        print("\nSample records with tokens:")
        for sample in samples:
            print(f"  Project: {sample[0]}, Time: {sample[1]}")
            print(f"    Input: {sample[2]}, Output: {sample[3]}")
            print(f"    Cache Create: {sample[4]}, Cache Read: {sample[5]}")
            print(f"    Total: {sample[6]}")
    
    conn.close()


def check_data_loading():
    """Check data loading functions."""
    print("\n=== CHECKING DATA LOADING ===")
    
    try:
        from cost_analyzer.data_processor import load_data_from_database
        
        df, projects, total_cost, token_df = load_data_from_database()
        
        print(f"Cost DataFrame shape: {df.shape}")
        print(f"Token DataFrame shape: {token_df.shape if token_df is not None else 'None'}")
        print(f"Projects: {projects}")
        
        if token_df is not None and len(token_df) > 0:
            print("\nToken DataFrame columns:")
            for col in token_df.columns[:10]:  # First 10 columns
                print(f"  - {col}")
            
            print("\nToken data summary:")
            # Check if any token columns have non-zero values
            for col in token_df.columns:
                if token_df[col].sum() > 0:
                    print(f"  {col}: sum={token_df[col].sum()}, max={token_df[col].max()}")
        else:
            print("\nNo token data loaded!")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()


def check_token_aggregation():
    """Check how tokens are being aggregated."""
    print("\n=== CHECKING TOKEN AGGREGATION ===")
    
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    conn = sqlite3.connect(db_path)
    
    # Check raw token data by date and project
    query = """
        SELECT 
            DATE(timestamp) as date,
            project,
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(cache_creation_input_tokens) as total_cache_create,
            SUM(cache_read_input_tokens) as total_cache_read,
            SUM(total_tokens) as grand_total
        FROM usage_records
        WHERE total_tokens > 0
        GROUP BY DATE(timestamp), project
        ORDER BY date DESC, grand_total DESC
        LIMIT 10
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    if results:
        print("Token totals by date and project:")
        for row in results:
            print(f"  {row[0]} - {row[1]}:")
            print(f"    Input: {row[2]}, Output: {row[3]}, Cache: {row[4]}/{row[5]}, Total: {row[6]}")
    else:
        print("No aggregated token data found!")
    
    conn.close()


def main():
    """Run all diagnostic checks."""
    print("Claude Code Cost Analyzer - Token Data Diagnostics")
    print("=" * 60)
    
    # Check JSONL files first
    sample_entries = check_jsonl_files()
    
    # Check database
    check_database()
    
    # Check data loading
    check_data_loading()
    
    # Check aggregation
    check_token_aggregation()
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")


if __name__ == "__main__":
    main()