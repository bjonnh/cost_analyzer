#!/usr/bin/env python
"""Automatically fix token data by reprocessing JSONL files."""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== FIXING TOKEN DATA ===")
print("Reprocessing all JSONL files with corrected token extraction...\n")

from cost_analyzer.data_processor import load_cost_data

# Reprocess all JSONL files
df, projects, total_cost, token_df = load_cost_data()

print(f"✅ Reprocessing complete!")
print(f"   Projects: {len(projects)}")
print(f"   Total cost: ${total_cost:.2f}")

if token_df is not None and len(token_df) > 0:
    # Count token data
    token_count = 0
    for col in token_df.columns:
        if col.endswith('_total') and token_df[col].sum() > 0:
            token_count += int(token_df[col].sum())
    
    if token_count > 0:
        print(f"   Total tokens: {token_count:,}")
        print("\n✅ Token data successfully extracted!")
        print("   You can now run ./analyzer.py to see token data in the dashboard.")
    else:
        print("\n⚠️  No token data found in JSONL files.")
        print("   This might be normal if your JSONL files are from an older version.")
else:
    print("\n❌ Failed to extract token data.")

# Quick database check
import sqlite3
db_path = os.path.expanduser("~/.claude/usage_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM usage_records WHERE total_tokens > 0")
records_with_tokens = cursor.fetchone()[0]
conn.close()

print(f"\nDatabase records with tokens: {records_with_tokens}")