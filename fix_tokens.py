#!/usr/bin/env python
"""Script to fix token data issues by reprocessing JSONL files."""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost_analyzer.data_processor import load_cost_data, load_data_from_database
from cost_analyzer.database import get_database_path
import sqlite3

def check_and_fix():
    """Check for issues and fix them."""
    print("=== TOKEN DATA FIX UTILITY ===\n")
    
    # First, let's see what we have in the database
    db_path = get_database_path()
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count records with and without tokens
        cursor.execute("SELECT COUNT(*) FROM usage_records WHERE total_tokens IS NULL OR total_tokens = 0")
        no_tokens = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM usage_records WHERE total_tokens > 0")
        with_tokens = cursor.fetchone()[0]
        
        print(f"Database status:")
        print(f"  Records without tokens: {no_tokens}")
        print(f"  Records with tokens: {with_tokens}")
        
        conn.close()
        
        if no_tokens > 0 and with_tokens == 0:
            print("\nIt looks like no token data has been extracted.")
            print("This might be because the JSONL files were processed before token tracking was added.")
            
            response = input("\nWould you like to reprocess all JSONL files to extract token data? (y/n): ")
            if response.lower() == 'y':
                print("\nReprocessing JSONL files...")
                # This will reload all data from JSONL files
                df, projects, total_cost, token_df = load_cost_data()
                
                print(f"\nReprocessing complete!")
                print(f"  Projects found: {len(projects)}")
                print(f"  Total cost: ${total_cost:.2f}")
                
                if token_df is not None and len(token_df) > 0:
                    # Check token totals
                    token_totals = {}
                    for col in token_df.columns:
                        if col.endswith('_total'):
                            total = token_df[col].sum()
                            if total > 0:
                                token_totals[col] = total
                    
                    if token_totals:
                        print(f"  Token data extracted successfully!")
                        print(f"  Sample token totals:")
                        for col, total in list(token_totals.items())[:5]:
                            print(f"    {col}: {int(total):,}")
                    else:
                        print("  WARNING: No token totals found after reprocessing!")
                else:
                    print("  WARNING: No token DataFrame created!")
            else:
                print("\nSkipping reprocessing.")
    else:
        print("Database doesn't exist. Running initial data load...")
        df, projects, total_cost, token_df = load_cost_data()
        print(f"\nInitial load complete!")
        print(f"  Projects found: {len(projects)}")
        print(f"  Total cost: ${total_cost:.2f}")
    
    # Now let's verify the current state
    print("\n=== VERIFYING CURRENT STATE ===")
    df, projects, total_cost, token_df = load_data_from_database()
    
    if token_df is None or len(token_df) == 0:
        print("ERROR: Token DataFrame is None or empty!")
        print("\nPossible causes:")
        print("1. JSONL files don't contain 'usage' field")
        print("2. Token extraction logic is failing")
        print("3. Database query is not returning token data")
        
        # Let's check a sample JSONL entry
        import glob
        import json
        
        project_dir = os.path.expanduser("~/.claude/projects/")
        jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
        
        if jsonl_files:
            print(f"\nChecking first JSONL file: {jsonl_files[0]}")
            with open(jsonl_files[0], 'r') as f:
                for i, line in enumerate(f):
                    if i >= 1:  # Just check first entry
                        break
                    try:
                        entry = json.loads(line)
                        print("\nSample entry structure:")
                        print(f"  Keys: {list(entry.keys())}")
                        if 'usage' in entry:
                            print(f"  Usage keys: {list(entry['usage'].keys())}")
                            print(f"  Usage data: {entry['usage']}")
                        else:
                            print("  No 'usage' field found!")
                    except:
                        pass
    else:
        print(f"Token DataFrame shape: {token_df.shape}")
        
        # Find columns with data
        cols_with_data = []
        for col in token_df.columns:
            if token_df[col].sum() > 0:
                cols_with_data.append(col)
        
        if cols_with_data:
            print(f"Columns with token data: {len(cols_with_data)}")
            print("Sample columns with data:")
            for col in cols_with_data[:5]:
                print(f"  {col}: sum={int(token_df[col].sum())}")
        else:
            print("WARNING: All token columns are zero!")
    
    print("\n=== CHECKING TOKEN STATISTICS ===")
    from cost_analyzer.data_processor import calculate_statistics
    
    stats = calculate_statistics(df, projects, token_df)
    
    if 'total_tokens' in stats:
        print(f"Total tokens in stats: {stats['total_tokens']:,}")
        print(f"Cost per 1K tokens: ${stats.get('cost_per_1k_tokens', 0):.4f}")
    else:
        print("No token statistics calculated!")
        print(f"Stats keys: {list(stats.keys())}")


if __name__ == "__main__":
    check_and_fix()