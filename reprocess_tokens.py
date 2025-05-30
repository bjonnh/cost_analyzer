#!/usr/bin/env python
"""Reprocess all JSONL files to extract token data correctly."""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=== REPROCESSING TOKEN DATA ===\n")
    print("This will reload all JSONL files and extract token data correctly.")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print("\nReprocessing...")
    
    from cost_analyzer.data_processor import load_cost_data
    
    # This will reprocess all JSONL files with the fixed token extraction
    df, projects, total_cost, token_df = load_cost_data()
    
    print(f"\nReprocessing complete!")
    print(f"Projects: {projects}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Cost data shape: {df.shape}")
    print(f"Token data shape: {token_df.shape if token_df is not None else 'None'}")
    
    if token_df is not None and len(token_df) > 0:
        # Check for non-zero token columns
        non_zero_cols = []
        for col in token_df.columns:
            total = token_df[col].sum()
            if total > 0:
                non_zero_cols.append((col, total))
        
        if non_zero_cols:
            print(f"\nToken data extracted successfully!")
            print(f"Columns with token data: {len(non_zero_cols)}")
            print("\nSample token totals:")
            for col, total in sorted(non_zero_cols, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {col}: {int(total):,}")
        else:
            print("\nWARNING: Token columns exist but all values are zero!")
    else:
        print("\nERROR: No token data was extracted!")
    
    # Verify database update
    print("\n=== VERIFYING DATABASE UPDATE ===")
    
    import sqlite3
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as records_with_tokens,
               SUM(total_tokens) as total_tokens
        FROM usage_records
        WHERE total_tokens > 0
    """)
    
    result = cursor.fetchone()
    print(f"Records with tokens in database: {result[0]}")
    print(f"Total tokens in database: {result[1]:,}" if result[1] else "Total tokens in database: 0")
    
    conn.close()
    
    print("\nYou can now run the dashboard to see token data in the Tokens tab!")


if __name__ == "__main__":
    main()