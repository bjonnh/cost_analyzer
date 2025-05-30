#!/usr/bin/env python
"""Display token data directly from database."""

import os
import sqlite3
import pandas as pd

def show_token_data():
    """Display token data from database."""
    db_path = os.path.expanduser("~/.claude/usage_data.db")
    
    if not os.path.exists(db_path):
        print("Database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    
    print("=== TOKEN DATA FROM DATABASE ===\n")
    
    # Query 1: Summary of token data
    print("1. Token Data Summary:")
    query1 = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN total_tokens > 0 THEN 1 END) as records_with_tokens,
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(cache_creation_input_tokens) as total_cache_creation,
            SUM(cache_read_input_tokens) as total_cache_read,
            SUM(total_tokens) as grand_total
        FROM usage_records
    """
    
    df1 = pd.read_sql_query(query1, conn)
    print(df1.to_string())
    
    # Query 2: Token data by project
    print("\n\n2. Token Data by Project:")
    query2 = """
        SELECT 
            project,
            COUNT(*) as records,
            SUM(total_tokens) as total_tokens,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens
        FROM usage_records
        WHERE total_tokens > 0
        GROUP BY project
        ORDER BY total_tokens DESC
    """
    
    df2 = pd.read_sql_query(query2, conn)
    if len(df2) > 0:
        print(df2.to_string())
    else:
        print("No token data by project!")
    
    # Query 3: Recent records with tokens
    print("\n\n3. Recent Records with Tokens:")
    query3 = """
        SELECT 
            timestamp,
            project,
            costUSD,
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            total_tokens
        FROM usage_records
        WHERE total_tokens > 0
        ORDER BY timestamp DESC
        LIMIT 10
    """
    
    df3 = pd.read_sql_query(query3, conn)
    if len(df3) > 0:
        print(df3.to_string())
    else:
        print("No records with tokens found!")
    
    # Query 4: Check for null vs zero tokens
    print("\n\n4. Token Data Status:")
    query4 = """
        SELECT 
            CASE 
                WHEN total_tokens IS NULL THEN 'NULL tokens'
                WHEN total_tokens = 0 THEN 'Zero tokens'
                ELSE 'Has tokens'
            END as status,
            COUNT(*) as count
        FROM usage_records
        GROUP BY status
    """
    
    df4 = pd.read_sql_query(query4, conn)
    print(df4.to_string())
    
    # Query 5: Sample of records to see what data looks like
    print("\n\n5. Sample Records (first 5):")
    query5 = """
        SELECT 
            uuid,
            timestamp,
            project,
            costUSD,
            model,
            service_tier,
            input_tokens,
            output_tokens,
            total_tokens
        FROM usage_records
        LIMIT 5
    """
    
    df5 = pd.read_sql_query(query5, conn)
    print(df5.to_string())
    
    conn.close()
    
    # Now let's check what the data processor returns
    print("\n\n=== DATA PROCESSOR OUTPUT ===")
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from cost_analyzer.data_processor import load_data_from_database
    
    df, projects, total_cost, token_df = load_data_from_database()
    
    print(f"\nData processor results:")
    print(f"  Cost DF shape: {df.shape}")
    print(f"  Token DF shape: {token_df.shape if token_df is not None else 'None'}")
    print(f"  Projects: {projects}")
    
    if token_df is not None:
        print(f"\nToken DF info:")
        print(f"  Columns: {list(token_df.columns[:10])}")  # First 10 columns
        print(f"  Index (dates): {list(token_df.index[:5])}")  # First 5 dates
        
        # Check for non-zero columns
        non_zero_cols = []
        for col in token_df.columns:
            if token_df[col].sum() > 0:
                non_zero_cols.append((col, token_df[col].sum()))
        
        if non_zero_cols:
            print(f"\nColumns with data:")
            for col, total in non_zero_cols[:10]:
                print(f"  {col}: {total}")
        else:
            print("\nAll token columns are zero!")


if __name__ == "__main__":
    show_token_data()