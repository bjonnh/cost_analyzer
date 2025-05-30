#!/usr/bin/env python
"""Quick verification that token data is now working."""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost_analyzer.data_processor import load_data_from_database, calculate_statistics

def main():
    print("=== VERIFYING TOKEN DATA ===\n")
    
    # Load data
    df, projects, total_cost, token_df = load_data_from_database()
    
    print(f"Projects: {projects}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Token DataFrame: {'EXISTS' if token_df is not None else 'NONE'}")
    
    if token_df is not None:
        print(f"Token DataFrame shape: {token_df.shape}")
        
        # Find non-zero columns
        non_zero = 0
        total_tokens = 0
        
        for col in token_df.columns:
            col_sum = token_df[col].sum()
            if col_sum > 0:
                non_zero += 1
                if col.endswith('_total'):
                    total_tokens += col_sum
        
        print(f"Non-zero token columns: {non_zero}")
        print(f"Total tokens across all projects: {int(total_tokens):,}")
        
        # Calculate statistics
        stats = calculate_statistics(df, projects, token_df)
        
        if 'total_tokens' in stats:
            print(f"\nToken Statistics:")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Average daily tokens: {stats.get('avg_daily_tokens', 0):,}")
            print(f"  Cost per 1K tokens: ${stats.get('cost_per_1k_tokens', 0):.4f}")
            print(f"  Input tokens: {stats.get('total_input_tokens', 0):,}")
            print(f"  Output tokens: {stats.get('total_output_tokens', 0):,}")
            print("\n✅ Token data is working correctly!")
        else:
            print("\n❌ Token statistics not calculated - check if token columns have data")
    else:
        print("\n❌ No token DataFrame loaded - reprocess data first")


if __name__ == "__main__":
    main()