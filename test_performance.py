#!/usr/bin/env python3
"""
Test script to measure performance of data loading operations.
"""

import sys
import os
import time

# Add the current directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run performance tests on data loading."""
    
    print("=== Claude Code Cost Analyzer Performance Test ===")
    print()
    
    # Test 1: Direct data loading
    print("1. Testing refresh_and_load_data()...")
    start_time = time.time()
    
    try:
        from cost_analyzer.data_processor import refresh_and_load_data, calculate_statistics
        from cost_analyzer.predictions import predict_future_costs
        from cost_analyzer.visualizations import create_additional_charts, create_token_charts
        
        # Load data
        df_full, projects, total_cost, token_df_full = refresh_and_load_data()
        data_load_time = time.time()
        print(f"   Data loading: {data_load_time - start_time:.4f}s")
        print(f"   Loaded {len(df_full)} dates, {len(projects)} projects, ${total_cost:.2f} total cost")
        
        if len(df_full) > 0:
            # Calculate statistics
            stats = calculate_statistics(df_full, projects, token_df_full)
            stats_time = time.time()
            print(f"   Statistics calculation: {stats_time - data_load_time:.4f}s")
            
            # Generate predictions
            predictions = predict_future_costs(df_full)
            predictions_time = time.time()
            print(f"   Predictions calculation: {predictions_time - stats_time:.4f}s")
            
            # Create charts
            charts = create_additional_charts(df_full, projects, stats, predictions)
            charts_time = time.time()
            print(f"   Chart creation: {charts_time - predictions_time:.4f}s")
            
            # Create token charts if available
            if token_df_full is not None and len(token_df_full) > 0:
                token_charts = create_token_charts(token_df_full, projects, stats)
                charts.update(token_charts)
                token_charts_time = time.time()
                print(f"   Token chart creation: {token_charts_time - charts_time:.4f}s")
            
            # Total time
            total_time = time.time() - start_time
            print(f"   TOTAL TIME: {total_time:.4f}s")
            
        else:
            print("   No data found!")
            
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=== Performance Log Contents ===")
    if os.path.exists("performance.log"):
        with open("performance.log", "r") as f:
            print(f.read())
    else:
        print("No performance.log found")

if __name__ == "__main__":
    main()