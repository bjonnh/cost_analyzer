#!/usr/bin/env python
"""Test that the fixes work correctly."""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== TESTING FIXES ===\n")

print("1. Testing token data is loaded...")
from cost_analyzer.data_processor import load_data_from_database

df, projects, total_cost, token_df = load_data_from_database()

if token_df is not None and len(token_df) > 0:
    non_zero_cols = sum(1 for col in token_df.columns if token_df[col].sum() > 0)
    print(f"   ✅ Token data loaded: {non_zero_cols} columns with data")
else:
    print("   ❌ No token data found")

print("\n2. Testing predictions date fix...")
if len(df) >= 6:
    from cost_analyzer.predictions import predict_future_costs
    predictions = predict_future_costs(df, days_ahead=7, model_types=['survival'])
    
    if predictions and 'survival' in predictions:
        hist_dates = predictions['survival']['historical_dates']
        # Check if any dates are from 1970
        bad_dates = [d for d in hist_dates if hasattr(d, 'year') and d.year < 2000]
        if bad_dates:
            print(f"   ❌ Found {len(bad_dates)} dates from 1970s")
        else:
            print(f"   ✅ All historical dates are valid (first: {hist_dates[0]}, last: {hist_dates[-1]})")
else:
    print("   ⚠️  Not enough data for predictions")

print("\n3. Testing visualization updates...")
from cost_analyzer.visualizations import create_token_charts
from cost_analyzer.data_processor import calculate_statistics

stats = calculate_statistics(df, projects, token_df)
if token_df is not None:
    charts = create_token_charts(token_df, projects, stats)
    
    if 'token_breakdown' in charts:
        # Check if it's a bar chart (not pie)
        fig = charts['token_breakdown']
        if hasattr(fig, 'data') and len(fig.data) > 0:
            first_trace = fig.data[0]
            if hasattr(first_trace, 'type') and first_trace.type == 'bar':
                print("   ✅ Token breakdown is now a bar chart (not pie)")
            else:
                print(f"   ❌ Token breakdown is still a {first_trace.type}")
    
    if 'token_by_project' in charts:
        fig = charts['token_by_project']
        if hasattr(fig, 'data') and len(fig.data) > 1:
            print(f"   ✅ Token by project shows {len(fig.data)} token types separately")
        else:
            print("   ❌ Token by project not showing separate types")

print("\n✨ All tests complete!")
print("\nYou can now run ./analyzer.py to see the updated dashboard.")