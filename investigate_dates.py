#!/usr/bin/env python
"""Investigate date issues causing 1970 dates in predictions."""

import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost_analyzer.data_processor import load_data_from_database
from cost_analyzer.predictions import predict_future_costs

def investigate_dates():
    """Check for date issues in the data."""
    print("=== INVESTIGATING DATE ISSUES ===\n")
    
    # Load data
    df, projects, total_cost, token_df = load_data_from_database()
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Date index type: {type(df.index)}")
    print(f"First few dates: {list(df.index[:5])}")
    print(f"Last few dates: {list(df.index[-5:])}")
    
    # Check date parsing
    print("\nChecking date parsing:")
    dates = pd.to_datetime(df.index)
    print(f"Parsed dates type: {type(dates)}")
    print(f"First parsed date: {dates[0]}")
    print(f"Last parsed date: {dates[-1]}")
    
    # Check for any weird dates
    print("\nChecking for dates before 2020:")
    for i, date in enumerate(dates):
        if date.year < 2020:
            print(f"  Index {i}: {df.index[i]} -> {date}")
    
    # Test prediction date generation
    print("\nTesting prediction date generation:")
    if len(df) >= 6:
        # Get daily totals
        daily_totals = df.sum(axis=1)
        
        # Test the date conversion logic from predictions
        X = (dates - dates[0]).days.values.reshape(-1, 1)
        print(f"X shape: {X.shape}")
        print(f"X first 5 values: {X[:5].flatten()}")
        print(f"X last 5 values: {X[-5:].flatten()}")
        
        # Check historical dates in predictions
        predictions = predict_future_costs(df, days_ahead=7, model_types=['survival'])
        
        if predictions and 'survival' in predictions:
            hist_dates = predictions['survival']['historical_dates']
            print(f"\nHistorical dates in predictions:")
            print(f"Type: {type(hist_dates)}")
            print(f"Length: {len(hist_dates)}")
            if len(hist_dates) > 0:
                print(f"First date: {hist_dates[0]}")
                print(f"Last date: {hist_dates[-1] if len(hist_dates) > 0 else 'N/A'}")
                
                # Check for 1970 dates
                for i, date in enumerate(hist_dates):
                    if hasattr(date, 'year') and date.year < 2020:
                        print(f"  Found 1970 date at index {i}: {date}")


if __name__ == "__main__":
    investigate_dates()