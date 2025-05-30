"""Prediction models for Claude Code Cost Analyzer."""

from datetime import timedelta
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def predict_future_costs(df: pd.DataFrame, days_ahead: int = 30, 
                        model_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Predict future costs using multiple regression models.
    
    Args:
        df: DataFrame with cost data
        days_ahead: Number of days to predict ahead
        model_types: List of model types to use
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary of prediction results
    """
    if len(df) < 6:  # Need at least 6 days of data
        return None
    
    # Default to all model types if not specified
    if model_types is None:
        model_types = ['survival', 'random']
    
    # Prepare data
    daily_totals = df.sum(axis=1)
    dates = pd.to_datetime(daily_totals.index)
    
    # Convert dates to numerical values (days since first date)
    X = (dates - dates[0]).days.values.reshape(-1, 1)
    y = daily_totals.values
    
    # Create future dates
    last_day = X[-1][0]
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    # Calculate physical limit based on max hourly rate and 8-hour constraint
    max_hourly_rate = np.max(y) / 8  # Maximum hourly rate observed
    daily_cap = max_hourly_rate * 8  # Physical limit per day
    
    results = {}
    
    # Survival analysis model
    if 'survival' in model_types:
        results['survival'] = _survival_analysis_model(
            X, y, future_days, future_dates, daily_cap, days_ahead, dates
        )
    
    # Random distribution model
    if 'random' in model_types:
        results['random'] = _random_distribution_model(
            y, future_dates, daily_cap, days_ahead, dates
        )
    
    return results


def _survival_analysis_model(X: np.ndarray, y: np.ndarray, 
                            future_days: np.ndarray, future_dates: List, 
                            daily_cap: float, days_ahead: int,
                            historical_dates: pd.DatetimeIndex) -> Dict[str, Any]:
    """Survival analysis model for cost prediction.
    
    Models cost patterns as "survival" of high-cost periods, 
    predicting decay toward median.
    """
    # Calculate the "hazard" of high costs
    percentiles = [50, 75, 90]
    threshold_values = np.percentile(y, percentiles)
    
    # Kaplan-Meier style analysis
    survival_data = {}
    
    for i, threshold in enumerate(threshold_values):
        # Count consecutive days above threshold
        above_threshold = y > threshold
        
        # Find runs of days above threshold
        runs = []
        current_run = 0
        for val in above_threshold:
            if val:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        
        # Calculate mean run length (expected duration above threshold)
        mean_duration = np.mean(runs) if runs else 1
        
        # Store survival data
        survival_data[f'p{percentiles[i]}'] = {
            'threshold': threshold,
            'mean_duration': mean_duration,
            'current_streak': current_run
        }
    
    # Current cost level and trend
    recent_costs = y[-7:] if len(y) >= 7 else y  # Last week
    current_level = y[-1]
    recent_trend = (recent_costs[-1] - recent_costs[0]) / len(recent_costs) if len(recent_costs) > 1 else 0
    
    # Make predictions based on survival model
    predictions_survival = []
    historical_fit_survival = []
    
    # For historical fit, use a smoothed version
    window = min(7, len(y) // 3)
    for i in range(len(y)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(y), i + window // 2 + 1)
        historical_fit_survival.append(np.mean(y[start_idx:end_idx]))
    
    # For future predictions, use exponential decay from current level
    # toward the median cost
    median_cost = np.median(y)
    
    for day_idx in range(days_ahead):
        # Exponential decay toward median
        decay_rate = 1 / survival_data['p75']['mean_duration']  # Use 75th percentile duration
        
        # Add some randomness based on historical volatility
        volatility = np.std(np.diff(y)) if len(y) > 1 else 0
        
        # Prediction with decay toward median
        days_ahead_current = day_idx + 1
        predicted = median_cost + (current_level - median_cost) * np.exp(-decay_rate * days_ahead_current)
        
        # Add trend component
        predicted += recent_trend * days_ahead_current * 0.5  # Damped trend
        
        # Ensure non-negative and apply daily cap
        predicted = max(0, min(predicted, daily_cap))
        
        predictions_survival.append(predicted)
    
    # Calculate R²
    ss_res = np.sum((y - historical_fit_survival) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Create description
    desc_parts = []
    for pct, data in survival_data.items():
        desc_parts.append(f"{pct}: ${data['threshold']:.0f} ({data['mean_duration']:.1f}d)")
    
    return {
        'name': 'Survival Analysis',
        'dates': future_dates,
        'predictions': np.array(predictions_survival),
        'historical_dates': historical_dates,
        'historical_fit': np.array(historical_fit_survival),
        'r_squared': r_squared,
        'equation': f'Decay to median ${median_cost:.2f} | Thresholds: {", ".join(desc_parts)}'
    }


def _random_distribution_model(y: np.ndarray, future_dates: List, 
                              daily_cap: float, days_ahead: int,
                              historical_dates: pd.DatetimeIndex) -> Dict[str, Any]:
    """Random distribution model for cost prediction.
    
    Treats daily costs as normally distributed random variables.
    """
    # Calculate statistics of the daily costs
    mean_cost = np.mean(y)
    std_cost = np.std(y)
    
    # Generate random predictions from normal distribution
    np.random.seed(42)  # For reproducibility
    predictions_random = []
    
    for _ in range(days_ahead):
        # Sample from normal distribution
        sample = np.random.normal(mean_cost, std_cost)
        
        # Apply constraints: non-negative and daily cap
        sample = max(0, min(sample, daily_cap))
        
        predictions_random.append(sample)
    
    # For historical fit, just use the actual mean for each day
    # This gives a flat line at the mean level
    historical_fit_random = np.full_like(y, mean_cost)
    
    # Apply daily cap to historical fit
    historical_fit_random = np.minimum(historical_fit_random, daily_cap)
    
    # Calculate R² (will be low since we're fitting with the mean)
    ss_res = np.sum((y - historical_fit_random) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'name': 'Random Distribution',
        'dates': future_dates,
        'predictions': np.array(predictions_random),
        'historical_dates': historical_dates,
        'historical_fit': historical_fit_random,
        'r_squared': r_squared,
        'equation': f'N(μ=${mean_cost:.2f}, σ=${std_cost:.2f}) | Cap=${daily_cap:.2f}/day'
    }