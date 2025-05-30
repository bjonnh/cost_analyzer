"""Tests for the predictions module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cost_analyzer.predictions import (
    predict_future_costs, _survival_analysis_model, _random_distribution_model
)


class TestPredictions:
    """Test prediction models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample cost data for testing."""
        # Create 30 days of sample data with some pattern
        dates = pd.date_range('2024-01-01', periods=30, freq='D').strftime('%Y-%m-%d')
        
        # Create cost data with trend and noise
        base_cost = 10.0
        trend = 0.5
        costs = []
        
        for i in range(30):
            daily_cost = base_cost + (trend * i) + np.random.normal(0, 2)
            costs.append(max(0, daily_cost))  # Ensure non-negative
        
        data = {
            'project1': costs[:30],
            'project2': [c * 0.5 for c in costs[:30]]
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_predict_future_costs_insufficient_data(self):
        """Test predictions with insufficient data."""
        # Less than 6 days of data
        dates = pd.date_range('2024-01-01', periods=5, freq='D').strftime('%Y-%m-%d')
        df = pd.DataFrame({'project1': [10, 20, 15, 25, 30]}, index=dates)
        
        result = predict_future_costs(df)
        assert result is None
    
    def test_predict_future_costs_all_models(self, sample_data):
        """Test predictions with all models."""
        result = predict_future_costs(sample_data, days_ahead=7)
        
        assert result is not None
        assert 'survival' in result
        assert 'random' in result
        
        # Check survival model
        survival = result['survival']
        assert survival['name'] == 'Survival Analysis'
        assert len(survival['predictions']) == 7
        assert len(survival['historical_fit']) == len(sample_data)
        assert 'r_squared' in survival
        assert 'equation' in survival
        assert all(p >= 0 for p in survival['predictions'])  # All predictions non-negative
        
        # Check random model
        random = result['random']
        assert random['name'] == 'Random Distribution'
        assert len(random['predictions']) == 7
        assert len(random['historical_fit']) == len(sample_data)
        assert 'r_squared' in random
        assert 'equation' in random
        assert all(p >= 0 for p in random['predictions'])  # All predictions non-negative
    
    def test_predict_future_costs_selected_models(self, sample_data):
        """Test predictions with selected models only."""
        # Test with only survival model
        result = predict_future_costs(sample_data, days_ahead=10, model_types=['survival'])
        assert 'survival' in result
        assert 'random' not in result
        
        # Test with only random model
        result = predict_future_costs(sample_data, days_ahead=10, model_types=['random'])
        assert 'random' in result
        assert 'survival' not in result
    
    def test_survival_analysis_model(self, sample_data):
        """Test survival analysis model directly."""
        daily_totals = sample_data.sum(axis=1)
        dates = pd.to_datetime(daily_totals.index)
        X = (dates - dates[0]).days.values.reshape(-1, 1)
        y = daily_totals.values
        
        last_day = X[-1][0]
        future_days = np.arange(last_day + 1, last_day + 8).reshape(-1, 1)
        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 8)]
        daily_cap = np.max(y) * 2  # High cap for testing
        
        result = _survival_analysis_model(X, y, future_days, future_dates, daily_cap, 7)
        
        assert result['name'] == 'Survival Analysis'
        assert len(result['predictions']) == 7
        assert len(result['historical_fit']) == len(y)
        assert result['r_squared'] >= -1 and result['r_squared'] <= 1
        
        # Check that predictions decay toward median
        median_cost = np.median(y)
        last_prediction = result['predictions'][-1]
        first_prediction = result['predictions'][0]
        
        # Later predictions should be closer to median than earlier ones
        assert abs(last_prediction - median_cost) <= abs(first_prediction - median_cost)
    
    def test_random_distribution_model(self, sample_data):
        """Test random distribution model directly."""
        daily_totals = sample_data.sum(axis=1)
        y = daily_totals.values
        
        dates = pd.to_datetime(daily_totals.index)
        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 8)]
        daily_cap = np.max(y) * 2  # High cap for testing
        
        result = _random_distribution_model(y, future_dates, daily_cap, 7)
        
        assert result['name'] == 'Random Distribution'
        assert len(result['predictions']) == 7
        assert len(result['historical_fit']) == len(y)
        assert result['r_squared'] >= -1 and result['r_squared'] <= 1
        
        # Check that predictions are within reasonable range
        mean_cost = np.mean(y)
        std_cost = np.std(y)
        
        for pred in result['predictions']:
            assert pred >= 0  # Non-negative
            assert pred <= daily_cap  # Below cap
            # Most predictions should be within 3 standard deviations
            assert abs(pred - mean_cost) <= 4 * std_cost
        
        # Check that historical fit is constant (mean)
        assert all(h == result['historical_fit'][0] for h in result['historical_fit'])
    
    def test_physical_limit_constraint(self):
        """Test that predictions respect physical limits."""
        # Create data with one very high day
        dates = pd.date_range('2024-01-01', periods=10, freq='D').strftime('%Y-%m-%d')
        costs = [10, 12, 11, 100, 13, 12, 14, 11, 13, 12]  # One outlier
        df = pd.DataFrame({'project1': costs}, index=dates)
        
        result = predict_future_costs(df, days_ahead=5)
        
        # Physical limit should be based on max daily cost / 8 * 8
        max_daily = 100
        daily_cap = max_daily  # Since it's already the max
        
        # Check all predictions are within cap
        for model_key, model_data in result.items():
            for pred in model_data['predictions']:
                assert pred <= daily_cap
    
    def test_prediction_dates(self, sample_data):
        """Test that prediction dates are correct."""
        result = predict_future_costs(sample_data, days_ahead=14)
        
        last_date = pd.to_datetime(sample_data.index[-1])
        
        for model_key, model_data in result.items():
            pred_dates = model_data['dates']
            assert len(pred_dates) == 14
            
            # Check dates are consecutive
            for i, date in enumerate(pred_dates):
                expected_date = last_date + timedelta(days=i + 1)
                assert date.date() == expected_date.date()
    
    def test_empty_cost_periods(self):
        """Test predictions with periods of zero cost."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D').strftime('%Y-%m-%d')
        # Pattern: active, inactive, active
        costs = [10, 15, 20, 0, 0, 0, 0, 25, 30, 20, 15, 10, 0, 0, 5, 10, 15, 20, 25, 30]
        df = pd.DataFrame({'project1': costs}, index=dates)
        
        result = predict_future_costs(df, days_ahead=7)
        
        assert result is not None
        # Should still produce valid predictions
        for model_key, model_data in result.items():
            assert len(model_data['predictions']) == 7
            assert all(p >= 0 for p in model_data['predictions'])