"""Tests for the visualizations module."""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from cost_analyzer.visualizations import (
    create_main_figure, create_project_breakdown_chart, create_treemap_chart,
    create_dow_analysis_chart, create_hourly_analysis_chart, create_moving_average_chart,
    create_token_usage_time_chart, create_token_breakdown_chart, create_token_efficiency_chart,
    create_token_by_project_chart, create_prediction_charts, create_additional_charts,
    create_token_charts
)


class TestVisualizations:
    """Test visualization functions."""
    
    @pytest.fixture
    def sample_cost_data(self):
        """Create sample cost data for testing."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D').strftime('%Y-%m-%d')
        data = {
            'project1': [10, 20, 15, 25, 30, 20, 15],
            'project2': [5, 10, 20, 15, 10, 25, 10]
        }
        df = pd.DataFrame(data, index=dates)
        projects = ['project1', 'project2']
        total_cost = df.sum().sum()
        return df, projects, total_cost
    
    @pytest.fixture
    def sample_token_data(self):
        """Create sample token data for testing."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D').strftime('%Y-%m-%d')
        data = {
            'project1_input': [100, 200, 150, 250, 300, 200, 150],
            'project1_output': [50, 100, 75, 125, 150, 100, 75],
            'project1_cache_creation': [10, 20, 15, 25, 30, 20, 15],
            'project1_cache_read': [5, 10, 7, 12, 15, 10, 7],
            'project1_total': [165, 330, 247, 412, 495, 330, 247],
            'project2_input': [80, 160, 120, 200, 240, 160, 120],
            'project2_output': [40, 80, 60, 100, 120, 80, 60],
            'project2_cache_creation': [8, 16, 12, 20, 24, 16, 12],
            'project2_cache_read': [4, 8, 6, 10, 12, 8, 6],
            'project2_total': [132, 264, 198, 330, 396, 264, 198]
        }
        df = pd.DataFrame(data, index=dates)
        projects = ['project1', 'project2']
        return df, projects
    
    @pytest.fixture
    def sample_stats(self):
        """Create sample statistics for testing."""
        return {
            'total_cost': 235.0,
            'avg_daily_cost': 33.57,
            'median_daily_cost': 30.0,
            'max_daily_cost': 55.0,
            'max_daily_date': '2024-01-05',
            'active_days': 7,
            'avg_hourly_cost': 4.20,
            'max_hourly_cost': 6.88,
            'top_project': 'project1',
            'top_project_cost': 135.0,
            'most_expensive_dow': 'Friday',
            'total_tokens': 3408,
            'avg_daily_tokens': 487,
            'max_daily_tokens': 891,
            'total_input_tokens': 2160,
            'total_output_tokens': 1080,
            'total_cache_creation_tokens': 216,
            'total_cache_read_tokens': 108,
            'cost_per_1k_tokens': 68.93
        }
    
    def test_create_main_figure(self, sample_cost_data):
        """Test main figure creation."""
        df, projects, total_cost = sample_cost_data
        fig = create_main_figure(df, projects, total_cost)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == f'Claude Code Usage Cost Analysis (Total: ${total_cost:.2f})'
        
        # Check subplots
        assert 'Daily Claude Code Usage Cost by Project' in str(fig)
        assert 'Cumulative Claude Code Usage Cost' in str(fig)
    
    def test_create_project_breakdown_chart(self, sample_cost_data):
        """Test project breakdown chart creation."""
        df, projects, _ = sample_cost_data
        fig = create_project_breakdown_chart(df, projects)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'bar'
        assert fig.data[0].orientation == 'h'
        assert fig.layout.title.text == "Project Cost Breakdown"
    
    def test_create_treemap_chart(self, sample_cost_data):
        """Test treemap chart creation."""
        df, projects, _ = sample_cost_data
        fig = create_treemap_chart(df, projects)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Project Cost Hierarchy'
        
        # Test with single project (should return None)
        df_single = pd.DataFrame({'project1': [10, 20]}, index=['2024-01-01', '2024-01-02'])
        fig_none = create_treemap_chart(df_single, ['project1'])
        assert fig_none is None
    
    def test_create_dow_analysis_chart(self, sample_cost_data):
        """Test day of week analysis chart."""
        df, _, _ = sample_cost_data
        fig = create_dow_analysis_chart(df)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'bar'
        assert fig.layout.title.text == "Total Costs by Day of Week"
    
    def test_create_hourly_analysis_chart(self, sample_cost_data):
        """Test hourly analysis chart."""
        df, _, _ = sample_cost_data
        fig = create_hourly_analysis_chart(df)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Box plot plus lines
        assert fig.data[0].type == 'box'
        assert fig.layout.title.text == "Hourly Cost Analysis (8-hour workday)"
    
    def test_create_moving_average_chart(self, sample_cost_data):
        """Test moving average chart."""
        df, _, _ = sample_cost_data
        fig = create_moving_average_chart(df)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Daily costs and 7-day MA
        assert any('Daily Cost' in trace.name for trace in fig.data if trace.name)
        assert any('7-Day MA' in trace.name for trace in fig.data if trace.name)
        
        # Test with enough data for 30-day MA
        dates = pd.date_range('2024-01-01', periods=35, freq='D').strftime('%Y-%m-%d')
        df_long = pd.DataFrame({'project1': np.random.rand(35) * 50}, index=dates)
        fig_long = create_moving_average_chart(df_long)
        assert any('30-Day MA' in trace.name for trace in fig_long.data if trace.name)
    
    def test_create_token_usage_time_chart(self, sample_token_data):
        """Test token usage time series chart."""
        df, projects = sample_token_data
        fig = create_token_usage_time_chart(df, projects)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # Input, Output, Cache Creation, Cache Read
        assert fig.layout.title.text == "Daily Token Usage by Type"
        
        # Check token types
        trace_names = [trace.name for trace in fig.data]
        assert 'Input Tokens' in trace_names
        assert 'Output Tokens' in trace_names
        assert 'Cache Creation Tokens' in trace_names
        assert 'Cache Read Tokens' in trace_names
    
    def test_create_token_breakdown_chart(self, sample_stats):
        """Test token breakdown pie chart."""
        fig = create_token_breakdown_chart(sample_stats)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'pie'
        assert fig.layout.title.text == "Token Usage Breakdown"
        
        # Test with no token data
        fig_none = create_token_breakdown_chart({})
        assert fig_none is None
        
        # Test with zero tokens
        stats_zero = {'total_tokens': 0}
        fig_zero = create_token_breakdown_chart(stats_zero)
        assert fig_zero is None
    
    def test_create_token_efficiency_chart(self, sample_token_data, sample_stats):
        """Test token efficiency chart."""
        df, projects = sample_token_data
        fig = create_token_efficiency_chart(df, projects, sample_stats)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert 'Token Usage Trend' in fig.layout.title.text
        
        # Test with no cost data
        stats_no_cost = {'total_cost': 0}
        fig_none = create_token_efficiency_chart(df, projects, stats_no_cost)
        assert fig_none is None
    
    def test_create_token_by_project_chart(self, sample_token_data):
        """Test token by project chart."""
        df, projects = sample_token_data
        fig = create_token_by_project_chart(df, projects)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'bar'
        assert fig.layout.title.text == "Token Usage by Project"
        
        # Test with no token data
        df_empty = pd.DataFrame(index=['2024-01-01'])
        fig_none = create_token_by_project_chart(df_empty, [])
        assert fig_none is None
    
    def test_create_prediction_charts(self, sample_cost_data):
        """Test prediction charts creation."""
        df, _, _ = sample_cost_data
        
        predictions = {
            'survival': {
                'name': 'Survival Analysis',
                'dates': pd.date_range('2024-01-08', periods=7, freq='D'),
                'predictions': np.array([25, 23, 21, 20, 19, 18, 17]),
                'historical_dates': pd.to_datetime(df.index),
                'historical_fit': np.array([12, 22, 25, 28, 30, 32, 20]),
                'r_squared': 0.85,
                'equation': 'Decay to median $20.00'
            }
        }
        
        charts = create_prediction_charts(predictions, df)
        
        assert 'prediction_survival' in charts
        assert 'prediction_comparison' in charts
        
        assert isinstance(charts['prediction_survival'], go.Figure)
        assert isinstance(charts['prediction_comparison'], go.Figure)
        
        # Check that historical and predicted data are in the charts
        survival_fig = charts['prediction_survival']
        assert any('Actual Costs' in trace.name for trace in survival_fig.data if trace.name)
        assert any('Predicted Costs' in trace.name for trace in survival_fig.data if trace.name)
    
    def test_create_additional_charts(self, sample_cost_data, sample_stats):
        """Test additional charts creation."""
        df, projects, _ = sample_cost_data
        
        # Without predictions
        charts = create_additional_charts(df, projects, sample_stats, None)
        
        assert 'project_breakdown' in charts
        assert 'treemap' in charts  # Should exist with 2 projects
        assert 'dow' in charts
        assert 'hourly_analysis' in charts
        assert 'moving_avg' in charts
        
        # Verify all are figures
        for chart in charts.values():
            assert isinstance(chart, go.Figure)
    
    def test_create_token_charts(self, sample_token_data, sample_stats):
        """Test token charts creation."""
        df, projects = sample_token_data
        
        charts = create_token_charts(df, projects, sample_stats)
        
        assert 'token_usage_time' in charts
        assert 'token_breakdown' in charts
        assert 'token_efficiency' in charts
        assert 'token_by_project' in charts
        
        # Verify all are figures
        for chart in charts.values():
            assert isinstance(chart, go.Figure)