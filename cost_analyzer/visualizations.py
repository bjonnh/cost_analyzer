"""Visualization module for Claude Code Cost Analyzer."""

from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_main_figure(df: pd.DataFrame, projects: List[str], total_cost: float) -> go.Figure:
    """Create the main Plotly figure with two subplots.
    
    Args:
        df: DataFrame with cost data
        projects: List of project names
        total_cost: Total cost across all projects
        
    Returns:
        go.Figure: Main dashboard figure
    """
    # Create a single figure with two subplots sharing the x-axis
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Daily Claude Code Usage Cost by Project', 'Cumulative Claude Code Usage Cost')
    )
    
    # 1. Daily stacked area chart (top subplot)
    for project in projects:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[project],
                mode='lines',
                name=project,
                stackgroup='one',
                hoverinfo='x+y+name'
            ),
            row=1, col=1
        )
    
    # Add total cost line to daily chart
    df['Total'] = df.sum(axis=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Total'],
            mode='lines',
            name='Total Cost',
            line=dict(width=2, color='black', dash='dash'),
        ),
        row=1, col=1
    )
    
    # 2. Cumulative cost chart (bottom subplot)
    df['Cumulative'] = df['Total'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Cumulative'],
            mode='lines',
            name='Cumulative Cost',
            line=dict(width=3, color='red'),
            fill='tozeroy',
        ),
        row=2, col=1
    )
    
    # Add daily cost bars to the cumulative chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Total'],
            name='Daily Cost',
            marker_color='rgba(55, 83, 109, 0.7)',
            opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # Update layout with dark theme
    fig.update_layout(
        title=f'Claude Code Usage Cost Analysis (Total: ${total_cost:.2f})',
        hovermode='x unified',
        legend_title='Projects/Costs',
        height=900,
        margin=dict(l=50, r=50, t=100, b=100),
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="Daily Cost (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Cost (USD)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Add physical limit line to daily chart
    max_hourly_rate = df.sum(axis=1).max() / 8
    daily_cap = max_hourly_rate * 8
    
    fig.add_hline(
        y=daily_cap, 
        line_dash="dot", 
        line_color="red",
        annotation_text=f"Physical Limit: ${daily_cap:.2f}/day",
        annotation_position="top right",
        row=1, col=1
    )
    
    return fig


def create_project_breakdown_chart(df: pd.DataFrame, projects: List[str]) -> go.Figure:
    """Create project cost breakdown horizontal bar chart.
    
    Args:
        df: DataFrame with cost data
        projects: List of project names
        
    Returns:
        go.Figure: Project breakdown chart
    """
    project_totals = df[projects].sum().sort_values(ascending=True)
    
    # Calculate percentages
    total_sum = project_totals.sum()
    percentages = (project_totals / total_sum * 100).round(1)
    
    # Create horizontal bar chart
    bar_fig = go.Figure()
    
    bar_fig.add_trace(go.Bar(
        y=project_totals.index,
        x=project_totals.values,
        orientation='h',
        text=[f'${val:.2f} ({pct}%)' for val, pct in zip(project_totals.values, percentages)],
        textposition='auto',
        marker=dict(
            color=project_totals.values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cost (USD)")
        )
    ))
    
    bar_fig.update_layout(
        title="Project Cost Breakdown",
        xaxis_title="Total Cost (USD)",
        yaxis_title="Project",
        height=max(400, len(projects) * 40),  # Dynamic height based on number of projects
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        showlegend=False
    )
    
    return bar_fig


def create_treemap_chart(df: pd.DataFrame, projects: List[str]) -> Optional[go.Figure]:
    """Create treemap visualization for project costs.
    
    Args:
        df: DataFrame with cost data
        projects: List of project names
        
    Returns:
        Optional[go.Figure]: Treemap chart or None if only one project
    """
    if len(projects) <= 1:
        return None
    
    project_totals = df[projects].sum().sort_values(ascending=False)
    
    treemap_data = []
    for project in project_totals.index:
        cost = project_totals[project]
        treemap_data.append({
            'name': project,
            'value': cost,
            'label': f'{project}<br>${cost:.2f}'
        })
    
    treemap_fig = px.treemap(
        treemap_data,
        path=['name'],
        values='value',
        title='Project Cost Hierarchy',
        custom_data=['label']
    )
    
    treemap_fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate='<b>%{customdata[0]}</b><br>%{percentParent}<extra></extra>'
    )
    
    treemap_fig.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    
    return treemap_fig


def create_dow_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Create day of week analysis chart.
    
    Args:
        df: DataFrame with cost data
        
    Returns:
        go.Figure: Day of week analysis chart
    """
    df_dow = df.copy()
    df_dow['dow'] = pd.to_datetime(df_dow.index).day_name()
    df_dow['dow_num'] = pd.to_datetime(df_dow.index).dayofweek
    dow_totals = df_dow.groupby(['dow', 'dow_num']).sum().sum(axis=1).reset_index()
    dow_totals.columns = ['dow', 'dow_num', 'cost']
    dow_totals = dow_totals.sort_values('dow_num')
    
    dow_fig = px.bar(
        dow_totals,
        x='dow',
        y='cost',
        title="Total Costs by Day of Week",
        labels={'cost': 'Total Cost (USD)', 'dow': 'Day of Week'}
    )
    dow_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    
    return dow_fig


def create_hourly_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Create hourly cost analysis chart.
    
    Args:
        df: DataFrame with cost data
        
    Returns:
        go.Figure: Hourly analysis chart
    """
    hourly_fig = go.Figure()
    
    # Calculate hourly costs for each day (assuming 8-hour workday)
    daily_totals = df.sum(axis=1)
    hourly_costs = daily_totals / 8
    
    # Create box plot for hourly costs
    hourly_fig.add_trace(go.Box(
        y=hourly_costs.values,
        name='Hourly Cost Distribution',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color='lightblue'),
        line=dict(color='blue')
    ))
    
    # Add average and max lines
    avg_hourly = hourly_costs.mean()
    max_hourly = hourly_costs.max()
    
    hourly_fig.add_hline(
        y=avg_hourly, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"Average: ${avg_hourly:.2f}/hr"
    )
    
    hourly_fig.add_hline(
        y=max_hourly, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Physical Limit: ${max_hourly:.2f}/hr"
    )
    
    hourly_fig.update_layout(
        title="Hourly Cost Analysis (8-hour workday)",
        yaxis_title="Cost per Hour (USD)",
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=400
    )
    
    return hourly_fig


def create_moving_average_chart(df: pd.DataFrame) -> go.Figure:
    """Create moving average chart.
    
    Args:
        df: DataFrame with cost data
        
    Returns:
        go.Figure: Moving average chart
    """
    ma_fig = go.Figure()
    daily_totals = df.sum(axis=1)
    
    ma_fig.add_trace(go.Scatter(
        x=daily_totals.index,
        y=daily_totals.values,
        mode='lines',
        name='Daily Cost',
        line=dict(color='lightblue')
    ))
    
    # Add 7-day moving average
    ma_7 = daily_totals.rolling(window=7, min_periods=1).mean()
    ma_fig.add_trace(go.Scatter(
        x=ma_7.index,
        y=ma_7.values,
        mode='lines',
        name='7-Day MA',
        line=dict(color='blue', width=2)
    ))
    
    # Add 30-day moving average if enough data
    if len(daily_totals) >= 30:
        ma_30 = daily_totals.rolling(window=30, min_periods=1).mean()
        ma_fig.add_trace(go.Scatter(
            x=ma_30.index,
            y=ma_30.values,
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=2)
        ))
    
    ma_fig.update_layout(
        title="Cost Trends with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Cost (USD)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff')
    )
    
    return ma_fig


def create_token_usage_time_chart(token_df: pd.DataFrame, projects: List[str]) -> go.Figure:
    """Create token usage over time chart.
    
    Args:
        token_df: DataFrame with token data
        projects: List of project names
        
    Returns:
        go.Figure: Token usage time series chart
    """
    token_fig = go.Figure()
    
    # Get daily totals for each token type
    dates = token_df.index
    
    # Calculate totals for each type
    input_total = pd.Series(0, index=dates)
    output_total = pd.Series(0, index=dates) 
    cache_creation_total = pd.Series(0, index=dates)
    cache_read_total = pd.Series(0, index=dates)
    
    for project in projects:
        if f"{project}_input" in token_df.columns:
            input_total += token_df[f"{project}_input"]
        if f"{project}_output" in token_df.columns:
            output_total += token_df[f"{project}_output"]
        if f"{project}_cache_creation" in token_df.columns:
            cache_creation_total += token_df[f"{project}_cache_creation"]
        if f"{project}_cache_read" in token_df.columns:
            cache_read_total += token_df[f"{project}_cache_read"]
    
    # Add traces for each token type
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=input_total,
        mode='lines',
        name='Input Tokens',
        stackgroup='one',
        fillcolor='rgba(65, 105, 225, 0.6)'
    ))
    
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=output_total,
        mode='lines',
        name='Output Tokens',
        stackgroup='one',
        fillcolor='rgba(50, 205, 50, 0.6)'
    ))
    
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=cache_creation_total,
        mode='lines',
        name='Cache Creation Tokens',
        stackgroup='one',
        fillcolor='rgba(255, 165, 0, 0.6)'
    ))
    
    token_fig.add_trace(go.Scatter(
        x=dates,
        y=cache_read_total,
        mode='lines',
        name='Cache Read Tokens',
        stackgroup='one',
        fillcolor='rgba(138, 43, 226, 0.6)'
    ))
    
    token_fig.update_layout(
        title="Daily Token Usage by Type",
        xaxis_title="Date",
        yaxis_title="Number of Tokens",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=600
    )
    
    return token_fig


def create_token_breakdown_chart(stats: Dict[str, Any]) -> Optional[go.Figure]:
    """Create token type breakdown horizontal bar chart.
    
    Args:
        stats: Dictionary of statistics
        
    Returns:
        Optional[go.Figure]: Token breakdown chart or None if no data
    """
    if not stats or 'total_tokens' not in stats or stats['total_tokens'] == 0:
        return None
    
    labels = []
    values = []
    colors = []
    percentages = []
    
    total = stats['total_tokens']
    
    if 'total_input_tokens' in stats and stats['total_input_tokens'] > 0:
        labels.append('Input Tokens')
        val = stats['total_input_tokens']
        values.append(val)
        colors.append('#4169E1')
        percentages.append(f"{val/total*100:.1f}%")
    
    if 'total_output_tokens' in stats and stats['total_output_tokens'] > 0:
        labels.append('Output Tokens')
        val = stats['total_output_tokens']
        values.append(val)
        colors.append('#32CD32')
        percentages.append(f"{val/total*100:.1f}%")
        
    if 'total_cache_creation_tokens' in stats and stats['total_cache_creation_tokens'] > 0:
        labels.append('Cache Creation Tokens')
        val = stats['total_cache_creation_tokens']
        values.append(val)
        colors.append('#FFA500')
        percentages.append(f"{val/total*100:.1f}%")
        
    if 'total_cache_read_tokens' in stats and stats['total_cache_read_tokens'] > 0:
        labels.append('Cache Read Tokens')
        val = stats['total_cache_read_tokens']
        values.append(val)
        colors.append('#8A2BE2')
        percentages.append(f"{val/total*100:.1f}%")
    
    if not labels:
        return None
    
    # Create horizontal bar chart
    bar_fig = go.Figure()
    
    # Reverse order so largest is on top
    labels_rev = labels[::-1]
    values_rev = values[::-1]
    colors_rev = colors[::-1]
    percentages_rev = percentages[::-1]
    
    bar_fig.add_trace(go.Bar(
        y=labels_rev,
        x=values_rev,
        orientation='h',
        text=[f'{v/1e6:.1f}M ({p})' for v, p in zip(values_rev, percentages_rev)],
        textposition='auto',
        marker=dict(color=colors_rev),
        hovertemplate='%{y}: %{x:,.0f} tokens<br>%{text}<extra></extra>'
    ))
    
    bar_fig.update_layout(
        title=f"Token Usage Breakdown (Total: {total:,})",
        xaxis_title="Number of Tokens",
        yaxis_title="Token Type",
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=400,
        showlegend=False,
        xaxis=dict(tickformat=',')
    )
    
    return bar_fig


def create_token_efficiency_chart(token_df: pd.DataFrame, projects: List[str], stats: Dict[str, Any]) -> Optional[go.Figure]:
    """Create token efficiency chart.
    
    Args:
        token_df: DataFrame with token data
        projects: List of project names
        stats: Dictionary of statistics
        
    Returns:
        Optional[go.Figure]: Token efficiency chart or None if no data
    """
    if not stats or stats.get('total_cost', 0) == 0:
        return None
    
    efficiency_fig = go.Figure()
    
    # Calculate daily token totals
    dates = token_df.index
    daily_tokens = pd.Series(0, index=dates)
    
    for project in projects:
        if f"{project}_total" in token_df.columns:
            daily_tokens += token_df[f"{project}_total"]
    
    # Calculate tokens per dollar
    cost_per_token = stats['total_cost'] / stats.get('total_tokens', 1) if stats.get('total_tokens', 0) > 0 else 0
    tokens_per_dollar = 1 / cost_per_token if cost_per_token > 0 else 0
    
    efficiency_fig.add_trace(go.Scatter(
        x=dates,
        y=daily_tokens / 1000,  # Show in K tokens
        mode='lines+markers',
        name='Daily Tokens (K)',
        line=dict(color='cyan', width=2)
    ))
    
    efficiency_fig.update_layout(
        title=f"Token Usage Trend (Avg: {tokens_per_dollar:.0f} tokens/$)",
        xaxis_title="Date",
        yaxis_title="Tokens (thousands)",
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=400
    )
    
    return efficiency_fig


def create_token_by_project_chart(token_df: pd.DataFrame, projects: List[str]) -> Optional[go.Figure]:
    """Create token usage by project chart with token type breakdown.
    
    Args:
        token_df: DataFrame with token data
        projects: List of project names
        
    Returns:
        Optional[go.Figure]: Token by project chart or None if no data
    """
    # Collect token data by project and type
    project_data = {}
    for project in projects:
        tokens_by_type = {
            'input': 0,
            'output': 0,
            'cache_creation': 0,
            'cache_read': 0
        }
        
        for token_type in tokens_by_type.keys():
            col_name = f"{project}_{token_type}"
            if col_name in token_df.columns:
                tokens_by_type[token_type] = token_df[col_name].sum()
        
        total = sum(tokens_by_type.values())
        if total > 0:
            project_data[project] = {
                'total': total,
                'breakdown': tokens_by_type
            }
    
    if not project_data:
        return None
    
    # Sort projects by total tokens
    sorted_projects = sorted(project_data.keys(), key=lambda x: project_data[x]['total'], reverse=True)
    
    # Create stacked bar chart
    bar_fig = go.Figure()
    
    # Define colors for each token type
    colors = {
        'input': '#4169E1',
        'output': '#32CD32',
        'cache_creation': '#FFA500',
        'cache_read': '#8A2BE2'
    }
    
    # Add traces for each token type
    for token_type, color in colors.items():
        values = []
        for project in sorted_projects:
            values.append(project_data[project]['breakdown'][token_type] / 1e6)  # Show in millions
        
        bar_fig.add_trace(go.Bar(
            name=token_type.replace('_', ' ').title(),
            x=sorted_projects,
            y=values,
            marker_color=color,
            hovertemplate='%{x}<br>%{fullData.name}: %{y:.1f}M tokens<extra></extra>'
        ))
    
    # Update layout for stacked bars
    bar_fig.update_layout(
        title="Token Usage by Project (Breakdown by Type)",
        xaxis_title="Project",
        yaxis_title="Tokens (millions)",
        barmode='stack',
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(tickformat=',')
    )
    
    # Add total values on top of bars
    for i, project in enumerate(sorted_projects):
        total = project_data[project]['total'] / 1e6
        bar_fig.add_annotation(
            x=project,
            y=total,
            text=f'{total:.1f}M',
            showarrow=False,
            yshift=10,
            font=dict(color='white', size=12)
        )
    
    return bar_fig


def create_prediction_charts(predictions: Dict[str, Any], df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create prediction charts for each model and comparison.
    
    Args:
        predictions: Dictionary of prediction results
        df: DataFrame with cost data
        
    Returns:
        Dict[str, go.Figure]: Dictionary of prediction charts
    """
    charts = {}
    daily_totals = df.sum(axis=1)
    
    # Create individual prediction charts
    for model_key, model_data in predictions.items():
        pred_fig = go.Figure()
        
        # Historical data
        pred_fig.add_trace(go.Scatter(
            x=pd.to_datetime(daily_totals.index),
            y=daily_totals.values,
            mode='markers',
            name='Actual Costs',
            marker=dict(size=8, color='lightblue')
        ))
        
        # Historical fit
        pred_fig.add_trace(go.Scatter(
            x=model_data['historical_dates'],
            y=model_data['historical_fit'],
            mode='lines',
            name=f'{model_data["name"]} Fit',
            line=dict(color='green', dash='dash', width=2)
        ))
        
        # Future predictions
        pred_fig.add_trace(go.Scatter(
            x=model_data['dates'],
            y=model_data['predictions'],
            mode='lines+markers',
            name='Predicted Costs',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
        
        pred_fig.update_layout(
            title=f"{model_data['name']} Model (R² = {model_data['r_squared']:.3f})<br><sub>{model_data['equation']}</sub>",
            xaxis_title="Date",
            yaxis_title="Daily Cost (USD)",
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font=dict(color='#fff')
        )
        
        charts[f'prediction_{model_key}'] = pred_fig
    
    # Create comparison chart
    comp_fig = go.Figure()
    
    # Historical data
    comp_fig.add_trace(go.Scatter(
        x=pd.to_datetime(daily_totals.index),
        y=daily_totals.values,
        mode='markers',
        name='Actual Costs',
        marker=dict(size=8, color='white'),
        showlegend=True
    ))
    
    # Add all model fits and predictions
    colors = {
        'survival': 'green',
        'random': 'orange'
    }
    
    for model_key, model_data in predictions.items():
        color = colors.get(model_key, 'gray')
        
        # Historical fit
        comp_fig.add_trace(go.Scatter(
            x=model_data['historical_dates'],
            y=model_data['historical_fit'],
            mode='lines',
            name=f'{model_data["name"]} (R²={model_data["r_squared"]:.2f})',
            line=dict(color=color, dash='dash', width=2),
            legendgroup=model_key
        ))
        
        # Future predictions
        comp_fig.add_trace(go.Scatter(
            x=model_data['dates'],
            y=model_data['predictions'],
            mode='lines',
            name=f'{model_data["name"]} Prediction',
            line=dict(color=color, width=3),
            legendgroup=model_key,
            showlegend=False
        ))
    
    comp_fig.update_layout(
        title="Model Comparison - All Prediction Models",
        xaxis_title="Date",
        yaxis_title="Daily Cost (USD)",
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#fff'),
        height=600
    )
    
    charts['prediction_comparison'] = comp_fig
    
    return charts


def create_additional_charts(df: pd.DataFrame, projects: List[str], 
                           stats: Dict[str, Any], 
                           predictions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Create all additional analysis charts.
    
    Args:
        df: DataFrame with cost data
        projects: List of project names
        stats: Dictionary of statistics
        predictions: Optional dictionary of prediction results
        
    Returns:
        Dict[str, Any]: Dictionary of all charts
    """
    charts = {}
    
    # Project breakdown
    charts['project_breakdown'] = create_project_breakdown_chart(df, projects)
    
    # Treemap
    treemap = create_treemap_chart(df, projects)
    if treemap:
        charts['treemap'] = treemap
    
    # Day of week analysis
    charts['dow'] = create_dow_analysis_chart(df)
    
    # Hourly analysis
    charts['hourly_analysis'] = create_hourly_analysis_chart(df)
    
    # Moving averages
    charts['moving_avg'] = create_moving_average_chart(df)
    
    # Prediction charts
    if predictions and isinstance(predictions, dict):
        prediction_charts = create_prediction_charts(predictions, df)
        charts.update(prediction_charts)
    
    return charts


def create_token_charts(token_df: pd.DataFrame, projects: List[str], 
                       stats: Dict[str, Any]) -> Dict[str, Any]:
    """Create all token analysis charts.
    
    Args:
        token_df: DataFrame with token data
        projects: List of project names
        stats: Dictionary of statistics
        
    Returns:
        Dict[str, Any]: Dictionary of token charts
    """
    charts = {}
    
    # Token usage over time
    charts['token_usage_time'] = create_token_usage_time_chart(token_df, projects)
    
    # Token breakdown
    breakdown = create_token_breakdown_chart(stats)
    if breakdown:
        charts['token_breakdown'] = breakdown
    
    # Token efficiency
    efficiency = create_token_efficiency_chart(token_df, projects, stats)
    if efficiency:
        charts['token_efficiency'] = efficiency
    
    # Token by project
    by_project = create_token_by_project_chart(token_df, projects)
    if by_project:
        charts['token_by_project'] = by_project
    
    return charts