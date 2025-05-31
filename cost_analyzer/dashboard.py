"""Dashboard UI module for Claude Code Cost Analyzer."""

from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any

import dash
from dash import dcc, html, Input, Output, State
import json
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from .data_processor import calculate_statistics, get_window_analysis
from .predictions import predict_future_costs
from .visualizations import (
    create_main_figure, create_additional_charts, create_token_charts,
    create_windows_timeline_chart, create_window_stats_chart, create_current_window_gauge
)


# Custom CSS for better styling
INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0a0a0a;
            }
            .card {
                background-color: #1a1a1a;
                border: 1px solid #333;
            }
            .stats-card {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                border: none;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                transition: transform 0.2s;
            }
            .stats-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            }
            .prediction-alert {
                background-color: #2d1b69;
                border: 1px solid #6c4ba8;
            }
            .DateRangePickerInput {
                background-color: #1a1a1a !important;
                border: 1px solid #444 !important;
            }
            .DateRangePickerInput__withBorder {
                border-color: #444 !important;
            }
            .DateInput_input {
                background-color: #1a1a1a !important;
                color: #fff !important;
            }
            .DateRangePickerInput_arrow {
                background-color: #1a1a1a !important;
            }
            .btn-group-sm > .btn {
                padding: 0.25rem 0.5rem;
                font-size: 0.875rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def create_layout() -> dbc.Container:
    """Create the main dashboard layout.
    
    Returns:
        dbc.Container: Dashboard layout
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Claude Code Cost Analytics", 
                       className="text-center mb-4 mt-3",
                       style={'color': '#fff', 'fontWeight': 'bold'}),
                html.Hr(style={'borderColor': '#444'}),
            ])
        ]),
        
        # Control panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("🔄 Refresh Data", 
                                          id='refresh-button', 
                                          color="primary", 
                                          size="lg",
                                          n_clicks=0),
                            ], width=2),
                            dbc.Col([
                                dcc.DatePickerRange(
                                    id='date-range-picker',
                                    display_format='YYYY-MM-DD',
                                    style={'width': '100%'},
                                    className="dash-bootstrap"
                                )
                            ], width=5),
                            dbc.Col([
                                html.Div(id='last-update', className="text-center mt-2")
                            ], width=3),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("1W", id="range-1w", color="secondary", size="sm"),
                                    dbc.Button("1M", id="range-1m", color="secondary", size="sm"),
                                    dbc.Button("3M", id="range-3m", color="secondary", size="sm"),
                                    dbc.Button("All", id="range-all", color="secondary", size="sm"),
                                ], size="sm")
                            ], width=2),
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Date range info
        dbc.Row([
            dbc.Col([
                dbc.Alert(id="date-range-info", color="info", className="text-center mb-3", is_open=False)
            ])
        ]),
        
        # Stats cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Cost", className="card-title"),
                        html.H2(id="total-cost-stat", className="text-primary"),
                    ])
                ], className="stats-card text-white")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Daily Average", className="card-title"),
                        html.H2(id="daily-avg-stat", className="text-info"),
                    ])
                ], className="stats-card text-white")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Top Project", className="card-title"),
                        html.H3(id="top-project-stat", className="text-warning"),
                    ])
                ], className="stats-card text-white")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Active Days", className="card-title"),
                        html.H2(id="active-days-stat", className="text-success"),
                    ])
                ], className="stats-card text-white")
            ], width=3),
        ], className="mb-4"),
        
        # Prediction alert
        dbc.Row([
            dbc.Col([
                dbc.Alert(
                    id="prediction-alert",
                    is_open=False,
                    className="prediction-alert",
                    color="info"
                )
            ])
        ], className="mb-4"),
        
        # Main content tabs
        dbc.Tabs([
            dbc.Tab(label="📊 Overview", tab_id="overview"),
            dbc.Tab(label="📈 Trends", tab_id="trends"),
            dbc.Tab(label="🎯 Analysis", tab_id="analysis"),
            dbc.Tab(label="🔢 Tokens", tab_id="tokens"),
            dbc.Tab(label="🔮 Predictions", tab_id="predictions"),
            dbc.Tab(label="⏰ Windows", tab_id="windows"),
        ], id="tabs", active_tab="overview", className="mb-4"),
        
        # Tab content
        html.Div(id="tab-content"),
        
        # Hidden divs to store data
        dcc.Store(id='data-store'),
        dcc.Store(id='stats-store'),
        dcc.Store(id='predictions-store'),
        dcc.Store(id='charts-store'),
        dcc.Store(id='full-data-store'),  # Store unfiltered data
        dcc.Store(id='token-data-store'),  # Store token data
        dcc.Store(id='window-data-store'),  # Store window analysis data
        
    ], fluid=True, style={'backgroundColor': '#0a0a0a'})


def create_overview_content(df: pd.DataFrame, projects: List[str], total_cost: float) -> dbc.Card:
    """Create overview tab content.
    
    Args:
        df: DataFrame with cost data
        projects: List of project names
        total_cost: Total cost
        
    Returns:
        dbc.Card: Overview content
    """
    fig = create_main_figure(df, projects, total_cost)
    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(figure=fig, style={'height': '900px'})
        ])
    ])


def create_trends_content(charts_json: Dict[str, str]) -> html.Div:
    """Create trends tab content.
    
    Args:
        charts_json: Dictionary of serialized charts
        
    Returns:
        html.Div: Trends content
    """
    if 'moving_avg' in charts_json:
        ma_fig = go.Figure(json.loads(charts_json['moving_avg']))
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=ma_fig, style={'height': '500px'})
                    ])
                ])
            ], width=12),
        ])
    return dbc.Alert("Not enough data for trend analysis", color="info")


def create_analysis_content(charts_json: Dict[str, str], stats: Dict[str, Any]) -> dbc.Container:
    """Create analysis tab content.
    
    Args:
        charts_json: Dictionary of serialized charts
        stats: Dictionary of statistics
        
    Returns:
        dbc.Container: Analysis content
    """
    charts_content = []
    
    if 'project_breakdown' in charts_json:
        breakdown_fig = go.Figure(json.loads(charts_json['project_breakdown']))
        charts_content.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=breakdown_fig, style={'height': 'auto'})
                    ])
                ])
            ], width=12)
        )
    
    # Add treemap if available
    if 'treemap' in charts_json:
        treemap_fig = go.Figure(json.loads(charts_json['treemap']))
        charts_content.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=treemap_fig, style={'height': '500px'})
                    ])
                ])
            ], width=12, className="mb-4")
        )
    
    if 'dow' in charts_json:
        dow_fig = go.Figure(json.loads(charts_json['dow']))
        charts_content.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=dow_fig, style={'height': '400px'})
                    ])
                ])
            ], width=6)
        )
    
    # Add hourly analysis if available
    if 'hourly_analysis' in charts_json:
        hourly_fig = go.Figure(json.loads(charts_json['hourly_analysis']))
        charts_content.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=hourly_fig, style={'height': '400px'})
                    ])
                ])
            ], width=6)
        )
    
    # Add detailed statistics
    if stats:
        stats_content = dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Detailed Statistics")),
                dbc.CardBody([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Maximum Daily Cost:", width=8),
                                dbc.Col(f"${stats.get('max_daily_cost', 0):.2f}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Date of Highest Cost:", width=8),
                                dbc.Col(stats.get('max_daily_date', 'N/A'), width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Median Daily Cost:", width=8),
                                dbc.Col(f"${stats.get('median_daily_cost', 0):.2f}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Most Expensive Day of Week:", width=8),
                                dbc.Col(stats.get('most_expensive_dow', 'N/A'), width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Average Hourly Cost (8h/day):", width=8),
                                dbc.Col(f"${stats.get('avg_hourly_cost', 0):.2f}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Maximum Hourly Cost (8h/day):", width=8),
                                dbc.Col(f"${stats.get('max_hourly_cost', 0):.2f}", width=4, className="text-end")
                            ])
                        ]),
                    ])
                ])
            ])
        ], width=12, className="mt-4")
        
        return dbc.Container([
            dbc.Row(charts_content),
            dbc.Row([stats_content])
        ])
    
    return dbc.Row(charts_content)


def create_token_content(charts_json: Dict[str, str], stats: Dict[str, Any]) -> dbc.Container:
    """Create tokens tab content.
    
    Args:
        charts_json: Dictionary of serialized charts
        stats: Dictionary of statistics
        
    Returns:
        dbc.Container: Token content
    """
    token_charts_content = []
    
    # Display token usage charts
    if 'token_usage_time' in charts_json:
        token_time_fig = go.Figure(json.loads(charts_json['token_usage_time']))
        token_charts_content.append(
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=token_time_fig, style={'height': '600px'})
                        ])
                    ])
                ], width=12)
            ], className="mb-4")
        )
    
    # Token breakdown and efficiency in same row
    row_content = []
    if 'token_breakdown' in charts_json:
        token_breakdown_fig = go.Figure(json.loads(charts_json['token_breakdown']))
        row_content.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=token_breakdown_fig, style={'height': '500px'})
                    ])
                ])
            ], width=6)
        )
    
    if 'token_efficiency' in charts_json:
        token_efficiency_fig = go.Figure(json.loads(charts_json['token_efficiency']))
        row_content.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=token_efficiency_fig, style={'height': '400px'})
                    ])
                ])
            ], width=6)
        )
    
    if row_content:
        token_charts_content.append(dbc.Row(row_content, className="mb-4"))
    
    # Token usage by project
    if 'token_by_project' in charts_json:
        token_project_fig = go.Figure(json.loads(charts_json['token_by_project']))
        token_charts_content.append(
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=token_project_fig, style={'height': '400px'})
                        ])
                    ])
                ], width=12)
            ])
        )
    
    # Add token statistics
    if stats and 'total_tokens' in stats:
        token_stats_content = dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Token Usage Statistics")),
                dbc.CardBody([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Total Tokens Used:", width=8),
                                dbc.Col(f"{stats.get('total_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Average Daily Tokens:", width=8),
                                dbc.Col(f"{stats.get('avg_daily_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Maximum Daily Tokens:", width=8),
                                dbc.Col(f"{stats.get('max_daily_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Cost per 1K Tokens:", width=8),
                                dbc.Col(f"${stats.get('cost_per_1k_tokens', 0):.4f}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Input Tokens:", width=8),
                                dbc.Col(f"{stats.get('total_input_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Output Tokens:", width=8),
                                dbc.Col(f"{stats.get('total_output_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Cache Creation Tokens:", width=8),
                                dbc.Col(f"{stats.get('total_cache_creation_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                        dbc.ListGroupItem([
                            dbc.Row([
                                dbc.Col("Cache Read Tokens:", width=8),
                                dbc.Col(f"{stats.get('total_cache_read_tokens', 0):,}", width=4, className="text-end")
                            ])
                        ]),
                    ])
                ])
            ])
        ], width=12, className="mt-4")
        
        token_charts_content.append(dbc.Row([token_stats_content]))
    
    return dbc.Container(token_charts_content)


def create_prediction_content(charts_json: Dict[str, str], stats: Dict[str, Any]) -> dbc.Container:
    """Create predictions tab content.
    
    Args:
        charts_json: Dictionary of serialized charts
        stats: Dictionary of statistics
        
    Returns:
        dbc.Container: Prediction content
    """
    prediction_content = []
    
    # Check if we have predictions
    has_predictions = any(key.startswith('prediction_') for key in charts_json)
    
    if not has_predictions:
        return dbc.Alert("Not enough data for predictions (need at least 6 days)", color="info")
    
    # Model selector
    model_options = []
    if 'prediction_survival' in charts_json:
        model_options.append({'label': 'Survival Analysis', 'value': 'survival'})
    if 'prediction_random' in charts_json:
        model_options.append({'label': 'Random Distribution', 'value': 'random'})
    
    # Add controls
    prediction_content.append(
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Prediction Model Selection", className="card-title"),
                        dcc.RadioItems(
                            id='model-selector',
                            options=[
                                {'label': 'Show Individual Model', 'value': 'individual'},
                                {'label': 'Show All Models Comparison', 'value': 'comparison'}
                            ],
                            value='comparison',
                            inline=True,
                            className="mb-3"
                        ),
                        dcc.Dropdown(
                            id='individual-model-dropdown',
                            options=model_options,
                            value='survival' if 'prediction_survival' in charts_json else model_options[0]['value'] if model_options else None,
                            style={'display': 'none'}
                        )
                    ])
                ])
            ], width=12)
        ], className="mb-4")
    )
    
    # Add model comparison if available
    if 'prediction_comparison' in charts_json:
        comp_fig = go.Figure(json.loads(charts_json['prediction_comparison']))
        prediction_content.append(
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='prediction-graph',
                                figure=comp_fig,
                                style={'height': '600px'}
                            )
                        ])
                    ])
                ], width=12)
            ])
        )
    
    # Add model statistics table
    if stats:
        prediction_content.append(
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Model Fit Statistics")),
                        dbc.CardBody([
                            html.Div(id='model-stats-table')
                        ])
                    ])
                ], width=12)
            ], className="mt-4")
        )
    
    return dbc.Container(prediction_content)