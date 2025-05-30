"""Main entry point for Claude Code Cost Analyzer."""

import webbrowser
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any
import json

import dash
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from .data_processor import refresh_and_load_data, calculate_statistics
from .predictions import predict_future_costs
from .visualizations import create_additional_charts, create_token_charts
from .dashboard import (
    create_layout, create_overview_content, create_trends_content,
    create_analysis_content, create_token_content, create_prediction_content,
    INDEX_STRING
)


def create_app() -> dash.Dash:
    """Create and configure the Dash application.
    
    Returns:
        dash.Dash: Configured Dash application
    """
    # Initialize the Dash app with Bootstrap theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    
    # Set custom index string
    app.index_string = INDEX_STRING
    
    # Set layout
    app.layout = create_layout()
    
    # Register callbacks
    register_callbacks(app)
    
    return app


def register_callbacks(app: dash.Dash) -> None:
    """Register all callbacks for the Dash application.
    
    Args:
        app: Dash application instance
    """
    
    @app.callback(
        [Output('date-range-picker', 'start_date'),
         Output('date-range-picker', 'end_date')],
        [Input('range-1w', 'n_clicks'),
         Input('range-1m', 'n_clicks'),
         Input('range-3m', 'n_clicks'),
         Input('range-all', 'n_clicks')],
        [State('full-data-store', 'data')]
    )
    def update_date_range(n1w, n1m, n3m, nall, full_data):
        """Update date range based on quick selection buttons."""
        if not full_data:
            return None, None
        
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return None, None
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Get the date range from full data
        df = pd.DataFrame.from_dict(full_data['df'])
        if len(df) == 0:
            return None, None
        
        end_date = pd.to_datetime(df.index).max()
        
        if button_id == 'range-1w':
            start_date = end_date - timedelta(days=7)
        elif button_id == 'range-1m':
            start_date = end_date - timedelta(days=30)
        elif button_id == 'range-3m':
            start_date = end_date - timedelta(days=90)
        else:  # range-all
            start_date = pd.to_datetime(df.index).min()
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    @app.callback(
        [Output('data-store', 'data'),
         Output('stats-store', 'data'),
         Output('predictions-store', 'data'),
         Output('charts-store', 'data'),
         Output('last-update', 'children'),
         Output('full-data-store', 'data'),
         Output('date-range-picker', 'min_date_allowed'),
         Output('date-range-picker', 'max_date_allowed'),
         Output('token-data-store', 'data')],
        [Input('refresh-button', 'n_clicks'),
         Input('date-range-picker', 'start_date'),
         Input('date-range-picker', 'end_date')]
    )
    def update_data_stores(n_clicks, start_date, end_date):
        """Load and process all data when refresh is clicked or date range changes."""
        # Refresh JSONL data and load from database
        df_full, projects, _, token_df_full = refresh_and_load_data()
        
        if len(df_full) == 0:
            return {}, {}, None, {}, "No data available", {}, None, None, {}
        
        # Store full data
        full_data = {
            'df': df_full.to_dict(),
            'projects': projects,
            'total_cost': df_full.sum().sum()
        }
        
        # Get date range bounds
        min_date = pd.to_datetime(df_full.index).min().strftime('%Y-%m-%d')
        max_date = pd.to_datetime(df_full.index).max().strftime('%Y-%m-%d')
        
        # Apply date filtering if dates are provided
        df = df_full.copy()
        token_df = token_df_full.copy() if token_df_full is not None else None
        
        if start_date and end_date:
            mask = (pd.to_datetime(df.index) >= pd.to_datetime(start_date)) & \
                   (pd.to_datetime(df.index) <= pd.to_datetime(end_date))
            df = df[mask]
            if token_df is not None and len(token_df) > 0:
                token_df = token_df[mask]
        
        # Recalculate total cost for filtered data
        total_cost = df.sum().sum()
        
        # Calculate statistics on filtered data
        stats = calculate_statistics(df, projects, token_df) if len(df) > 0 else {}
        
        # Generate predictions on filtered data
        predictions = predict_future_costs(df) if len(df) > 0 else None
        
        # Create additional charts with filtered data
        charts = create_additional_charts(df, projects, stats, predictions) if len(df) > 0 else {}
        
        # Create token charts if token data is available
        if token_df is not None and len(token_df) > 0:
            token_charts = create_token_charts(token_df, projects, stats)
            charts.update(token_charts)
        
        # Update timestamp
        date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else " (All time)"
        last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{date_range_text}"
        
        # Store filtered data
        data = {
            'df': df.to_dict(),
            'projects': projects,
            'total_cost': total_cost
        }
        
        # Store token data
        token_data = token_df.to_dict() if token_df is not None else {}
        
        # Convert charts to JSON
        charts_json = {k: v.to_json() for k, v in charts.items()}
        
        return data, stats, predictions, charts_json, last_update, full_data, min_date, max_date, token_data
    
    @app.callback(
        [Output('date-range-info', 'children'),
         Output('date-range-info', 'is_open')],
        [Input('date-range-picker', 'start_date'),
         Input('date-range-picker', 'end_date')],
        [State('full-data-store', 'data')]
    )
    def update_date_range_info(start_date, end_date, full_data):
        """Update the date range information alert."""
        if not start_date or not end_date:
            return "", False
        
        if full_data:
            df_full = pd.DataFrame.from_dict(full_data['df'])
            min_date = pd.to_datetime(df_full.index).min()
            max_date = pd.to_datetime(df_full.index).max()
            
            # Check if we're showing all data
            if (pd.to_datetime(start_date).date() == min_date.date() and 
                pd.to_datetime(end_date).date() == max_date.date()):
                return "Showing all available data", True
        
        # Calculate days in range
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        return f"Showing data from {start_date} to {end_date} ({days} days)", True
    
    @app.callback(
        [Output('total-cost-stat', 'children'),
         Output('daily-avg-stat', 'children'),
         Output('top-project-stat', 'children'),
         Output('active-days-stat', 'children')],
        [Input('stats-store', 'data')]
    )
    def update_stats_cards(stats):
        """Update the statistics cards."""
        if not stats:
            return "$0.00", "$0.00", "N/A", "0"
        
        total = f"${stats.get('total_cost', 0):.2f}"
        daily_avg = f"${stats.get('avg_daily_cost', 0):.2f}"
        top_project = stats.get('top_project', 'N/A')
        if len(top_project) > 15:
            top_project = top_project[:15] + "..."
        active_days = str(stats.get('active_days', 0))
        
        return total, daily_avg, top_project, active_days
    
    @app.callback(
        [Output('prediction-alert', 'children'),
         Output('prediction-alert', 'is_open')],
        [Input('predictions-store', 'data'),
         Input('stats-store', 'data')]
    )
    def update_prediction_alert(predictions, stats):
        """Update the prediction alert."""
        if not predictions or not stats:
            return "", False
        
        # Find the best model based on R²
        best_model_key = None
        best_r2 = -1
        best_model_data = None
        
        for model_key, model_data in predictions.items():
            if model_data['r_squared'] > best_r2:
                best_r2 = model_data['r_squared']
                best_model_key = model_key
                best_model_data = model_data
        
        if not best_model_data:
            return "", False
        
        # Calculate 30-day prediction with best model
        next_30_days_cost = sum(best_model_data['predictions'])
        
        alert_text = [
            html.H5("📊 30-Day Forecast", className="alert-heading"),
            html.P([
                f"Based on the {best_model_data['name']} model (best fit with R²={best_r2:.2f}), ",
                f"your estimated cost for the next 30 days is ",
                html.Strong(f"${next_30_days_cost:.2f}"),
                f" (${next_30_days_cost/30:.2f}/day)."
            ]),
            html.P([
                "Other models predict: ",
                html.Ul([
                    html.Li(f"{data['name']}: ${sum(data['predictions']):.2f} (R²={data['r_squared']:.2f})")
                    for key, data in predictions.items()
                    if key != best_model_key
                ])
            ], className="small")
        ]
        
        return alert_text, True
    
    @app.callback(
        Output('tab-content', 'children'),
        [Input('tabs', 'active_tab'),
         Input('data-store', 'data'),
         Input('stats-store', 'data'),
         Input('charts-store', 'data'),
         Input('token-data-store', 'data')]
    )
    def render_tab_content(active_tab, data_store, stats, charts_json, token_data_store):
        """Render content based on active tab."""
        if not data_store:
            return dbc.Alert("No data available. Click 'Refresh Data' to load.", color="warning")
        
        # Reconstruct dataframe
        df = pd.DataFrame.from_dict(data_store['df'])
        projects = data_store['projects']
        total_cost = data_store['total_cost']
        
        if active_tab == "overview":
            return create_overview_content(df, projects, total_cost)
        
        elif active_tab == "trends":
            return create_trends_content(charts_json)
        
        elif active_tab == "analysis":
            return create_analysis_content(charts_json, stats)
        
        elif active_tab == "tokens":
            if not token_data_store:
                return dbc.Alert("No token usage data available.", color="info")
            return create_token_content(charts_json, stats)
        
        elif active_tab == "predictions":
            return create_prediction_content(charts_json, stats)
        
        return dbc.div()
    
    @app.callback(
        Output('individual-model-dropdown', 'style'),
        [Input('model-selector', 'value')]
    )
    def toggle_model_dropdown(selector_value):
        """Show/hide the individual model dropdown based on selector."""
        if selector_value == 'individual':
            return {'display': 'block'}
        return {'display': 'none'}
    
    @app.callback(
        Output('prediction-graph', 'figure'),
        [Input('model-selector', 'value'),
         Input('individual-model-dropdown', 'value')],
        [State('charts-store', 'data')]
    )
    def update_prediction_graph(selector_value, individual_model, charts_json):
        """Update the prediction graph based on model selection."""
        if not charts_json:
            return go.Figure()
        
        if selector_value == 'comparison' and 'prediction_comparison' in charts_json:
            return go.Figure(json.loads(charts_json['prediction_comparison']))
        elif selector_value == 'individual' and f'prediction_{individual_model}' in charts_json:
            return go.Figure(json.loads(charts_json[f'prediction_{individual_model}']))
        
        return go.Figure()
    
    @app.callback(
        Output('model-stats-table', 'children'),
        [Input('predictions-store', 'data')]
    )
    def update_model_stats(predictions):
        """Update the model statistics table."""
        if not predictions:
            return dbc.p("No model statistics available")
        
        # Create table rows
        rows = []
        for model_key, model_data in predictions.items():
            # Calculate 30-day cost
            cost_30d = sum(model_data['predictions'])
            
            rows.append(
                html.Tr([
                    html.Td(model_data['name']),
                    html.Td(f"{model_data['r_squared']:.4f}"),
                    html.Td(f"${cost_30d:.2f}"),
                    html.Td(f"${cost_30d/30:.2f}"),
                    html.Td(model_data['equation'], style={'fontSize': '0.9em'})
                ])
            )
        
        # Sort by R² descending
        rows.sort(key=lambda x: float(x.children[1].children), reverse=True)
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Model"),
                    html.Th("R² Score"),
                    html.Th("30-Day Forecast"),
                    html.Th("Daily Average"),
                    html.Th("Equation")
                ])
            ]),
            html.Tbody(rows)
        ], striped=True, hover=True, responsive=True, className="text-white")


def main():
    """Main entry point for the application."""
    app = create_app()
    
    port = 8050
    print(f"Starting Claude Code Cost Dashboard on http://localhost:{port}/")
    print("Press Ctrl+C to stop the server")
    

    # Run the app
    app.run(debug=False, port=port)


if __name__ == '__main__':
    main()