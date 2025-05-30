# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced Claude Code Cost Analytics Dashboard that provides comprehensive visualization and predictive analysis of Claude Code usage costs. It reads JSONL files from `~/.claude/projects/` and creates a sophisticated interactive web dashboard with multiple analysis views, statistics, and cost predictions.

## Running the Dashboard

The analyzer is a self-contained Python script with inline dependencies managed by `uv`:

```bash
# Make the script executable (if needed)
chmod +x analyzer.py

# Run the dashboard
./analyzer.py
```

The script will:
- Automatically install dependencies (pandas, plotly, dash, scikit-learn, numpy, dash-bootstrap-components) via uv
- Start a local web server on http://localhost:8050/
- Open your browser automatically to display the dashboard
- Provide real-time data refresh capabilities

## Key Architecture

The dashboard processes Claude Code usage data through these components:

### Data Processing Pipeline
1. **Data Loading**: Scans `~/.claude/projects/` recursively for `.jsonl` files
2. **Project Extraction**: Intelligently extracts and cleans project names from file paths
3. **Cost Aggregation**: Aggregates costs by date and project from the `costUSD` field
4. **Statistical Analysis**: Calculates comprehensive statistics including averages, medians, and trends
5. **Predictive Modeling**: Uses polynomial regression to forecast future costs

### Dashboard Features
- **Date Range Selection**: 
  - Interactive date picker for custom date ranges
  - Quick selection buttons (1 Week, 1 Month, 3 Months, All)
  - All charts and statistics update based on selected range
  - Visual indicator showing current date range
- **Multiple Views**: Tabbed interface with Overview, Trends, Analysis, and Predictions
- **Real-time Statistics**: Live-updating cards showing total cost, daily average, top project, and active days
- **Advanced Visualizations**:
  - Stacked area charts for daily costs by project
  - Cumulative cost tracking with bar/line combination
  - Horizontal bar charts for project cost breakdown (replaced pie charts)
  - Treemap visualization for hierarchical cost display
  - Day-of-week analysis
  - Moving averages (7-day and 30-day)
  - Cost prediction charts with confidence indicators
- **Dark Theme**: Professional dark theme using Bootstrap Cyborg theme
- **Responsive Design**: Fully responsive layout that works on different screen sizes

## Predictive Analysis

The dashboard includes sophisticated cost prediction capabilities with multiple regression models:

### Available Models
1. **Survival Analysis**: Models cost patterns as "survival" of high-cost periods, predicting decay toward median
2. **Random Distribution**: Assumes daily costs follow a normal distribution N(μ, σ) with physical constraints

### Features
- **Model Selection**: Choose between individual model view or comparison of all models
- **Model Comparison Chart**: View all models on a single chart to compare predictions
- **Fit Statistics Table**: See R² scores, 30-day forecasts, and equations for each model
- **Best Model Recommendation**: The prediction alert automatically shows the best-fitting model
- **30-Day Forecast**: Each model predicts costs for the next 30 days
- **Model Confidence**: R² values indicate how well each model fits the historical data
- **Visual Indicators**: Shows historical fit lines alongside actual data points
- **Minimum Data Requirement**: Needs at least 6 days of data for predictions

### Physical Constraints
All models now respect the physical limit of an 8-hour workday:
- **Maximum Hourly Rate**: Based on your highest observed daily cost divided by 8 hours
- **Daily Cap**: All predictions are capped at 8 hours × maximum hourly rate
- **Visual Indicator**: Red dotted line shows the physical limit on charts
- **Realistic Bounds**: Prevents any model from predicting costs beyond physical capacity

### Random Distribution Model
Treats daily costs as normally distributed random variables:
- **Mean and Standard Deviation**: Calculated from historical daily costs
- **Normal Distribution**: Future costs sampled from N(μ, σ)
- **Physical Constraints**: Each sample is capped at the daily maximum
- **Uncertainty Representation**: Shows natural variability in daily costs
- **No Trend Assumption**: Assumes costs fluctuate around a stable mean

### Survival Analysis
Treats high-cost periods as "events" that have a certain survival probability:
- **Threshold Analysis**: Calculates how long costs typically stay above 50th, 75th, and 90th percentiles
- **Decay Model**: Predicts exponential decay from current levels toward the median cost
- **Mean Duration**: Shows average duration of high-cost periods for each threshold
- **Trend Integration**: Incorporates recent trends with damping to avoid unrealistic projections

### Hourly Cost Analysis
- **Box Plot Visualization**: Shows distribution of hourly costs across all days
- **Statistical Markers**: Displays average hourly rate and physical limit (maximum)
- **Physical Limit**: Maximum hourly rate represents your physical productivity ceiling
- **Work Capacity Insights**: Helps understand how close you are to maximum productivity

## Development Notes

- The script uses `uv` for dependency management with inline script dependencies
- Requires Python >=3.9
- Uses `os.path.expanduser()` for cross-platform home directory resolution
- Handles malformed JSONL entries gracefully with try/except blocks
- Project names are simplified to show only the last portion of the path
- All charts use Plotly's dark theme for consistency
- Data is stored in Dash Store components for efficient cross-callback communication
- Date filtering is applied consistently across all visualizations and statistics