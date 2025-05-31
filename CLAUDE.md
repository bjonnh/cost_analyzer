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

## Data Validation Testing

A standalone test script is available to validate JSONL data loading and check for timezone/time issues:

```bash
# Make the script executable (if needed)
chmod +x test_data_validation.py

# Run the validation test
./test_data_validation.py
```

The test script performs comprehensive validation:

### What It Tests
1. **Timestamp Validation**
   - Checks for missing or invalid timestamps
   - Validates timezone conversions (UTC to local)
   - Identifies suspicious times (e.g., exact midnight)
   - Shows examples of timestamp conversion

2. **Credit Window Detection**
   - Detects 5-hour credit windows based on activity gaps
   - Windows are global across ALL JSONL files
   - Tracks Opus → Sonnet model transitions
   - Identifies when "half-credit" is reached using a rolling window approach:
     - Checks the last 10 messages in the window
     - Half-credit triggers when >80% of these recent messages are Sonnet
   - Shows windows with high Sonnet usage (>50%) for analysis

3. **Half-Credit Time Analysis**
   - Analyzes all "half-credit reached" times
   - Detects suspicious patterns (e.g., always showing 00:37)
   - Calculates hours to reach half-credit
   - Shows recent half-credit times with full context

4. **Data Summary**
   - Date range of all data
   - Model distribution (Opus vs Sonnet usage)
   - Total messages and costs

### Example Issues It Can Detect
- **Timezone Problems**: If times are showing in UTC instead of local time
- **Stale Data**: If "half-credit reached" shows time from a previous window
- **Data Anomalies**: Unusually quick transitions to half-credit
- **Pattern Detection**: Repeated exact times (like always 00:37)

### Output Format
The test uses Rich for formatted console output with:
- Color-coded status messages
- Formatted tables showing timestamp conversions
- Highlighted warnings for suspicious patterns
- Summary statistics

This test is useful for debugging data issues before they appear in the dashboard.

### Important: After Changes
When the window detection logic is changed (e.g., the half-credit threshold), you need to:
1. Restart the dashboard: `./analyzer.py`
2. Run the validation test: `./test_data_validation.py`
3. Compare the results to ensure windows are detected correctly

## Key Architecture

The dashboard processes Claude Code usage data through these components:

### Data Processing Pipeline
1. **Data Loading**: Scans `~/.claude/projects/` recursively for `.jsonl` files
2. **Data Persistence**: All usage data and credit windows are stored in SQLite database at `~/.claude/usage_data.db`
3. **Window Detection**: Credit windows are global across ALL JSONL files, not per-file
   - Windows are saved to the database for persistence
   - Automatically reloaded even if JSONL files are deleted
   - Recalculated only when new data is detected
4. **Project Extraction**: Intelligently extracts and cleans project names from file paths
5. **Cost Aggregation**: Aggregates costs by date and project from the `costUSD` field
6. **Statistical Analysis**: Calculates comprehensive statistics including averages, medians, and trends
7. **Predictive Modeling**: Uses polynomial regression to forecast future costs

### Credit Windows
- **5-Hour Windows**: Each window lasts 5 hours from first activity
- **Global Windows**: Windows span across all JSONL files based on timestamps
- **Half-Credit Detection**: 
  - Triggered when switching from Opus to Sonnet
  - **Important**: Checks the last 10 messages in a window
  - Half-credit is reached when >80% of the last 10 messages are Sonnet
  - This rolling window approach detects sustained switches to Sonnet, not temporary spikes

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

- **IMPORTANT**: The script uses `uv` for dependency management with inline script dependencies. You MUST use `uv` for all Python tools and scripts you create in this project. This is critical for proper dependency management and script execution.
- All Python scripts should start with `#!/usr/bin/env -S uv run --script` and include inline dependencies in the script header
- Requires Python >=3.9
- Uses `os.path.expanduser()` for cross-platform home directory resolution
- Handles malformed JSONL entries gracefully with try/except blocks
- Project names are simplified to show only the last portion of the path
- All charts use Plotly's dark theme for consistency
- Data is stored in Dash Store components for efficient cross-callback communication
- Date filtering is applied consistently across all visualizations and statistics
- All timestamps from JSONL files (UTC with 'Z' suffix) are converted to local timezone for display
- Half-credit times are properly converted from UTC to local timezone in the dashboard

### Database Structure
- SQLite database at `~/.claude/usage_data.db` stores:
  - `usage_records`: Individual usage entries from JSONL files
  - `credit_windows`: Detected 5-hour credit windows
  - `window_messages`: Messages within each window
- Windows are persisted to survive JSONL file deletions
- Database is automatically created on first run

## Workflow

- Don't hesitate to ask clarifying questions and update your plan when you get answers
- Type check and run builds regularly once things are ok
- Make plans and describe how you will know you have achieved those. If you detect issues you can write them in TODO.md and continue on with your task
- If this is a feature that requires to manage data or that require analysis don't hesitate to write a tool in an adequate language (Python or the project language preferred)
- Write and maintain tests, tests are inexpensive and allow you to work much better
- When things don't work as you had planned, update the CLAUDE.md file to reflect that (talk always in positive)
- **Remember to always run tests and always write tests**