# Claude Code Cost Analyzer

An advanced analytics dashboard for tracking and visualizing Claude Code usage costs with comprehensive token tracking.

## Features

- **Cost Tracking**: Track daily costs by project with beautiful visualizations
- **Token Usage Analysis**: Monitor token usage patterns (input, output, cache creation, cache read)
- **Predictive Analytics**: Forecast future costs using survival analysis and statistical models
- **Interactive Dashboard**: Web-based dashboard with dark theme and responsive design
- **Database Persistence**: SQLite database for efficient data storage and retrieval
- **Advanced Statistics**: Detailed cost breakdowns, trends, and efficiency metrics

## Installation

### Using uv (recommended)

If you have `uv` installed, you can run the analyzer directly:

```bash
# Make the script executable (first time only)
chmod +x analyzer.py

# Run directly
./analyzer.py

# Or with uv run
uv run analyzer.py
```

### Using pip

Install the package and its dependencies:

```bash
pip install -e .
# or for development with tests
pip install -e ".[dev]"
```

Then run:

```bash
python analyzer.py
# or if installed
claude-cost-analyzer
```

## Architecture

The project is organized into modular components:

- `cost_analyzer/database.py` - SQLite database operations
- `cost_analyzer/data_processor.py` - Data loading and processing
- `cost_analyzer/predictions.py` - Cost prediction models
- `cost_analyzer/visualizations.py` - Chart generation
- `cost_analyzer/dashboard.py` - Dash UI components
- `cost_analyzer/main.py` - Application entry point

## Testing

Run the comprehensive test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=cost_analyzer
```

## Dashboard Views

1. **Overview**: Daily and cumulative cost charts
2. **Trends**: Moving averages and trend analysis
3. **Analysis**: Project breakdowns, day-of-week analysis, hourly cost distribution
4. **Tokens**: Token usage tracking with type breakdowns and efficiency metrics
5. **Predictions**: 30-day cost forecasts using multiple models

## Token Tracking

The dashboard now tracks all token types from Claude Code:
- Input tokens
- Output tokens  
- Cache creation tokens
- Cache read tokens

Token data is automatically extracted from the `usage` object in each JSONL entry and stored in the database for analysis.

## Cost Predictions

Two prediction models are available:
- **Survival Analysis**: Models cost patterns as "survival" of high-cost periods
- **Random Distribution**: Treats daily costs as normally distributed random variables

Both models respect physical constraints (8-hour workday limit) and provide R� scores for model fit evaluation.