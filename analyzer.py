#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas>=2.0.0",
#   "plotly>=6.0.0",
#   "dash>=2.0.0",
#   "dash-bootstrap-components>=1.0.0",
#   "scikit-learn>=1.0.0",
#   "numpy>=1.20.0",
# ]
# ///
"""
Claude Code Cost Dashboard

This script analyzes Claude Code usage costs from .jsonl files in the Claude projects directory.
It creates an interactive Dash dashboard with comprehensive visualizations including:
- Daily costs by project (stacked area chart)
- Cumulative costs over time (line/bar chart)
- Token usage tracking and analysis
- Cost predictions using multiple models
- Detailed statistics and trends

Usage:
  ./analyzer.py
  # or
  uv run analyzer.py
  # or if installed:
  claude-cost-analyzer
"""

import sys
import os

# Add the current directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost_analyzer.main import main

if __name__ == '__main__':
    main()