#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas>=2.0.0"
# ]
# ///

"""
Test script to verify that credit windows are correctly persisted to the database.
"""

import os
import sys
import sqlite3
from datetime import datetime

# Add the cost_analyzer module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cost_analyzer.database import get_database_path, init_database, load_credit_windows
from cost_analyzer.data_processor import get_window_analysis


def check_database_tables():
    """Check if the window tables exist in the database."""
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        print("❌ Database does not exist yet")
        return False
    
    print(f"✅ Database found at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check for tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name IN ('credit_windows', 'window_messages')
    """)
    
    tables = cursor.fetchall()
    table_names = [t[0] for t in tables]
    
    if 'credit_windows' in table_names:
        print("✅ credit_windows table exists")
        
        # Count windows
        cursor.execute("SELECT COUNT(*) FROM credit_windows")
        window_count = cursor.fetchone()[0]
        print(f"   Found {window_count} credit windows")
        
        # Show recent windows
        cursor.execute("""
            SELECT window_id, start_time, end_time, total_cost, reached_half_credit
            FROM credit_windows
            ORDER BY start_time DESC
            LIMIT 5
        """)
        
        recent_windows = cursor.fetchall()
        if recent_windows:
            print("\n   Recent windows:")
            for w in recent_windows:
                window_id, start, end, cost, half_credit = w
                print(f"   - {start} to {end}: ${cost:.2f} (half-credit: {'Yes' if half_credit else 'No'})")
    else:
        print("❌ credit_windows table does not exist")
    
    if 'window_messages' in table_names:
        print("\n✅ window_messages table exists")
        
        # Count messages
        cursor.execute("SELECT COUNT(*) FROM window_messages")
        message_count = cursor.fetchone()[0]
        print(f"   Found {message_count} window messages")
    else:
        print("❌ window_messages table does not exist")
    
    conn.close()
    return True


def test_window_loading():
    """Test loading windows from the database."""
    print("\n\n📊 Testing Window Loading")
    print("-" * 40)
    
    try:
        # Load windows directly from database
        windows = load_credit_windows()
        print(f"✅ Loaded {len(windows)} windows from database")
        
        if windows:
            # Show summary
            total_cost = sum(w['total_cost'] for w in windows)
            half_credit_count = sum(1 for w in windows if w['reached_half_credit'])
            
            print(f"\nWindow Summary:")
            print(f"  Total windows: {len(windows)}")
            print(f"  Total cost: ${total_cost:.2f}")
            print(f"  Windows with half-credit: {half_credit_count}")
            
            # Show first and last window
            first_window = windows[0]
            last_window = windows[-1]
            
            print(f"\n  First window: {first_window['start_time']}")
            print(f"  Last window: {last_window['start_time']}")
    except Exception as e:
        print(f"❌ Error loading windows: {e}")
        import traceback
        traceback.print_exc()


def test_window_analysis():
    """Test the get_window_analysis function."""
    print("\n\n🔍 Testing Window Analysis")
    print("-" * 40)
    
    try:
        analysis = get_window_analysis()
        
        print(f"✅ Window analysis completed")
        print(f"   Total windows: {analysis.get('total_windows', 0)}")
        print(f"   Windows with half-credit: {analysis.get('windows_with_half_credit', 0)}")
        print(f"   Average hours to half-credit: {analysis.get('avg_hours_to_half_credit', 'N/A')}")
        
        if 'windows' in analysis and analysis['windows']:
            print(f"\n   Windows in analysis: {len(analysis['windows'])}")
            
            # Check if windows have UUIDs
            window_with_uuid = 0
            for w in analysis['windows']:
                if w.get('messages'):
                    for msg in w['messages']:
                        if msg.get('uuid'):
                            window_with_uuid += 1
                            break
            
            print(f"   Windows with message UUIDs: {window_with_uuid}")
            
    except Exception as e:
        print(f"❌ Error in window analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 Testing Credit Window Persistence")
    print("=" * 50)
    
    # Initialize database if needed
    print("\n📁 Initializing Database")
    print("-" * 40)
    db_path = init_database()
    print(f"✅ Database initialized at: {db_path}")
    
    # Check database structure
    print("\n\n🗄️  Checking Database Structure")
    print("-" * 40)
    check_database_tables()
    
    # Test window loading
    test_window_loading()
    
    # Test window analysis
    test_window_analysis()
    
    print("\n\n✅ Test complete!")
    print("\nNOTE: If windows were detected and saved, they will persist even if JSONL files are deleted.")
    print("The dashboard will automatically use saved windows unless new data is detected.")


if __name__ == "__main__":
    main()