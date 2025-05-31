#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

"""Clear saved credit windows to force recalculation with updated logic."""

import sqlite3
import os
from pathlib import Path

def clear_windows():
    """Clear the credit_windows and window_messages tables."""
    db_path = Path.home() / '.claude' / 'usage_data.db'
    
    if not db_path.exists():
        print("Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Clear window messages first (due to foreign key)
        cursor.execute("DELETE FROM window_messages")
        messages_deleted = cursor.rowcount
        
        # Clear credit windows
        cursor.execute("DELETE FROM credit_windows")
        windows_deleted = cursor.rowcount
        
        conn.commit()
        print(f"✅ Cleared {windows_deleted} windows and {messages_deleted} messages")
        print("Windows will be recalculated on next dashboard load")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    clear_windows()