#!/usr/bin/env python
"""Examine JSONL structure to find token data."""

import os
import json
import glob
from collections import defaultdict

def examine_jsonl_structure():
    """Examine the structure of JSONL files to find where token data might be."""
    print("=== EXAMINING JSONL STRUCTURE ===\n")
    
    project_dir = os.path.expanduser("~/.claude/projects/")
    jsonl_files = glob.glob(f"{project_dir}/**/*.jsonl", recursive=True)
    
    if not jsonl_files:
        print("No JSONL files found!")
        return
    
    # Collect all unique field names and their types
    field_stats = defaultdict(lambda: {'count': 0, 'types': set(), 'samples': []})
    message_fields = defaultdict(lambda: {'count': 0, 'types': set(), 'samples': []})
    
    total_entries = 0
    entries_with_cost = 0
    
    print(f"Examining {len(jsonl_files)} JSONL files...\n")
    
    for file_idx, file_path in enumerate(jsonl_files):
        if file_idx >= 10:  # Limit to first 10 files
            break
            
        try:
            with open(file_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx >= 20:  # Check up to 20 entries per file
                        break
                        
                    try:
                        entry = json.loads(line)
                        total_entries += 1
                        
                        if entry.get('costUSD', 0) > 0:
                            entries_with_cost += 1
                        
                        # Analyze top-level fields
                        for field, value in entry.items():
                            field_stats[field]['count'] += 1
                            field_stats[field]['types'].add(type(value).__name__)
                            
                            # Collect sample values for interesting fields
                            if field in ['type', 'version'] and len(field_stats[field]['samples']) < 5:
                                if value not in field_stats[field]['samples']:
                                    field_stats[field]['samples'].append(value)
                        
                        # Analyze message field specifically
                        if 'message' in entry and isinstance(entry['message'], dict):
                            for msg_field, msg_value in entry['message'].items():
                                message_fields[msg_field]['count'] += 1
                                message_fields[msg_field]['types'].add(type(msg_value).__name__)
                                
                                # Look for token-related fields
                                if 'token' in msg_field.lower() or msg_field in ['usage', 'metrics', 'stats']:
                                    if len(message_fields[msg_field]['samples']) < 3:
                                        message_fields[msg_field]['samples'].append(msg_value)
                                        
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Print findings
    print(f"Total entries examined: {total_entries}")
    print(f"Entries with cost > 0: {entries_with_cost}")
    
    print("\n=== TOP-LEVEL FIELDS ===")
    sorted_fields = sorted(field_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for field, stats in sorted_fields[:20]:
        types = ', '.join(stats['types'])
        print(f"\n{field}: {stats['count']} occurrences ({types})")
        if stats['samples']:
            print(f"  Samples: {stats['samples'][:3]}")
    
    print("\n=== MESSAGE FIELDS ===")
    if message_fields:
        sorted_msg_fields = sorted(message_fields.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for field, stats in sorted_msg_fields[:20]:
            types = ', '.join(stats['types'])
            print(f"\n{field}: {stats['count']} occurrences ({types})")
            if stats['samples']:
                print(f"  Samples: {stats['samples'][:3]}")
                
        # Look for any field that might contain token data
        print("\n=== POTENTIAL TOKEN FIELDS ===")
        token_related = []
        for field, stats in message_fields.items():
            if any(word in field.lower() for word in ['token', 'usage', 'metric', 'stat', 'count']):
                token_related.append((field, stats))
        
        if token_related:
            print("Found fields that might contain token data:")
            for field, stats in token_related:
                print(f"  {field}: {stats['count']} occurrences")
                if stats['samples']:
                    print(f"    Sample: {stats['samples'][0]}")
        else:
            print("No token-related fields found in message objects")
    else:
        print("No message fields found or messages are not dictionaries")
    
    # Look for a specific entry with cost to see its full structure
    print("\n=== SAMPLE ENTRY WITH COST ===")
    for file_path in jsonl_files[:10]:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('costUSD', 0) > 0:
                            print(f"\nFrom file: {os.path.basename(file_path)}")
                            print(json.dumps(entry, indent=2))
                            return  # Just show one example
                    except:
                        pass
        except:
            pass


if __name__ == "__main__":
    examine_jsonl_structure()