#!/usr/bin/env python3
"""
Script to find Intel copyrights with year ranges and convert them to single years.
Example: "Copyright 2022-2023 Intel Corporation" -> "Copyright 2022 Intel Corporation"
"""

import re
import os
import glob
import argparse
from pathlib import Path

def process_file(filepath, dry_run=False):
    """
    Process a single file to convert Intel copyright year ranges to single years.
    
    Args:
        filepath (str): Path to the file to process
        dry_run (bool): If True, only show what would be changed without modifying files
    
    Returns:
        bool: True if changes were made (or would be made in dry_run), False otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    # Pattern to match Intel copyright with year ranges
    # Matches: "Copyright YYYY-YYYY Intel Corporation" or "* Copyright YYYY-YYYY Intel Corporation"
    pattern = r'(\s*\*?\s*Copyright\s+)(\d{4})-(\d{4})(\s+Intel\s+Corporation)'
    
    def replace_intel_copyright(match):
        prefix = match.group(1)  # "* Copyright " or "Copyright "
        first_year = match.group(2)  # First year (e.g., "2022")
        second_year = match.group(3)  # Second year (e.g., "2023")
        suffix = match.group(4)  # " Intel Corporation"
        
        # Keep only the first year
        return f"{prefix}{first_year}{suffix}"
    
    # Find all matches
    matches = list(re.finditer(pattern, content))
    
    if not matches:
        return False
    
    # Show what will be changed
    print(f"\n📄 {filepath}")
    for match in matches:
        old_text = match.group(0)
        new_text = replace_intel_copyright(match)
        print(f"  🔄 '{old_text.strip()}' -> '{new_text.strip()}'")
    
    if not dry_run:
        # Apply the changes
        new_content = re.sub(pattern, replace_intel_copyright, content)
        
        # Write back to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ File updated successfully")
        except Exception as e:
            print(f"  ❌ Error writing {filepath}: {e}")
            return False
    else:
        print(f"  ℹ️  Dry run - no changes made")
    
    return True

def find_source_files(root_dir):
    """
    Find all source files (C/C++ headers and implementation files).
    
    Args:
        root_dir (str): Root directory to search
    
    Returns:
        list: List of source file paths
    """
    patterns = [
        '**/*.cpp',
        '**/*.hpp', 
        '**/*.cl', 
        '**/*.txt', 
        '**/*.cmake', 
        '**/*.h.in', 
        '**/*.db',
        '**/*.yaml',  
        '**/*.*.in',  
        '**/*.py', 
        '**/*.c',
        '**/*.h',
        '**/*.cc',
        '**/*.cxx',
        '**/*.hxx'
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    
    return sorted(set(files))

def main():
    parser = argparse.ArgumentParser(
        description='Convert Intel copyright year ranges to single years'
    )
    parser.add_argument(
        'directory', 
        nargs='?', 
        default='.', 
        help='Directory to search for files (default: current directory)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--file', 
        help='Process a single file instead of directory'
    )
    parser.add_argument(
        '--extensions', 
        nargs='+', 
        default=['cpp', 'hpp', 'c', 'h', 'cc', 'cxx', 'hxx'],
        help='File extensions to process'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1
        
        files = [args.file]
    else:
        # Process directory
        if not os.path.isdir(args.directory):
            print(f"❌ Directory not found: {args.directory}")
            return 1
        
        print(f"🔍 Searching for source files in: {args.directory}")
        files = find_source_files(args.directory)
        
        if not files:
            print("❌ No source files found")
            return 1
        
        print(f"📁 Found {len(files)} source files")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    # Process files
    modified_count = 0
    for filepath in files:
        if process_file(filepath, args.dry_run):
            modified_count += 1
    
    # Summary
    print(f"\n📊 Summary:")
    if args.dry_run:
        print(f"  📄 Files that would be modified: {modified_count}")
        print(f"  📄 Total files scanned: {len(files)}")
    else:
        print(f"  ✅ Files modified: {modified_count}")
        print(f"  📄 Total files scanned: {len(files)}")
    
    return 0

if __name__ == '__main__':
    exit(main())