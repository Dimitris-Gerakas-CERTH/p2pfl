#!/usr/bin/env python3
"""
Clean Actigraph CSV files for Federated Learning Health Demo.

This script:
1. Reads each Actigraph{1-22}.csv file
2. Keeps only the columns: Axis1, Axis2, Axis3, HR
3. Adds a sequential 'timestep' column (0, 1, 2, ...) for time series ordering
4. Handles missing values (drops rows with NaN in HR since that's our target)
5. Saves cleaned files to a specified output directory

Usage:
    python clean_actigraph_data.py --input_dir /path/to/raw --output_dir /path/to/clean
    
    # Or with defaults (current directory):
    python clean_actigraph_data.py
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import sys


def clean_single_file(
    input_path: Path,
    output_path: Path,
    columns_to_keep: list = ['Axis1', 'Axis2', 'Axis3', 'HR'],
    add_timestep: bool = True,
    drop_na_hr: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Clean a single Actigraph CSV file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        columns_to_keep: List of column names to retain
        add_timestep: Whether to add a sequential timestep column
        drop_na_hr: Whether to drop rows where HR is NaN
        verbose: Print progress information
        
    Returns:
        Dictionary with statistics about the cleaning process
    """
    stats = {
        'input_file': str(input_path),
        'output_file': str(output_path),
        'original_rows': 0,
        'original_cols': 0,
        'final_rows': 0,
        'final_cols': 0,
        'rows_dropped_na': 0,
        'success': False,
        'error': None
    }
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        stats['original_rows'] = len(df)
        stats['original_cols'] = len(df.columns)
        
        if verbose:
            print(f"  📂 Loaded {input_path.name}: {len(df):,} rows, {len(df.columns)} columns")
        
        # Check which columns exist (handle case sensitivity)
        available_cols = df.columns.tolist()
        cols_to_select = []
        
        for col in columns_to_keep:
            # Try exact match first
            if col in available_cols:
                cols_to_select.append(col)
            # Try case-insensitive match
            else:
                matches = [c for c in available_cols if c.lower() == col.lower()]
                if matches:
                    cols_to_select.append(matches[0])
                else:
                    print(f"  ⚠️  Warning: Column '{col}' not found in {input_path.name}")
        
        if not cols_to_select:
            raise ValueError(f"None of the required columns found. Available: {available_cols}")
        
        # Select only the columns we want
        df_clean = df[cols_to_select].copy()
        
        # Standardize column names (ensure consistent casing)
        df_clean.columns = [col.lower() for col in df_clean.columns]
        
        # Drop rows with NaN in HR (since it's our prediction target)
        if drop_na_hr and 'hr' in df_clean.columns:
            rows_before = len(df_clean)
            df_clean = df_clean.dropna(subset=['hr'])
            stats['rows_dropped_na'] = rows_before - len(df_clean)
            if stats['rows_dropped_na'] > 0 and verbose:
                print(f"  🗑️  Dropped {stats['rows_dropped_na']:,} rows with missing HR values")
        
        # Reset index to ensure continuous ordering
        df_clean = df_clean.reset_index(drop=True)
        
        # Add timestep column at the beginning
        if add_timestep:
            df_clean.insert(0, 'timestep', range(len(df_clean)))
        
        # Save the cleaned file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        
        stats['final_rows'] = len(df_clean)
        stats['final_cols'] = len(df_clean.columns)
        stats['success'] = True
        
        if verbose:
            print(f"  ✅ Saved {output_path.name}: {len(df_clean):,} rows, {len(df_clean.columns)} columns")
            print(f"     Columns: {list(df_clean.columns)}")
        
    except Exception as e:
        stats['error'] = str(e)
        if verbose:
            print(f"  ❌ Error processing {input_path.name}: {e}")
    
    return stats


def clean_all_files(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "Actigraph{}.csv",
    num_files: int = 22,
    verbose: bool = True
) -> Tuple[list, dict]:
    """
    Clean all Actigraph CSV files.
    
    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory to save cleaned CSV files
        file_pattern: Pattern for input filenames (use {} for number)
        num_files: Number of files to process (1 to num_files)
        verbose: Print progress information
        
    Returns:
        Tuple of (list of stats dicts, summary dict)
    """
    all_stats = []
    
    print("=" * 60)
    print("🏥 ACTIGRAPH DATA CLEANING FOR FEDERATED LEARNING")
    print("=" * 60)
    print(f"\n📁 Input directory:  {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Files to process: {num_files}")
    print()
    
    # Process each file
    for i in range(1, num_files + 1):
        input_filename = file_pattern.format(i)
        output_filename = f"patient_{i:02d}_clean.csv"
        
        input_path = input_dir / input_filename
        output_path = output_dir / output_filename
        
        print(f"\n[{i}/{num_files}] Processing {input_filename}...")
        
        if not input_path.exists():
            print(f"  ⚠️  File not found: {input_path}")
            all_stats.append({
                'input_file': str(input_path),
                'success': False,
                'error': 'File not found'
            })
            continue
        
        stats = clean_single_file(input_path, output_path, verbose=verbose)
        all_stats.append(stats)
    
    # Calculate summary
    successful = [s for s in all_stats if s.get('success', False)]
    failed = [s for s in all_stats if not s.get('success', False)]
    
    summary = {
        'total_files': num_files,
        'successful': len(successful),
        'failed': len(failed),
        'total_rows_processed': sum(s.get('final_rows', 0) for s in successful),
        'total_rows_dropped': sum(s.get('rows_dropped_na', 0) for s in successful),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CLEANING SUMMARY")
    print("=" * 60)
    print(f"\n  ✅ Successfully processed: {summary['successful']}/{summary['total_files']} files")
    if summary['failed'] > 0:
        print(f"  ❌ Failed: {summary['failed']} files")
        for s in failed:
            print(f"     - {s.get('input_file', 'unknown')}: {s.get('error', 'unknown error')}")
    print(f"\n  📈 Total rows in cleaned data: {summary['total_rows_processed']:,}")
    print(f"  🗑️  Total rows dropped (missing HR): {summary['total_rows_dropped']:,}")
    
    if successful:
        print(f"\n  📁 Cleaned files saved to: {output_dir}")
        print(f"  📋 Output format: patient_XX_clean.csv")
        print(f"  📊 Columns: timestep, axis1, axis2, axis3, hr")
    
    print("\n" + "=" * 60)
    
    return all_stats, summary


def preview_cleaned_data(output_dir: Path, num_files: int = 3):
    """Preview the first few cleaned files."""
    print("\n📋 PREVIEW OF CLEANED DATA")
    print("-" * 40)
    
    for i in range(1, min(num_files + 1, 4)):
        filepath = output_dir / f"patient_{i:02d}_clean.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"\n{filepath.name}:")
            print(df.head(10).to_string(index=False))
            print(f"... ({len(df):,} total rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Actigraph CSV files for Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Clean files in current directory
    python clean_actigraph_data.py
    
    # Specify input and output directories
    python clean_actigraph_data.py --input_dir ./raw_data --output_dir ./clean_data
    
    # Process only first 5 files
    python clean_actigraph_data.py --num_files 5
    
    # Show preview of cleaned data
    python clean_actigraph_data.py --preview
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='.',
        help='Directory containing Actigraph CSV files (default: current directory)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./cleaned_data',
        help='Directory to save cleaned CSV files (default: ./cleaned_data)'
    )
    parser.add_argument(
        '--num_files', '-n',
        type=int,
        default=22,
        help='Number of files to process (default: 22)'
    )
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show preview of cleaned data after processing'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Run cleaning
    all_stats, summary = clean_all_files(
        input_dir=input_dir,
        output_dir=output_dir,
        num_files=args.num_files,
        verbose=not args.quiet
    )
    
    # Preview if requested
    if args.preview and summary['successful'] > 0:
        preview_cleaned_data(output_dir)
    
    # Return exit code based on success
    if summary['failed'] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()