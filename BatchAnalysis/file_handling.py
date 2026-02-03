# -*- coding: utf-8 -*-
"""
FILE HANDLING MODULE
Loads and processes Analysis_Results.csv files from multiple batches.
"""

import os
import pandas as pd
import numpy as np

def load_and_process_data(root_path, categories):
    """
    Recursively finds all Analysis_Results.csv files, combines them,
    assigns categories based on folder names, and filters invalid data.
    """
    all_dfs = []
    print(f"Scanning directory: {root_path}...\n")

    for dirpath, dirnames, filenames in os.walk(root_path):
        if "Analysis_Results.csv" in filenames:
            file_path = os.path.join(dirpath, "Analysis_Results.csv")
            
            # Attempt 1: Strict Parsing (Standard)
            try:
                df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)
                
            # Attempt 2: Fallback for "Expected 18 fields, saw 19" errors
            except Exception:
                try:
                    # 'on_bad_lines' requires pandas >= 1.3
                    # We use engine='python' for more robust parsing of complex quotes
                    df = pd.read_csv(
                        file_path, 
                        quotechar='"', 
                        skipinitialspace=True, 
                        on_bad_lines='skip', 
                        engine='python'
                    )
                    print(f"  ⚠ Loaded (with skips): {file_path} (Skipped malformed rows)")
                except Exception as e:
                    print(f"  ✗ Error reading {file_path}: {e}")
                    continue

            # Process the successfully loaded dataframe
            if 'Date' in df.columns:
                all_dfs.append(df)
                # Only print standard success if we didn't already print the warning
                if "Skipped malformed rows" not in str(df): 
                     # Note: logic above prints warning inside the except block, 
                     # so strictly speaking we can just print the count here.
                     pass
            else:
                print(f"  ✗ Skipped {file_path}: Missing 'Date' column")

    if not all_dfs:
        print("No data found!")
        return None

    # 1. Combine Data
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # 2. Type Conversion
    # Ensure numeric columns are actually numeric
    numeric_cols = [
        'Radius', 'M Background', 'A Background', 'A localization', 
        'A Lumen', 'A Lumen/Bg', 't_cortex', 'ISM', 
        'Gini_Index', 'Radial_Kurtosis', 'Refined Radius (um)'
    ]
    
    for col in numeric_cols:
        if col in master_df.columns:
            # errors='coerce' turns non-numeric strings into NaN
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
    # 3. Filter Invalid Data
    # Remove entries with Radius 0.0 or NaN (failed fits)
    initial_count = len(master_df)
    master_df = master_df.dropna(subset=['Radius'])
    master_df = master_df[master_df['Radius'] > 0]
    removed = initial_count - len(master_df)
    
    if removed > 0:
        print(f"\n  → Removed {removed} vesicles with invalid Radius (0 or NaN)")
    
    # 4. Assign Categories
    master_df['Category'] = 'Other'
    for cat in categories:
        # Case-insensitive match for category keywords in the Date/Folder name
        master_df.loc[master_df['Date'].str.contains(cat, case=False, na=False), 'Category'] = cat

    print(f"\n Successfully loaded {len(master_df)} vesicles across {master_df['Date'].nunique()} batches.")
    
    # Debug: Print category distribution
    print(f" Categories: {master_df['Category'].value_counts().to_dict()}")
    
    return master_df

def create_output_folder(root_path, folder_name):
    """Creates a timestamped output directory."""
    path = os.path.join(root_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path