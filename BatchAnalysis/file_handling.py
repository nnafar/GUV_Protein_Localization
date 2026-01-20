# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:28:03 2026

@author: nnafar
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
            try:
                df = pd.read_csv(file_path)
                
                # Ensure 'Date' column exists (used as Batch ID)
                if 'Date' in df.columns:
                    all_dfs.append(df)
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not all_dfs:
        print("No data found!")
        return None

    # 1. Combine Data
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # 2. FORCE NUMERIC CONVERSION (Crucial Step)
    # This turns "margins", "error", or text into NaN (Not a Number)
    # causing them to be ignored by plots instead of crashing them.
    numeric_cols = [
        'Radius', 'M Background', 'A Background', 'A localization', 
        't_cortex', 'rho_actin', 'uniformity', 'Refined Radius (um)'
    ]
    
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
    # 3. Filter Invalid Data
    # Remove rows where Refined Radius is NaN (Critical failure)
    # We keep rows with 0 localization (Empty vesicles) as they are now valid numbers (0.0)
    master_df = master_df.dropna(subset=['Radius'])
    
    # 4. Assign Categories
    master_df['Category'] = 'Other'
    for cat in categories:
        # Case-insensitive match for category keywords in the Date/Folder name
        master_df.loc[master_df['Date'].str.contains(cat, case=False, na=False), 'Category'] = cat

    print(f"Successfully loaded {len(master_df)} vesicles across {master_df['Date'].nunique()} batches.")
    return master_df

def create_output_folder(root_path, folder_name="Results"):
    """Creates the main results directory if it doesn't exist."""
    path = os.path.join(root_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path