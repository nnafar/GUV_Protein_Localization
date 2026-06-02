# -*- coding: utf-8 -*-
"""
FILE HANDLING MODULE
Loads and processes Analysis_Results.csv files from multiple batches.

CHANGES FROM PREVIOUS VERSION:
-------------------------------
1. Category detection now reads from the FOLDER NAME, not the 'Date' column.
   Previously, the code searched for 'BranchedCortex' inside the 'Date' column
   (which contains a date like '260208'), so it never matched anything.

   Now it works like this:
     Path:   .../Output/260208_BranchedCortex_1/Region0000/Analysis_Results.csv
     Folder: 260208_BranchedCortex_1   ← we look for the condition name HERE
     Result: Category = 'BranchedCortex'

2. A 'Batch_ID' column is added so you can trace each row back to its
   original experiment folder (e.g., '260208_BranchedCortex_1').
   This is useful for batch-to-batch comparisons later.
"""

import os
import pandas as pd
import numpy as np


def load_and_process_data(root_path, categories):
    """
    Scans the root folder, finds every Analysis_Results.csv, loads them,
    assigns a Category based on the folder name, and returns one combined table.

    Think of this like going into a filing cabinet, reading every folder,
    and putting all the papers into one big pile — but also labelling
    each paper with which folder it came from.

    Parameters
    ----------
    root_path  : str  — path to the top-level Output folder
    categories : list — condition names to look for in folder names
                        e.g. ['BranchedCortex', 'LinearCortex', 'Factin', 'Empty']

    Returns
    -------
    master_df : pandas DataFrame, or None if no files were found
    """
    all_dfs = []
    print(f"Scanning directory: {root_path}\n")

    # os.walk steps through every sub-folder inside root_path.
    # For each folder it gives us:
    #   dirpath   = full path of the current folder
    #   dirnames  = names of sub-folders inside it (we don't need this)
    #   filenames = names of files inside it
    for dirpath, dirnames, filenames in os.walk(root_path):

        if "Analysis_Results.csv" not in filenames:
            continue  # Skip folders that don't have a results file

        file_path = os.path.join(dirpath, "Analysis_Results.csv")

        # ---- Determine the experiment folder name ----
        # Example dirpath: .../Output/260208_BranchedCortex_1/Region0000
        # os.path.dirname removes the last component → .../Output/260208_BranchedCortex_1
        # os.path.basename takes only the last component → 260208_BranchedCortex_1
        experiment_folder = os.path.basename(os.path.dirname(dirpath))

        # ---- Determine which condition this folder belongs to ----
        # We check whether any of our condition names appears inside the folder name.
        # str.lower() makes the comparison case-insensitive.
        assigned_category = 'Other'  # default if nothing matches
        for cat in categories:
            if cat.lower() in experiment_folder.lower():
                assigned_category = cat
                break  # stop once we find a match

        # ---- Load the CSV file ----
        # We try the standard way first, then a fallback if the file is malformed.
        df = None

        try:
            df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)

        except Exception:
            try:
                df = pd.read_csv(
                    file_path,
                    quotechar='"',
                    skipinitialspace=True,
                    on_bad_lines='skip',   # skip rows that can't be parsed
                    engine='python'
                )
                print(f"  ⚠ Loaded with skipped rows: {file_path}")
            except Exception as e:
                print(f"  ✗ Could not read: {file_path}  ({e})")
                continue  # skip this file entirely

        # Check that the file has the columns we expect
        if df is None or 'Date' not in df.columns:
            print(f"  ✗ Skipped (missing 'Date' column): {file_path}")
            continue

        # ---- Tag each row with its origin ----
        # This lets you trace any row back to its experiment and region later.
        df['Category'] = assigned_category
        df['Batch_ID'] = experiment_folder           # e.g. '260208_BranchedCortex_1'
        df['Region_ID'] = os.path.basename(dirpath)  # e.g. 'Region0000'

        all_dfs.append(df)
        print(f"  ✓ Loaded [{assigned_category:>15}]  {experiment_folder} / {os.path.basename(dirpath)}"
              f"  ({len(df)} rows)")

    # ---- Nothing found ----
    if not all_dfs:
        print("\nNo Analysis_Results.csv files found. Check ROOT_PATH.")
        return None

    # ---- Combine all tables into one ----
    # pd.concat stacks the tables vertically (one on top of the other).
    # Columns that exist in one condition but not another (e.g. actin columns
    # in Empty vesicles) will be filled with NaN automatically.
    master_df = pd.concat(all_dfs, ignore_index=True)

    # ---- Convert columns to numbers ----
    # When pandas reads a CSV, some columns might be stored as text even though
    # they contain numbers. errors='coerce' turns anything that isn't a number
    # into NaN instead of crashing.
    numeric_cols = [
        'Radius', 'M Background', 'A Background', 'A localization',
        'A Lumen', 'A Lumen/Bg', 't_cortex', 'ISM',
        'Gini_Index', 'Refined Radius (um)',
        'Deformability_Score', 'Radial_Bumpiness', 'Sector_Uniformity', 'Clustering_Risk',
    ]
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

    # ---- Filter 1: Raw Radius validity (still drop these -- no data exists) ----
    # A Radius of 0 or NaN means Hough detection failed and there is no
    # meaningful row to keep. These are genuinely missing observations.
    initial_count = len(master_df)
    master_df = master_df.dropna(subset=['Radius'])
    master_df = master_df[master_df['Radius'] > 0]
    removed_raw = initial_count - len(master_df)
    if removed_raw > 0:
        print(f"\n  → Removed {removed_raw} rows with invalid raw Radius (0 or NaN)")
    

    # ---- Filter 2: Minimum physical size (Refined Radius) ----
    # The refined radius comes from membrane peak detection in skeleton.py
    # and is a more accurate measure of the true vesicle size than the
    # raw Hough radius.
    #
    # Two categories of rows are removed here:
    #   a) Refined Radius = 0 or NaN — membrane detection failed entirely
    #      (e.g. vesicle was at image margin, or no membrane peak was found).
    #      These rows were written with a placeholder of 0 by skeleton.py.
    #   b) Refined Radius < MIN_VESICLE_RADIUS_UM — genuine debris or
    #      sub-resolution objects too small to analyse meaningfully.
    #
    # *** Keep MIN_VESICLE_RADIUS_UM in sync with
    #     'min_vesicle_radius_um' in main.py's ANALYSIS_CONFIG ***
    MIN_VESICLE_RADIUS_UM = 2.50   # µm

    # Boolean flag: True for vesicles that should be analysed for cortex/actin.
    master_df['Shape_Quality_Flag'] = (
        master_df['Refined Radius (um)'].notna() &
        (master_df['Refined Radius (um)'] >= MIN_VESICLE_RADIUS_UM)
    )
    
    n_excluded = (~master_df['Shape_Quality_Flag']).sum()
    if n_excluded > 0:
        print(f"  → Flagged {n_excluded} vesicles as size-EXCLUDED "
              f"(refined radius NaN or < {MIN_VESICLE_RADIUS_UM} µm)")

    # ---- Summary ----
    print(f"\n  ✓ Total vesicles loaded: {len(master_df)}")
    print(f"  ✓ Across {master_df['Batch_ID'].nunique()} experiment batches")
    print(f"  ✓ Category breakdown:")
    for cat, count in master_df['Category'].value_counts().items():
        print(f"      {cat:>15} : {count} vesicles")

    return master_df


def create_output_folder(root_path, folder_name):
    """
    Creates an output directory if it doesn't already exist.
    Returns the full path to that directory.
    """
    path = os.path.join(root_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path