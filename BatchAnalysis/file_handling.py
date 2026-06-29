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


def _scan_and_tag_csv_files(root_path, categories, filename, required_column):
    """
    Walks every sub-folder under `root_path`, finds every file named
    `filename`, loads it, and tags each row with which experiment it came
    from (Category / Batch_ID / Region_ID) based on the FOLDER NAME.

    THE ANALOGY: this is like going through a filing cabinet, opening
    every drawer, and for every paper of the right type, stapling a sticky
    note to it that says which drawer (folder) it came from — before
    putting all the papers into one big combined pile.

    This is the shared "folder walking + tagging" logic originally used
    only for Analysis_Results.csv. It is now factored out into its own
    function so that BOTH Analysis_Results.csv (one row per vesicle) and
    Radial_Intensity_Profiles.csv (one row per radius point per vesicle)
    get tagged the exact same way — which is essential, since that's
    what lets us join the two files together later using
    Category + Batch_ID + Region_ID + Vesicle id.

    Parameters
    ----------
    root_path        : str  — path to the top-level Output folder
    categories       : list — condition names to look for in folder names
    filename          : str  — exact filename to look for in each folder,
                        e.g. "Analysis_Results.csv" or
                        "Radial_Intensity_Profiles.csv"
    required_column   : str  — a column name that MUST be present for a
                        loaded file to be kept (used as a quick sanity
                        check that the file isn't empty/corrupted)

    Returns
    -------
    list of pandas DataFrames (NOT yet concatenated), each one already
    tagged with 'Category', 'Batch_ID', 'Region_ID'. Returns an empty
    list if nothing was found.
    """
    all_dfs = []

    for dirpath, dirnames, filenames in os.walk(root_path):

        if filename not in filenames:
            continue  # Skip folders that don't have this particular file

        file_path = os.path.join(dirpath, filename)

        # ---- Determine the experiment folder name ----
        # Example dirpath: .../Output/260208_BranchedCortex_1/Region0000
        experiment_folder = os.path.basename(os.path.dirname(dirpath))

        # ---- Determine which condition this folder belongs to ----
        assigned_category = 'Other'
        for cat in categories:
            if cat.lower() in experiment_folder.lower():
                assigned_category = cat
                break

        # ---- Load the CSV file ----
        df = None
        try:
            df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)
        except Exception:
            try:
                df = pd.read_csv(
                    file_path,
                    quotechar='"',
                    skipinitialspace=True,
                    on_bad_lines='skip',
                    engine='python'
                )
                print(f"  ⚠ Loaded with skipped rows: {file_path}")
            except Exception as e:
                print(f"  ✗ Could not read: {file_path}  ({e})")
                continue

        if df is None or required_column not in df.columns:
            print(f"  ✗ Skipped (missing '{required_column}' column): {file_path}")
            continue

        # ---- Tag each row with its origin ----
        df['Category']  = assigned_category
        df['Batch_ID']  = experiment_folder
        df['Region_ID'] = os.path.basename(dirpath)

        all_dfs.append(df)

    return all_dfs


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
    print(f"Scanning directory: {root_path}\n")

    all_dfs = _scan_and_tag_csv_files(
        root_path, categories, "Analysis_Results.csv", required_column="Date")

    for df in all_dfs:
        cat   = df['Category'].iloc[0]
        batch = df['Batch_ID'].iloc[0]
        region = df['Region_ID'].iloc[0]
        print(f"  ✓ Loaded [{cat:>15}]  {batch} / {region}  ({len(df)} rows)")

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


def load_radial_profiles(root_path, categories):
    """
    Scans the root folder, finds every Radial_Intensity_Profiles.csv
    (one per region, written by skeleton.create_radial_profile_csv via
    main.py), loads them, tags each row with Category/Batch_ID/Region_ID,
    and returns one big combined long-format table.

    THE ANALOGY: load_and_process_data() above collects every vesicle's
    "report card" (one row per vesicle). This function instead collects
    every vesicle's full "growth chart" (one row per radius measurement
    point) — there will be far more rows here than in the master dataset.

    The returned table has columns:
        Vesicle id, radius_um, Membrane_Intensity, Actin_Intensity,
        Category, Batch_ID, Region_ID

    To connect a row in THIS table back to that same vesicle's phenotype
    (Lumenal / Sparse / Continuous / ...), merge it with the master
    dataset on ['Category', 'Batch_ID', 'Region_ID', 'Vesicle id'] — see
    analysis_batch.compute_representative_radial_profiles for an example.

    Parameters
    ----------
    root_path  : str  — same top-level Output folder used in
                 load_and_process_data
    categories : list — condition names, e.g.
                 ['BranchedCortex', 'LinearCortex', 'Factin', 'Empty']

    Returns
    -------
    radial_df : pandas DataFrame, or None if no files were found
    """
    print(f"\nScanning directory for radial profiles: {root_path}\n")

    all_dfs = _scan_and_tag_csv_files(
        root_path, categories, "Radial_Intensity_Profiles.csv",
        required_column="Vesicle id")

    if not all_dfs:
        print("  No Radial_Intensity_Profiles.csv files found.\n"
              "  (Did you re-run main.py with the updated skeleton.py?)")
        return None

    radial_df = pd.concat(all_dfs, ignore_index=True)

    # Make sure the numeric columns are actually numeric (same defensive
    # pattern as load_and_process_data above — a CSV value can sometimes
    # get read in as text).
    for col in ['Vesicle id', 'radius_um', 'Membrane_Intensity', 'Actin_Intensity']:
        radial_df[col] = pd.to_numeric(radial_df[col], errors='coerce')

    print(f"  ✓ Loaded {len(radial_df)} radial-profile points "
          f"across {radial_df['Batch_ID'].nunique()} batches")

    return radial_df


def create_output_folder(root_path, folder_name):
    """
    Creates an output directory if it doesn't already exist.
    Returns the full path to that directory.
    """
    path = os.path.join(root_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path