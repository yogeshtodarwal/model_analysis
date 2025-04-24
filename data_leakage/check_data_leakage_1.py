# File: check_data_leakage.py
import argparse
import os
import sys
import pandas as pd
import numpy as np # Good practice, might be needed indirectly
from collections.abc import MutableMapping # For flatten_dict type checking

# --- Import required functions from your utility package ---
# Ensure utils_tbox is installed in your environment
try:
    from utils_tbox.utils_tbox import read_pklz, decompress_obj
except ImportError:
    print("Error: Could not import 'read_pklz' and 'decompress_obj' from 'utils_tbox.utils_tbox'.", file=sys.stderr)
    print("Please ensure the 'utils_tbox' package is installed correctly in your Python environment.", file=sys.stderr)
    print("You might need to run 'pip install -r requirements.txt' or install the utils packages manually.", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# Helper Function for Configuration Flattening
# ==============================================================================

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    """
    Flattens a nested dictionary.

    Args:
        d (MutableMapping): Dictionary to flatten.
        parent_key (str): The prefix for keys in the current level.
        sep (str): Separator to use between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        # Convert lists/tuples to string representation for CSV compatibility
        elif isinstance(v, (list, tuple)):
             items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

# ==============================================================================
# Modified Main Leakage Check Logic (operates on a single file)
# ==============================================================================

def check_leakage_in_file(pklz_filepath):
    """
    Loads a single PKLZ file, checks for leakage, and extracts data for CSV reports.

    Args:
        pklz_filepath (str): Path to the specific .pklz file.

    Returns:
        tuple: (
            bool: file_has_leakage,
            list: file_leakage_summary_records (for master leakage summary),
            list: file_leaked_ids_records (for detailed leaked IDs),
            dict: file_config_record (for master config summary)
        )
        Returns (False, [], [], {}) on file reading errors.
    """
    print(f"\n--- Checking File: {pklz_filepath} ---")
    # Initialize return structures for this file
    file_leakage_summary_records = []
    file_leaked_ids_records = []
    file_config_record = {'filename': os.path.basename(pklz_filepath)} # Start with filename
    file_has_leakage = False

    try:
        data = read_pklz(pklz_filepath) # Use imported function
    except FileNotFoundError:
        print(f"Error: File not found at {pklz_filepath}", file=sys.stderr)
        return False, [], [], {} # Indicate no leakage, return empty data
    except Exception as e:
        print(f"Error reading file {pklz_filepath}: {e}", file=sys.stderr)
        return False, [], [], {} # Indicate no leakage, return empty data

    # --- Extract and Flatten Configuration ---
    if isinstance(data, dict) and 'cfg' in data:
        if isinstance(data['cfg'], dict):
            try:
                flat_config = flatten_dict(data['cfg'])
                file_config_record.update(flat_config)
            except Exception as e:
                print(f"  Warning: Could not flatten 'cfg' dictionary in {pklz_filepath}: {e}", file=sys.stderr)
        else:
            print(f"  Warning: 'cfg' key in {pklz_filepath} is not a dictionary (type: {type(data['cfg'])}).", file=sys.stderr)
    else:
         print(f"  Warning: Could not find 'cfg' key in the loaded data dictionary for {pklz_filepath}.", file=sys.stderr)


    # --- Validate Data Structure for Leakage Check ---
    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary at the top level in {pklz_filepath}, but got {type(data)}", file=sys.stderr)
        return False, [], [], file_config_record # Return collected config, but indicate no leakage check possible
    if "results" not in data:
        print(f"Error: File {pklz_filepath} does not contain the required 'results' key.", file=sys.stderr)
        return False, [], [], file_config_record
    if not isinstance(data["results"], list):
         print(f"Error: Expected 'results' key in {pklz_filepath} to contain a list, but got {type(data['results'])}", file=sys.stderr)
         return False, [], [], file_config_record
    # --- End Validation ---

    compressed_datasets = data["results"]

    if not compressed_datasets:
         print(f"  Info: The 'results' list in {pklz_filepath} is empty. No folds to check for leakage.")
         # Add a record indicating no folds were checked for this file
         file_leakage_summary_records.append({
             'filename': os.path.basename(pklz_filepath),
             'fold_number': 'N/A',
             'leakage_detected': False,
             'leaked_id_count': 0
         })
         return False, file_leakage_summary_records, [], file_config_record

    total_folds = len(compressed_datasets)
    print(f"  Found {total_folds} fold(s) in 'results'.")

    for i, compressed_fold_data in enumerate(compressed_datasets):
        fold_num = i + 1
        fold_leakage_detected = False
        leaked_count_for_fold = 0
        leaked_ids_for_fold = []

        try:
            fold_data = decompress_obj(compressed_fold_data) # Use imported function

            # --- Basic Fold Data Validation ---
            if fold_data is None:
                print(f"  Error: Failed to decompress data for Fold {fold_num} in {pklz_filepath}. Skipping.", file=sys.stderr)
                continue
            if not isinstance(fold_data, dict) or "train" not in fold_data or "val" not in fold_data:
                 print(f"  Error: Fold {fold_num} data structure invalid in {pklz_filepath}. Skipping.", file=sys.stderr)
                 continue

            train_df = fold_data["train"]
            val_df = fold_data["val"]

            # --- DataFrame Validation ---
            if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
                 print(f"  Error: Fold {fold_num} 'train' or 'val' in {pklz_filepath} is not a DataFrame. Skipping.", file=sys.stderr)
                 continue
            if "ids__uid" not in train_df.columns or "ids__uid" not in val_df.columns:
                 print(f"  Error: Fold {fold_num} DataFrame missing 'ids__uid' in {pklz_filepath}. Skipping.", file=sys.stderr)
                 continue

            # --- Perform Leakage Check ---
            train_ids = set(train_df['ids__uid'].unique()) if not train_df.empty else set()
            val_ids = set(val_df['ids__uid'].unique()) if not val_df.empty else set()
            intersection = train_ids.intersection(val_ids)

            if intersection:
                fold_leakage_detected = True
                file_has_leakage = True # Mark overall file leakage
                leak_list = sorted(list(intersection))
                leaked_count_for_fold = len(leak_list)
                leaked_ids_for_fold = leak_list # Store the list for the detailed CSV

                print(f"  Leakage DETECTED in Fold {fold_num}! ({leaked_count_for_fold} IDs)")
                # Optionally print sample IDs if needed (can be verbose for many files)
                # if leaked_count_for_fold > 5:
                #     print(f"    Sample Leaked IDs: {leak_list[:5]}...")
                # else:
                #     print(f"    Leaked IDs: {leak_list}")

            # --- Record Results for this Fold ---
            file_leakage_summary_records.append({
                'filename': os.path.basename(pklz_filepath),
                'fold_number': fold_num,
                'leakage_detected': fold_leakage_detected,
                'leaked_id_count': leaked_count_for_fold
            })

            # If leakage, add records to the detailed list (one row per ID)
            if fold_leakage_detected:
                for leaked_id in leaked_ids_for_fold:
                    file_leaked_ids_records.append({
                        'filename': os.path.basename(pklz_filepath),
                        'fold_number': fold_num,
                        'leaked_ids__uid': leaked_id
                    })

        except Exception as e:
             print(f"  An unexpected error occurred processing Fold {fold_num} of {pklz_filepath}: {e}. Skipping fold.", file=sys.stderr)
             # Add a record indicating error for this fold? Or just skip. Let's skip for now.

    if not file_has_leakage:
        print(f"  SUCCESS: No leakage detected in any fold within {pklz_filepath}.")
    # else: # Summary is implicitly generated by the per-fold messages
    #     print(f"  SUMMARY: Leakage detected in {sum(r['leakage_detected'] for r in file_leakage_summary_records)}/{total_folds} fold(s) for {pklz_filepath}.")

    return file_has_leakage, file_leakage_summary_records, file_leaked_ids_records, file_config_record

# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for patient ID leakage in PKLZ files and generate CSV reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pklz_files",
        type=str,
        nargs='+', # Accept one or more file paths
        help="Path(s) to the .pklz file(s). Shell globbing (e.g., results/*.pklz) can be used."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="master_leakage_summary.csv",
        help="Output filename for the master leakage summary CSV."
    )
    parser.add_argument(
        "--config-csv",
        type=str,
        default="master_config_summary.csv",
        help="Output filename for the master configuration summary CSV."
    )
    parser.add_argument(
        "--leaked-ids-csv",
        type=str,
        default="detailed_leaked_ids.csv",
        help="Output filename for the detailed list of leaked patient IDs."
    )

    args = parser.parse_args()

    # --- Initialize Data Accumulators ---
    all_leakage_summary_data = []
    all_config_summary_data = []
    all_leaked_ids_data = []
    overall_leakage_found = False
    files_processed = 0
    files_with_leakage = 0

    print(f"Checking {len(args.pklz_files)} file path(s)...")

    # --- Process Each File ---
    for filepath in args.pklz_files:
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            print(f"\nWarning: Skipping invalid path: {filepath}", file=sys.stderr)
            continue

        files_processed += 1
        try:
            # Process the file and get data for reports
            leakage_found_in_file, \
            file_summary, \
            file_leaked_ids, \
            file_config = check_leakage_in_file(filepath)

            # Aggregate data
            all_leakage_summary_data.extend(file_summary)
            all_leaked_ids_data.extend(file_leaked_ids)
            if file_config: # Add config only if successfully extracted
                 all_config_summary_data.append(file_config)

            if leakage_found_in_file:
                overall_leakage_found = True
                files_with_leakage += 1

        except Exception as e:
            print(f"\nCritical Error processing file {filepath}: {e}", file=sys.stderr)
            print("  Attempting to continue with other files...", file=sys.stderr)
            # Optionally add an error record to the summary CSV here

    # --- Write CSV Files ---
    print("\n--- Writing CSV Reports ---")

    # Master Leakage Summary
    if all_leakage_summary_data:
        try:
            summary_df = pd.DataFrame.from_records(all_leakage_summary_data)
            # Reorder columns for clarity
            summary_cols = ['filename', 'fold_number', 'leakage_detected', 'leaked_id_count']
            summary_df = summary_df[summary_cols]
            summary_df.to_csv(args.summary_csv, index=False)
            print(f"  Master leakage summary saved to: {args.summary_csv}")
        except Exception as e:
            print(f"  Error writing {args.summary_csv}: {e}", file=sys.stderr)
    else:
        print(f"  No data to write for {args.summary_csv}.")

    # Master Configuration Summary
    if all_config_summary_data:
         try:
            config_df = pd.DataFrame.from_records(all_config_summary_data)
            # Ensure 'filename' is the first column
            cols = config_df.columns.tolist()
            if 'filename' in cols:
                cols.insert(0, cols.pop(cols.index('filename')))
                config_df = config_df[cols]
            config_df.to_csv(args.config_csv, index=False)
            print(f"  Master configuration summary saved to: {args.config_csv}")
         except Exception as e:
             print(f"  Error writing {args.config_csv}: {e}", file=sys.stderr)
    else:
         print(f"  No data to write for {args.config_csv}.")

    # Detailed Leaked IDs
    if all_leaked_ids_data:
         try:
            leaked_ids_df = pd.DataFrame.from_records(all_leaked_ids_data)
            leaked_ids_df.to_csv(args.leaked_ids_csv, index=False)
            print(f"  Detailed leaked IDs saved to: {args.leaked_ids_csv}")
         except Exception as e:
             print(f"  Error writing {args.leaked_ids_csv}: {e}", file=sys.stderr)
    else:
         print(f"  No leaked IDs found, {args.leaked_ids_csv} not created.")


    # --- Final Overall Summary ---
    print("\n\n===== Overall Leakage Check Summary =====")
    print(f"Files provided: {len(args.pklz_files)}")
    print(f"Files processed: {files_processed}")
    print(f"Files with leakage detected: {files_with_leakage}")

    if not overall_leakage_found:
        if files_processed > 0:
            print("\nSUCCESS: No patient ID leakage detected in any of the processed files.")
            sys.exit(0)
        else:
            print("\nWarning: No files were processed.")
            sys.exit(2)
    else:
        print("\nERROR: Patient ID leakage DETECTED in one or more files.")
        print(f"Please review the generated CSV files for details:")
        if all_leakage_summary_data: print(f"  - Summary: {args.summary_csv}")
        if all_config_summary_data: print(f"  - Configs: {args.config_csv}")
        if all_leaked_ids_data: print(f"  - Leaked IDs: {args.leaked_ids_csv}")
        sys.exit(1)
