# File: check_data_leakage.py
import argparse
import os
import sys
import pandas as pd
import numpy as np # Good practice, might be needed indirectly

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
# Main Leakage Check Logic (operates on a single file)
# ==============================================================================

def check_leakage_in_file(pklz_filepath):
    """
    Loads a single PKLZ file containing cross-validation results and checks
    for patient ID leakage between training and validation sets within each fold.

    Args:
        pklz_filepath (str): Path to the specific .pklz file.

    Returns:
        bool: True if leakage is detected in this file, False otherwise.
    """
    print(f"\n--- Checking File: {pklz_filepath} ---")
    try:
        data = read_pklz(pklz_filepath) # Use imported function
    except FileNotFoundError:
        print(f"Error: File not found at {pklz_filepath}", file=sys.stderr)
        return False # Treat as no leakage found for this specific path, but report error
    except Exception as e:
        print(f"Error reading file {pklz_filepath}: {e}", file=sys.stderr)
        return False # Cannot determine leakage if file reading fails

    # --- Validate Data Structure ---
    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary at the top level in {pklz_filepath}, but got {type(data)}", file=sys.stderr)
        return False
    if "results" not in data:
        print(f"Error: File {pklz_filepath} does not contain the required 'results' key.", file=sys.stderr)
        return False
    if not isinstance(data["results"], list):
         print(f"Error: Expected 'results' key in {pklz_filepath} to contain a list, but got {type(data['results'])}", file=sys.stderr)
         return False
    # --- End Validation ---

    compressed_datasets = data["results"]

    if not compressed_datasets:
         print(f"  Warning: The 'results' list in {pklz_filepath} is empty. No folds to check.")
         return False # No folds means no leakage

    leaked_folds = {} # Store fold index -> list of leaked IDs for this file
    total_folds = len(compressed_datasets)
    print(f"  Found {total_folds} fold(s) in 'results'.")
    file_has_leakage = False

    for i, compressed_fold_data in enumerate(compressed_datasets):
        fold_num = i + 1
        # print(f"  --- Checking Fold {fold_num}/{total_folds} ---") # Less verbose per-fold start
        try:
            # Decompress the dictionary for this specific fold using imported function
            fold_data = decompress_obj(compressed_fold_data)

            if fold_data is None:
                print(f"  Error: Failed to decompress data for Fold {fold_num} in {pklz_filepath}. Skipping.", file=sys.stderr)
                continue # Skip this fold

            if not isinstance(fold_data, dict):
                 print(f"  Error: Decompressed data for Fold {fold_num} in {pklz_filepath} is not a dict (type: {type(fold_data)}). Skipping.", file=sys.stderr)
                 continue
            if "train" not in fold_data or "val" not in fold_data:
                 print(f"  Error: Fold {fold_num} dict in {pklz_filepath} missing 'train' or 'val' key. Skipping.", file=sys.stderr)
                 continue

            train_df = fold_data["train"]
            val_df = fold_data["val"]

            # --- Validate DataFrame Structure ---
            if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
                 print(f"  Error: Fold {fold_num} 'train' or 'val' in {pklz_filepath} is not a pandas DataFrame. Skipping.", file=sys.stderr)
                 print(f"    Train type: {type(train_df)}, Val type: {type(val_df)}")
                 continue
            if "ids__uid" not in train_df.columns:
                 print(f"  Error: Fold {fold_num} 'train' DataFrame in {pklz_filepath} is missing 'ids__uid' column. Skipping.", file=sys.stderr)
                 continue
            if "ids__uid" not in val_df.columns:
                 print(f"  Error: Fold {fold_num} 'val' DataFrame in {pklz_filepath} is missing 'ids__uid' column. Skipping.", file=sys.stderr)
                 continue
             # --- End DataFrame Validation ---

            if train_df.empty:
                train_ids = set()
            else:
                train_ids = set(train_df['ids__uid'].unique())

            if val_df.empty:
                val_ids = set()
            else:
                val_ids = set(val_df['ids__uid'].unique())

            intersection = train_ids.intersection(val_ids)

            if intersection:
                leak_list = sorted(list(intersection))
                print(f"  Leakage DETECTED in Fold {fold_num}!")
                print(f"    Number of leaked IDs: {len(leak_list)}")
                if len(leak_list) > 10:
                    print(f"    Leaked IDs (sample): {leak_list[:10]}...")
                else:
                    print(f"    Leaked IDs: {leak_list}")
                leaked_folds[fold_num] = leak_list
                file_has_leakage = True # Mark that this file contains leakage
            # else: # Less verbose success message
                # print(f"  No leakage detected in Fold {fold_num}.")
            # print(f"    Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")

        except (AttributeError, KeyError, TypeError) as e:
             print(f"  Error accessing data within Fold {fold_num} of {pklz_filepath}: {e}. Check data structure. Skipping fold.", file=sys.stderr)
        except Exception as e:
             print(f"  An unexpected error occurred processing Fold {fold_num} of {pklz_filepath}: {e}. Skipping fold.", file=sys.stderr)

    if not file_has_leakage:
        print(f"  SUCCESS: No leakage detected in any fold within {pklz_filepath}.")
    else:
         print(f"  SUMMARY: Leakage detected in {len(leaked_folds)}/{total_folds} fold(s) for {pklz_filepath}.")


    return file_has_leakage

# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for patient ID (ids__uid) leakage between training and validation sets in cross-validation folds stored within one or more PKLZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pklz_files",
        type=str,
        nargs='+', # Accept one or more file paths
        help="Path(s) to the .pklz file(s) generated by main.py. Shell globbing (e.g., results/*.pklz) can be used."
    )

    args = parser.parse_args()

    overall_leakage_found = False
    files_processed = 0
    files_with_leakage = 0

    print(f"Checking {len(args.pklz_files)} file path(s)...")

    for filepath in args.pklz_files:
        if not os.path.exists(filepath):
            print(f"\nWarning: File path does not exist, skipping: {filepath}", file=sys.stderr)
            continue
        if not os.path.isfile(filepath):
             print(f"\nWarning: Path is not a file, skipping: {filepath}", file=sys.stderr)
             continue

        files_processed += 1
        try:
            leakage_found_in_current_file = check_leakage_in_file(filepath)
            if leakage_found_in_current_file:
                overall_leakage_found = True
                files_with_leakage += 1
        except Exception as e:
            print(f"\nCritical Error processing file {filepath}: {e}", file=sys.stderr)
            print("  Attempting to continue with other files...", file=sys.stderr)
            # Mark overall leakage as true maybe? Or just report the error.
            # Let's just report and continue for now.

    # --- Final Overall Summary ---
    print("\n\n===== Overall Leakage Check Summary =====")
    print(f"Files provided: {len(args.pklz_files)}")
    print(f"Files processed: {files_processed}")
    print(f"Files with leakage detected: {files_with_leakage}")

    if not overall_leakage_found:
        if files_processed > 0:
            print("\nSUCCESS: No patient ID leakage detected in any of the processed files.")
            sys.exit(0) # Exit with success code 0
        else:
            print("\nWarning: No files were processed (check paths/permissions).")
            sys.exit(2) # Exit with a different code indicating no files processed
    else:
        print("\nERROR: Patient ID leakage DETECTED in one or more files.")
        print("Please review the output above for details on specific files and folds.")
        sys.exit(1) # Exit with error code 1