# File: check_data_leakage.py
import argparse
import os
import sys
import pandas as pd
import numpy as np
from collections.abc import MutableMapping
import warnings
import re
from collections import defaultdict # For grouping predictions

# --- Imports (Keep as before) ---
try:
    from utils_tbox.utils_tbox import read_pklz, decompress_obj
except ImportError:
    print("Error: Could not import 'read_pklz' and 'decompress_obj' from 'utils_tbox.utils_tbox'.", file=sys.stderr)
    sys.exit(1)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found. Metrics will not be calculated.", file=sys.stderr)
    SKLEARN_AVAILABLE = False

# --- flatten_dict (Keep as before) ---
def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    # ... (previous implementation) ...
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple, set)):
             items.append((new_key, str(v))) # Convert list/tuple/set to string
        else:
            items.append((new_key, v))
    return dict(items)


# --- calculate_metrics (Keep as before) ---
def calculate_metrics(y_true, y_pred, task_type='binary', target_name='unknown'):
    # ... (previous implementation) ...
    metrics = {'auroc': np.nan, 'ap': np.nan}
    if not SKLEARN_AVAILABLE: return metrics
    try:
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        valid_idx_true = ~np.isnan(y_true); valid_idx_pred = ~np.isnan(y_pred).any(axis=1)
        valid_idx = valid_idx_true & valid_idx_pred
        if not np.any(valid_idx):
             # print(f"      Warning [{target_name}]: No valid samples after filtering.", file=sys.stderr) # Less verbose
             return metrics
        y_true = y_true[valid_idx]; y_pred = y_pred[valid_idx, :]
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # print(f"      Warning [{target_name}]: Only one class present ({unique_classes}) after filtering.", file=sys.stderr) # Less verbose
            return metrics
    except Exception as e: print(f"      Error preparing data [{target_name}]: {e}", file=sys.stderr); return metrics
    # AUROC calculation
    try:
        if task_type == 'binary':
            if y_pred.shape[1] != 1: print(f"      Warning [{target_name}]: Binary task expected 1 pred col for AUROC, got {y_pred.shape[1]}. Using first.", file=sys.stderr)
            metrics['auroc'] = roc_auc_score(y_true, y_pred[:, 0])
        elif task_type == 'multiclass':
            if y_pred.shape[1] < 2 : raise ValueError(f"Multiclass expects >= 2 pred cols, got {y_pred.shape[1]}")
            metrics['auroc'] = roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')
        else: print(f"      Warning [{target_name}]: Unknown task_type '{task_type}' for AUROC.", file=sys.stderr)
    except ValueError as ve: print(f"      Warning [{target_name}]: AUROC calc failed: {ve}", file=sys.stderr)
    except Exception as e: print(f"      Error calc AUROC [{target_name}]: {e}", file=sys.stderr)
    # AP calculation
    try:
        if task_type == 'binary':
            if y_pred.shape[1] != 1: print(f"      Warning [{target_name}]: Binary task expected 1 pred col for AP, got {y_pred.shape[1]}. Using first.", file=sys.stderr)
            metrics['ap'] = average_precision_score(y_true, y_pred[:, 0])
        elif task_type == 'multiclass': metrics['ap'] = np.nan
        else: print(f"      Warning [{target_name}]: Unknown task_type '{task_type}' for AP.", file=sys.stderr)
    except ValueError as ve: print(f"      Warning [{target_name}]: AP calc failed: {ve}", file=sys.stderr)
    except Exception as e: print(f"      Error calc AP [{target_name}]: {e}", file=sys.stderr)
    return metrics


# ==============================================================================
# Function to Parse Prediction Columns and Infer Task (REVISED VALIDATION)
# ==============================================================================
def parse_prediction_info(df_columns):
    """
    Analyzes prediction columns to infer model name, task structure, and target columns.
    Validation checks only for EXISTENCE of required target columns.
    """
    pred_col_pattern = re.compile(r"^pred__([^_].*?)__(.+)$")
    predictions = defaultdict(list)
    found_match = False
    for col in df_columns:
        match = pred_col_pattern.match(col)
        if match:
            found_match = True
            try:
                model_name, description = match.groups()
                predictions[model_name].append({'full_col': col, 'desc': description})
            except ValueError: pass # Ignore unpacking errors

    if not found_match:
        print("      DEBUG: No columns matched prediction regex pattern: ^pred__([^_].*?)__(.+)$")
        return None

    # Assume one model per fold (or pick one)
    if len(predictions) > 1:
         model_name = max(predictions, key=lambda k: len(predictions[k]))
         print(f"      Warning: Found predictions for multiple models: {list(predictions.keys())}. Processing '{model_name}'.", file=sys.stderr)
    else: model_name = list(predictions.keys())[0]
    model_predictions = predictions[model_name]

    # Sort by description for consistent class ordering
    model_predictions.sort(key=lambda p: p['desc'])
    descriptions = [p['desc'] for p in model_predictions]
    full_pred_cols = [p['full_col'] for p in model_predictions]

    task_info = { # Initialize structure
        'model_name': model_name,
        'pred_cols': full_pred_cols,
        'pred_descriptions': descriptions,
        'required_target_bases': [], # Base names like 'los' needed for y_true
        'task_type': None,
        'class_mapping': None # Only for multiclass: desc -> index
    }

    # --- Identify ACTUAL targets mentioned and analyze structure ---
    # Regex specifically for descriptions indicating a positive target outcome
    positive_target_pattern = re.compile(r"^target__(\w+)$")
    actual_target_bases = set() # Store base names like 'los', 'abdominal_nec'
    has_negation = False # Check for descriptions indicating a negative/baseline class

    for desc in descriptions:
        match = positive_target_pattern.match(desc)
        if match:
            actual_target_bases.add(match.group(1)) # Add 'los' from 'target__los'
        # Check for common negation patterns
        if desc.startswith('not_target') or desc.startswith('no_target') or desc.lower() == 'none' or desc.lower() == 'other' or '_neg' in desc:
            has_negation = True

    task_info['required_target_bases'] = sorted(list(actual_target_bases))
    num_descs = len(descriptions)
    num_actual_targets = len(actual_target_bases)

    # --- Inference Logic ---
    # Case 1: Single prediction column, assume binary score FOR the identified target
    if num_descs == 1 and num_actual_targets == 1:
        task_info['task_type'] = 'binary'
        print(f"      Inferred BINARY task (single pred col) for target '{task_info['required_target_bases'][0]}'")
    # Case 2: Binary with explicit positive/negative classes (e.g., target__A, not_target__A)
    elif num_descs == 2 and num_actual_targets == 1 and has_negation:
         task_info['task_type'] = 'binary'
         # Find the column corresponding to the positive description ('target__outcome')
         pos_desc = f"target__{task_info['required_target_bases'][0]}"
         pos_col = next((p['full_col'] for p in model_predictions if p['desc'] == pos_desc), None)
         if pos_col:
             task_info['pred_cols'] = [pos_col] # Use only positive class prob for binary metrics
             print(f"      Inferred BINARY task (explicit pos/neg) for target '{task_info['required_target_bases'][0]}'. Using positive pred: {pos_col}")
         else:
              print(f"      Warning: Could not find positive prediction column for inferred binary task. Desc='{pos_desc}'. Skipping.", file=sys.stderr)
              return None # Cannot proceed without positive prediction
    # Case 3: Multiclass inferred from multiple descriptions involving actual targets
    elif num_descs > 1 and num_actual_targets >= 1:
         # This covers the user case: target__A, target__B, not_target...
         task_info['task_type'] = 'multiclass'
         task_info['class_mapping'] = {desc: i for i, desc in enumerate(descriptions)} # Use sorted order
         print(f"      Inferred MULTICLASS task involving targets {task_info['required_target_bases']} from {num_descs} prediction columns.")
         # print(f"      Class mapping: {task_info['class_mapping']}") # Less verbose
    else:
         # If none of the above patterns match, we can't determine the task
         print(f"      Warning: Could not reliably infer task structure from descriptions: {descriptions}. Actual targets found: {actual_target_bases}. Skipping metrics.", file=sys.stderr)
         return None

    # --- **REVISED** Final Validation: Check ONLY for existence of required ACTUAL target columns ---
    missing_targets = []
    for target_base in task_info['required_target_bases']: # Use the refined list
        target_col_name = f"target__{target_base}"
        if target_col_name not in df_columns:
             missing_targets.append(target_col_name)

    if missing_targets:
         print(f"      Warning: The inferred task requires actual target column(s) missing from the DataFrame: {missing_targets}. Skipping metrics.", file=sys.stderr)
         return None # Return None if required truth columns are missing

    # If all checks pass, return the populated task_info dictionary
    print(f"      Task inference successful: {task_info['task_type']}") # Confirmation
    return task_info

# --- construct_multiclass_y_true (Keep as before, should work with correct task_info) ---
def construct_multiclass_y_true(df, task_info):
    """Constructs the integer-coded y_true array for a multiclass task."""
    if task_info['task_type'] != 'multiclass': return None
    n_samples = len(df)
    y_true = np.full(n_samples, -1, dtype=int)
    class_mapping = task_info['class_mapping']
    descriptions = task_info['pred_descriptions'] # Sorted
    neg_class_index = -1
    # Find negative class index
    for desc, index in class_mapping.items():
        if 'not_target' in desc or 'no_target' in desc or desc.lower() == 'none' or desc.lower() == 'other' or '_neg' in desc:
             if neg_class_index != -1: print(f"      Warning: Multiple potential negative classes found. Using first: index {neg_class_index}.", file=sys.stderr)
             else: neg_class_index = index; # print(f"      DEBUG: Identified potential negative class: '{desc}' (index {index})")
    # Assign positive classes first based on actual target columns
    positive_target_pattern = re.compile(r"^target__(\w+)$")
    assigned_positive = np.zeros(n_samples, dtype=bool) # Track if any positive class assigned
    # Iterate in reverse order of index? Or standard? Standard order should be fine.
    for desc, class_index in class_mapping.items():
        if class_index == neg_class_index: continue
        match = positive_target_pattern.match(desc) # Only map explicit 'target__NAME' descriptions
        if match:
            target_base = match.group(1)
            target_col = f"target__{target_base}"
            if target_col not in df.columns:
                print(f"      Error: construct_y_true needs '{target_col}' but it's missing.", file=sys.stderr); return None
            # Assign class where target is 1 AND no other positive class has been assigned yet
            # Assumes mutual exclusivity enforced by processing order or data reality
            assign_mask = (df[target_col] == 1) & (~assigned_positive) & (y_true == -1)
            y_true[assign_mask] = class_index
            assigned_positive[assign_mask] = True # Mark as assigned to a positive class
            # if np.any(assign_mask): print(f"      DEBUG: Assigned class {class_index} ('{desc}') to {np.sum(assign_mask)} samples.")
    # Assign negative class to remaining unassigned samples
    if neg_class_index != -1:
        remaining_mask = (y_true == -1)
        y_true[remaining_mask] = neg_class_index
        # print(f"      DEBUG: Assigned negative class {neg_class_index} to {np.sum(remaining_mask)} remaining samples.")
    elif np.any(y_true == -1): # No clear negative class, default to lowest index
        lowest_index = min(class_mapping.values())
        print(f"      Warning: Some samples unassigned, no clear negative class. Assigning lowest index {lowest_index}.", file=sys.stderr)
        y_true[y_true == -1] = lowest_index
    if np.any(y_true == -1): # Should not happen
        print(f"      Error: Failed to assign all samples a class in y_true construction.", file=sys.stderr); return None
    return y_true


# --- check_leakage_in_file (Minor logging change) ---
def check_leakage_in_file(pklz_filepath):
    """Loads PKLZ, checks leakage, infers task, calculates metrics, reports."""
    print(f"\n--- Checking File: {pklz_filepath} ---")
    # ... (Initializations as before) ...
    file_leakage_summary_records = []
    file_leaked_ids_records = []
    file_config_record = {'filename': os.path.basename(pklz_filepath)}
    file_has_leakage = False
    overall_cfg = {}
    fold_val_aurocs = []
    fold_val_aps = []
    metrics_task_description = "N/A"
    consistent_task_info_across_folds = None

    try: data = read_pklz(pklz_filepath)
    except FileNotFoundError: return False, [], [], {}
    except Exception as e: print(f"Error reading {pklz_filepath}: {e}", file=sys.stderr); return False, [], [], {}
    # Config extraction...
    if isinstance(data, dict) and 'cfg' in data and isinstance(data['cfg'], dict):
        overall_cfg = data['cfg']
        try: file_config_record.update(flatten_dict(overall_cfg))
        except Exception as e: print(f"  Warning: Could not flatten 'cfg': {e}", file=sys.stderr)
    else: print(f"  Warning: No valid 'cfg' found.", file=sys.stderr)
    # Results validation...
    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        print(f"Error: Invalid/missing 'results' list in {pklz_filepath}.", file=sys.stderr)
        # ... (return with error info) ...
        file_config_record.update({'metrics_task': 'Error: Invalid results', 'mean_val_auroc': np.nan, 'std_val_auroc': np.nan, 'mean_val_ap': np.nan, 'std_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': 0})
        return False, [], [], file_config_record

    compressed_datasets = data["results"]
    if not compressed_datasets:
        # ... (handle empty results) ...
        print(f"  Info: 'results' list is empty.")
        file_leakage_summary_records.append({'filename': os.path.basename(pklz_filepath),'fold_number': 'N/A','leakage_detected': False,'leaked_id_count': 0})
        file_config_record.update({'metrics_task': 'No folds', 'mean_val_auroc': np.nan, 'std_val_auroc': np.nan, 'mean_val_ap': np.nan, 'std_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': len(compressed_datasets)})
        return False, file_leakage_summary_records, [], file_config_record

    total_folds = len(compressed_datasets)
    print(f"  Found {total_folds} fold(s) in 'results'.")

    # --- Process Folds ---
    for i, compressed_fold_data in enumerate(compressed_datasets):
        fold_num = i + 1
        print(f"   --- Processing Fold {fold_num}/{total_folds} ---")
        fold_summary_record = {'filename': os.path.basename(pklz_filepath),'fold_number': fold_num,'leakage_detected': False,'leaked_id_count': 0}
        try:
            fold_data = decompress_obj(compressed_fold_data)
            if fold_data is None or not isinstance(fold_data, dict): # Error handling
                # ... (handle error, add to summary, continue) ...
                 print(f"    Error: Fold {fold_num} structure invalid/decompress failed.", file=sys.stderr)
                 fold_summary_record['error'] = 'Decompress/Structure Error'
                 file_leakage_summary_records.append(fold_summary_record)
                 continue

            val_df = fold_data.get("val"); train_df = fold_data.get("train")
            if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame): # Error handling
                # ... (handle error, add to summary, continue) ...
                 print(f"    Error: Fold {fold_num} train/val not DataFrame.", file=sys.stderr)
                 fold_summary_record['error'] = 'Invalid train/val type'
                 file_leakage_summary_records.append(fold_summary_record)
                 continue

            # --- Leakage Check ---
            fold_leakage_detected, leaked_count_for_fold = False, 0 # Reset for fold
            if "ids__uid" not in train_df.columns or "ids__uid" not in val_df.columns:
                 print(f"    Warning: Missing 'ids__uid'. Leakage check skipped.", file=sys.stderr)
            else:
                # ... (leakage check logic as before) ...
                train_ids = set(train_df['ids__uid'].dropna().unique())
                val_ids = set(val_df['ids__uid'].dropna().unique())
                intersection = train_ids.intersection(val_ids)
                if intersection:
                     fold_leakage_detected = True; file_has_leakage = True
                     leak_list = sorted(list(intersection)); leaked_count_for_fold = len(leak_list)
                     print(f"    Leakage DETECTED in Fold {fold_num}! ({leaked_count_for_fold} IDs)")
                     for leaked_id in leak_list: file_leaked_ids_records.append({'filename': os.path.basename(pklz_filepath),'fold_number': fold_num,'leaked_ids__uid': leaked_id})
            fold_summary_record['leakage_detected'] = fold_leakage_detected
            fold_summary_record['leaked_id_count'] = leaked_count_for_fold
            file_leakage_summary_records.append(fold_summary_record)

            # --- Performance Metric Calculation ---
            if not SKLEARN_AVAILABLE: continue
            if val_df.empty: print(f"    Warning: Validation DataFrame empty. Skipping metrics.", file=sys.stderr); continue

            # Infer task structure (THE KEY PART)
            task_info = parse_prediction_info(val_df.columns)

            # Check if parsing SUCCEEDED
            if task_info is None:
                 # *Now* this message should only appear if parse_prediction_info returned None
                 print(f"    Skipping metrics for Fold {fold_num} (task inference failed or required columns missing).")
                 continue # Skip metrics for this fold

            # Store consistent task info
            if consistent_task_info_across_folds is None:
                consistent_task_info_across_folds = task_info
                req_targets = '/'.join(task_info['required_target_bases']) if task_info['required_target_bases'] else 'Unknown'
                metrics_task_description = f"{task_info['task_type']} ({req_targets})"

            # Prepare y_true and y_pred
            y_true = None; y_pred_df = val_df[task_info['pred_cols']]
            metrics_target_name_label = metrics_task_description # Label for logging

            if task_info['task_type'] == 'binary':
                 target_base_name = task_info['required_target_bases'][0]
                 target_col = f"target__{target_base_name}"
                 y_true = val_df[target_col]
                 metrics_target_name_label = f"Binary ({target_base_name})"
            elif task_info['task_type'] == 'multiclass':
                 y_true = construct_multiclass_y_true(val_df, task_info)
                 if y_true is None: print(f"    Error: Failed construct multiclass y_true. Skip metrics Fold {fold_num}.", file=sys.stderr); continue
                 metrics_target_name_label = f"Multi ({'/'.join(task_info['required_target_bases'])})"

            # Calculate
            if y_true is not None:
                 fold_metrics = calculate_metrics(y_true, y_pred_df, task_info['task_type'], target_name=metrics_target_name_label)
                 print(f"      Fold {fold_num} Metrics [{metrics_target_name_label}]: AUROC={fold_metrics.get('auroc', np.nan):.4f}, AP={fold_metrics.get('ap', np.nan):.4f}")
                 if not np.isnan(fold_metrics.get('auroc', np.nan)): fold_val_aurocs.append(fold_metrics['auroc'])
                 if not np.isnan(fold_metrics.get('ap', np.nan)): fold_val_aps.append(fold_metrics['ap'])

        except Exception as e: # Catch unexpected errors during fold processing
             print(f"    Unexpected error processing Fold {fold_num}: {e}. Skipping fold.", file=sys.stderr)
             import traceback; traceback.print_exc(file=sys.stderr)
             # Update fold summary if possible
             if 'error' not in fold_summary_record: fold_summary_record['error'] = f"Unexpected: {e}"
             if fold_summary_record not in file_leakage_summary_records: file_leakage_summary_records.append(fold_summary_record)

    # --- Aggregate Metrics ---
    # ... (Aggregation logic as before) ...
    num_folds_metrics_calculated = len(fold_val_aurocs)
    mean_auroc, std_auroc, mean_ap, std_ap = np.nan, np.nan, np.nan, np.nan
    if num_folds_metrics_calculated > 0:
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning)
             mean_auroc = np.mean(fold_val_aurocs); std_auroc = np.std(fold_val_aurocs)
             mean_ap = np.mean(fold_val_aps); std_ap = np.std(fold_val_aps)
        print(f"  Aggregated Val Metrics [{metrics_task_description}] ({num_folds_metrics_calculated}/{total_folds} folds):")
        print(f"    AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
        print(f"    AP:    {mean_ap:.4f} +/- {std_ap:.4f}")
    else:
        if SKLEARN_AVAILABLE and total_folds > 0: print(f"  No valid metrics calculated across folds for the inferred task.")

    # ... (Update file_config_record as before) ...
    file_config_record.update({
        'metrics_task': metrics_task_description,
        'mean_val_auroc': mean_auroc, 'std_val_auroc': std_auroc,
        'mean_val_ap': mean_ap, 'std_val_ap': std_ap,
        'num_folds_metrics_calculated': num_folds_metrics_calculated,
        'num_total_folds': total_folds
    })

    if not file_has_leakage and files_processed > 0: print(f"  SUCCESS: No leakage detected in {pklz_filepath}.")

    return file_has_leakage, file_leakage_summary_records, file_leaked_ids_records, file_config_record


# --- Main Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    # ... (Argument parsing as before - no target key needed) ...
    parser = argparse.ArgumentParser(
        description="Check leakage, infer task, calculate metrics from PKLZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pklz_files", type=str, nargs='+', help="Path(s) to .pklz file(s).")
    parser.add_argument("--summary-csv", type=str, default="master_leakage_summary.csv", help="Output leakage summary CSV.")
    parser.add_argument("--config-csv", type=str, default="master_config_summary.csv", help="Output config and metrics summary CSV.")
    parser.add_argument("--leaked-ids-csv", type=str, default="detailed_leaked_ids.csv", help="Output detailed leaked IDs CSV.")
    args = parser.parse_args()

    # ... (Initialize accumulators as before) ...
    all_leakage_summary_data = []
    all_config_summary_data = []
    all_leaked_ids_data = []
    overall_leakage_found = False
    files_processed = 0
    files_with_leakage = 0

    print(f"Checking {len(args.pklz_files)} file path(s)...")
    print("Attempting to infer prediction task and calculate metrics automatically.")

    # --- Process Files ---
    # ... (Loop, call check_leakage_in_file, aggregate results as before) ...
    for filepath in args.pklz_files:
        if not os.path.isfile(filepath): print(f"\nWarning: Skip invalid path: {filepath}", file=sys.stderr); continue
        files_processed += 1
        try:
            leakage_found, summary, leaked_ids, config = check_leakage_in_file(filepath)
            all_leakage_summary_data.extend(summary)
            all_leaked_ids_data.extend(leaked_ids)
            if config: all_config_summary_data.append(config)
            if leakage_found: overall_leakage_found = True; files_with_leakage += 1
        except Exception as e: # Catch critical errors
            print(f"\nCRITICAL Error processing file {filepath}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            all_config_summary_data.append({'filename': os.path.basename(filepath), 'processing_error': str(e), 'metrics_task': 'CRITICAL ERROR'})
            print("  Attempting to continue...", file=sys.stderr)


    # --- Write CSV Reports ---
    # ... (Write reports as before, using the final column ordering) ...
    print("\n--- Writing CSV Reports ---")
    # Leakage Summary
    if all_leakage_summary_data:
        try:
            summary_df = pd.DataFrame.from_records(all_leakage_summary_data)
            summary_cols = ['filename', 'fold_number', 'leakage_detected', 'leaked_id_count', 'error']
            summary_cols = [col for col in summary_cols if col in summary_df.columns]
            summary_df = summary_df[summary_cols]
            summary_df.to_csv(args.summary_csv, index=False)
            print(f"  Leakage summary saved to: {args.summary_csv}")
        except Exception as e: print(f"  Error writing {args.summary_csv}: {e}", file=sys.stderr)
    else: print(f"  No leakage summary data for {args.summary_csv}.")
    # Config/Metrics Summary
    if all_config_summary_data:
         try:
            config_df = pd.DataFrame.from_records(all_config_summary_data)
            leading_cols = ['filename', 'metrics_task', 'mean_val_auroc', 'std_val_auroc', 'mean_val_ap', 'std_val_ap', 'num_folds_metrics_calculated', 'num_total_folds', 'processing_error']
            existing_cols = config_df.columns.tolist()
            other_config_cols = sorted([col for col in existing_cols if col not in leading_cols])
            final_cols = [col for col in leading_cols if col in existing_cols] + [col for col in other_config_cols if col not in leading_cols]
            config_df = config_df[final_cols]
            config_df.to_csv(args.config_csv, index=False, float_format='%.5f')
            print(f"  Config/metrics summary saved to: {args.config_csv}")
         except Exception as e: print(f"  Error writing {args.config_csv}: {e}", file=sys.stderr)
    else: print(f"  No config/metrics data for {args.config_csv}.")
    # Leaked IDs
    if all_leaked_ids_data:
         try:
            leaked_ids_df = pd.DataFrame.from_records(all_leaked_ids_data); leaked_ids_df.to_csv(args.leaked_ids_csv, index=False)
            print(f"  Detailed leaked IDs saved to: {args.leaked_ids_csv}")
         except Exception as e: print(f"  Error writing {args.leaked_ids_csv}: {e}", file=sys.stderr)
    else:
         if overall_leakage_found: print(f"  Warning: Leakage detected, but no detailed ID data collected. {args.leaked_ids_csv} not created.", file=sys.stderr)
         else: print(f"  No leaked IDs found, {args.leaked_ids_csv} not created.")

    # --- Final Summary ---
    # ... (Final summary and exit logic as before) ...
    print("\n\n===== Overall Check Summary =====")
    print(f"Files provided: {len(args.pklz_files)}"); print(f"Files processed: {files_processed}"); print(f"Files with leakage detected: {files_with_leakage}")
    if SKLEARN_AVAILABLE: print("Attempted performance metrics calculation based on inferred task structure.")
    else: print("Performance metrics NOT calculated (scikit-learn unavailable).")
    if not overall_leakage_found and files_processed > 0:
        print("\nSUCCESS: No patient ID leakage detected."); print(f"Check '{args.config_csv}' for config/metrics summaries."); sys.exit(0)
    elif files_processed == 0: print("\nWarning: No files were processed."); sys.exit(2)
    else:
        print("\nERROR: Patient ID leakage DETECTED or critical errors occurred."); print(f"Review CSV files:");
        if os.path.exists(args.summary_csv): print(f"  - Leakage Summary: {args.summary_csv}")
        if os.path.exists(args.config_csv): print(f"  - Configs & Metrics: {args.config_csv}")
        if os.path.exists(args.leaked_ids_csv): print(f"  - Leaked IDs: {args.leaked_ids_csv}")
        sys.exit(1)