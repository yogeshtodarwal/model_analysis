# File: check_data_leakage.py
import argparse
import os
import sys
import pandas as pd
import numpy as np
from collections.abc import MutableMapping
import warnings
import re
from collections import defaultdict

# --- Imports ---
try:
    from utils_tbox.utils_tbox import read_pklz, decompress_obj
except ImportError:
    print("Error: Could not import 'read_pklz' and 'decompress_obj' from 'utils_tbox.utils_tbox'.", file=sys.stderr)
    sys.exit(1)
try:
    # Add confusion_matrix import
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found. Metrics and stats will not be calculated.", file=sys.stderr)
    SKLEARN_AVAILABLE = False

# --- flatten_dict (Keep as before) ---
def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping): items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple, set)): items.append((new_key, str(v)))
        else: items.append((new_key, v))
    return dict(items)

# ==============================================================================
# REVISED Helper Function for Metric Calculation (More Metrics + Confusion Matrix)
# ==============================================================================
def calculate_all_metrics(y_true, y_pred_probs, task_type='binary', target_name='unknown'):
    """
    Calculates AUROC, AP, Precision, Recall, F1, Accuracy, and Confusion Matrix elements (for binary).
    Normalizes multiclass probabilities for AUROC/AP if needed.
    """
    metrics = {
        'auroc': np.nan, 'ap': np.nan,
        'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'accuracy': np.nan,
        'tn': np.nan, 'fp': np.nan, 'fn': np.nan, 'tp': np.nan # For binary confusion matrix
    }
    if not SKLEARN_AVAILABLE: return metrics

    try:
        y_true_np = np.asarray(y_true)
        y_pred_probs_np = np.asarray(y_pred_probs)
        if y_pred_probs_np.ndim == 1: y_pred_probs_np = y_pred_probs_np.reshape(-1, 1)

        valid_idx_true = ~np.isnan(y_true_np)
        # Check for NaNs in *any* prediction column for the row
        valid_idx_pred = ~np.isnan(y_pred_probs_np).any(axis=1) if y_pred_probs_np.ndim > 1 else ~np.isnan(y_pred_probs_np).flatten()

        valid_idx = valid_idx_true & valid_idx_pred

        if not np.any(valid_idx):
            print(f"      Warn [{target_name}]: No valid (non-NaN) samples found for metric calculation.", file=sys.stderr)
            return metrics
        y_true_filt = y_true_np[valid_idx]
        y_pred_probs_filt = y_pred_probs_np[valid_idx, :] if y_pred_probs_np.ndim > 1 else y_pred_probs_np[valid_idx].reshape(-1, 1)


        unique_classes_true = np.unique(y_true_filt)
        if len(unique_classes_true) < 2:
             print(f"      Warn [{target_name}]: Only one class present in y_true after filtering. Skipping most metrics.", file=sys.stderr)
             # Still calculate accuracy/confusion if possible
             if len(unique_classes_true) == 1:
                 y_pred_class = None
                 if task_type == 'binary':
                     threshold = 0.5
                     y_pred_class = (y_pred_probs_filt[:, 0] >= threshold).astype(int)
                 elif task_type == 'multiclass':
                     if y_pred_probs_filt.shape[1] > 0: # Avoid error if no pred cols somehow
                         y_pred_class = np.argmax(y_pred_probs_filt, axis=1)

                 if y_pred_class is not None:
                    try: metrics['accuracy'] = accuracy_score(y_true_filt, y_pred_class)
                    except Exception: pass # Ignore errors if only one class
                    if task_type == 'binary':
                        try:
                            # Calculate confusion matrix even for single class case
                            cm = confusion_matrix(y_true_filt, y_pred_class, labels=[0, 1])
                            if cm.shape == (2, 2): # Ensure correct shape
                                metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
                        except Exception: pass # Ignore errors
             return metrics

    except Exception as e: print(f"      Error preparing data for metrics [{target_name}]: {e}", file=sys.stderr); return metrics

    # --- AUROC and AP (from probabilities) ---
    try:
        if task_type == 'binary':
            if y_pred_probs_filt.shape[1] != 1: print(f"      Warn [{target_name}]: Binary AUROC/AP expects 1 pred col, got {y_pred_probs_filt.shape[1]}. Using first.", file=sys.stderr)
            metrics['auroc'] = roc_auc_score(y_true_filt, y_pred_probs_filt[:, 0])
            metrics['ap'] = average_precision_score(y_true_filt, y_pred_probs_filt[:, 0])
        elif task_type == 'multiclass':
            if y_pred_probs_filt.shape[1] < 2 : raise ValueError("Multiclass AUROC/AP expects >= 2 pred cols")

            # <<< NORMALIZATION ADDED HERE >>>
            pred_sums = y_pred_probs_filt.sum(axis=1)
            # Check if normalization is needed (avoid if sums are already ~1 or 0)
            needs_norm = np.any((pred_sums > 1e-6) & (np.abs(pred_sums - 1.0) > 1e-6))
            if needs_norm:
                print(f"      Note [{target_name}]: Normalizing multiclass prediction probabilities for AUROC/AP.", file=sys.stderr)
                # Add epsilon to avoid division by zero
                y_pred_probs_norm = y_pred_probs_filt / (pred_sums[:, np.newaxis] + 1e-9)
            else:
                y_pred_probs_norm = y_pred_probs_filt
            # <<< END NORMALIZATION >>>

            metrics['auroc'] = roc_auc_score(y_true_filt, y_pred_probs_norm, multi_class='ovr', average='weighted')
            try:
                # Use normalized probabilities for AP as well
                metrics['ap'] = average_precision_score(y_true_filt, y_pred_probs_norm, average='weighted')
            except TypeError:
                metrics['ap'] = np.nan

    except ValueError as ve: print(f"      Warn [{target_name}]: AUROC/AP calc failed: {ve}", file=sys.stderr)
    except Exception as e: print(f"      Error calc AUROC/AP [{target_name}]: {e}", file=sys.stderr)

    # --- Threshold-based Metrics (Precision, Recall, F1, Accuracy, Confusion Matrix) ---
    y_pred_class = None
    if task_type == 'binary':
        threshold = 0.5
        y_pred_class = (y_pred_probs_filt[:, 0] >= threshold).astype(int)
    elif task_type == 'multiclass':
        y_pred_class = np.argmax(y_pred_probs_filt, axis=1)

    if y_pred_class is not None:
        try:
            avg_type = 'binary' if task_type == 'binary' else 'weighted'
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_filt, y_pred_class, average=avg_type, zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            metrics['accuracy'] = accuracy_score(y_true_filt, y_pred_class)

            # --- Confusion Matrix for BINARY ---
            if task_type == 'binary':
                # Ensure labels=[0, 1] to get consistent TN, FP, FN, TP order
                cm = confusion_matrix(y_true_filt, y_pred_class, labels=[0, 1])
                if cm.shape == (2, 2): # Handle edge cases where maybe only one class predicted
                   metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
                else:
                   print(f"      Warn [{target_name}]: Confusion matrix shape unexpected ({cm.shape}). TN/FP/FN/TP set to NaN.", file=sys.stderr)

        except ValueError as ve: print(f"      Warn [{target_name}]: Prec/Rec/F1/Acc/CM calc failed: {ve}", file=sys.stderr)
        except Exception as e: print(f"      Error calc Prec/Rec/F1/Acc/CM [{target_name}]: {e}", file=sys.stderr)

    return metrics

# ==============================================================================
# REVISED Function to Parse Prediction Columns (No major change needed here)
# ==============================================================================
def parse_prediction_info(df_columns):
    """
    Analyzes prediction columns to infer model name, task structure, and target columns.
    """
    pred_col_pattern = re.compile(r"^pred__([^_].*?)__(.+)$")
    predictions = defaultdict(list)
    found_match = False
    # Ensure df_columns is iterable, handle potential None
    if df_columns is None: df_columns = []
    for col in df_columns:
        match = pred_col_pattern.match(col)
        if match:
            found_match = True
            try: model_name, description = match.groups(); predictions[model_name].append({'full_col': col, 'desc': description})
            except ValueError: pass

    if not found_match: return None # No prediction columns found

    if len(predictions) > 1: model_name = max(predictions, key=lambda k: len(predictions[k])); print(f"      Warn: Multiple models found. Using '{model_name}'.", file=sys.stderr)
    elif not predictions: return None # Handle case where regex matched but dict is empty (shouldn't happen)
    else: model_name = list(predictions.keys())[0]

    model_predictions = predictions[model_name]
    model_predictions.sort(key=lambda p: p['desc']) # Sort for consistency
    descriptions = [p['desc'] for p in model_predictions]
    full_pred_cols = [p['full_col'] for p in model_predictions]

    task_info = {
        'model_name': model_name, 'pred_cols': full_pred_cols, 'pred_descriptions': descriptions,
        'required_target_bases': [], 'task_type': None, 'class_mapping': None,
        'positive_descriptions': [], 'negative_description': None # Store positive/negative descs
    }

    positive_target_pattern = re.compile(r"^target__(\w+)$")
    actual_target_bases = set()
    has_negation = False
    negative_desc = None
    positive_descs = []

    for desc in descriptions:
        match = positive_target_pattern.match(desc)
        if match:
            actual_target_bases.add(match.group(1))
            positive_descs.append(desc)
        # Broader check for negative class indicators
        elif desc.startswith('not_target') or desc.startswith('no_target') or desc.lower() in ['none', 'other'] or '_neg' in desc or 'negative' in desc.lower():
            has_negation = True
            if negative_desc is None: negative_desc = desc # Store the first encountered negative description

    task_info['required_target_bases'] = sorted(list(actual_target_bases))
    task_info['positive_descriptions'] = sorted(positive_descs)
    task_info['negative_description'] = negative_desc

    num_descs = len(descriptions)
    num_actual_targets = len(actual_target_bases)

    # --- Inference Logic ---
    if num_descs == 1 and num_actual_targets == 0:
        target_base_from_desc = descriptions[0]
        task_info['required_target_bases'] = [target_base_from_desc]
        task_info['positive_descriptions'] = descriptions # Assume the single desc is positive
        task_info['task_type'] = 'binary'
        # print(f"      Inferred BINARY task (single pred col, direct desc) for target '{target_base_from_desc}'")

    elif num_descs == 1 and num_actual_targets == 1: # Single pred, e.g., pred__model__target_death
        task_info['task_type'] = 'binary'
        # print(f"      Inferred BINARY task (single pred col) for target '{task_info['required_target_bases'][0]}'")

    elif num_descs == 2 and num_actual_targets == 1 and has_negation: # target__A, not_target__A
         task_info['task_type'] = 'binary'
         # Find the positive prediction column (starts with target__)
         pos_desc = next((d for d in descriptions if d.startswith("target__")), None)
         pos_col = next((p['full_col'] for p in model_predictions if p['desc'] == pos_desc), None)
         if pos_col:
             task_info['pred_cols'] = [pos_col] # Use only the positive class probability column for binary metrics
             # print(f"      Inferred BINARY task (explicit pos/neg) for target '{task_info['required_target_bases'][0]}'. Using positive pred: {pos_col}")
         else: print(f"      Warn: Could not find positive pred col for binary task. Desc='{pos_desc}'. Skip.", file=sys.stderr); return None

    elif num_descs > 1 and num_actual_targets >= 1: # Multiclass
         task_info['task_type'] = 'multiclass'
         task_info['class_mapping'] = {desc: i for i, desc in enumerate(descriptions)}
         # print(f"      Inferred MULTICLASS task involving targets {task_info['required_target_bases']} from {num_descs} prediction columns.")

    else:
         print(f"      Warning: Could not reliably infer task structure: Descs={descriptions}, ActualTargets={actual_target_bases}. Skip.", file=sys.stderr)
         return None

    # --- Final Validation: Check for existence of required ACTUAL target columns ---
    missing_targets_truth_cols = []
    if not task_info['required_target_bases']: # Check if we expect at least one target truth column
        # This is okay for the single-column binary case where description is the target
        if not (task_info['task_type'] == 'binary' and num_descs == 1 and num_actual_targets == 0):
            print(f"      Warning: Task inference issue - no required target bases identified. Skip.", file=sys.stderr)
            return None

    # Construct full target column names needed based on inferred bases
    required_truth_cols = [f"target__{target_base}" for target_base in task_info['required_target_bases']]
    for target_col_name in required_truth_cols:
        if target_col_name not in df_columns:
             missing_targets_truth_cols.append(target_col_name)

    if missing_targets_truth_cols:
         print(f"      Warning: Inferred task requires missing truth column(s): {missing_targets_truth_cols}. Skip.", file=sys.stderr)
         return None

    # print(f"      Task inference successful: {task_info['task_type']}")
    return task_info

# ==============================================================================
# REVISED construct_multiclass_y_true (Return neg_class_index)
# ==============================================================================
def construct_multiclass_y_true(df, task_info):
    """Constructs multiclass y_true and identifies the negative class index."""
    if task_info['task_type'] != 'multiclass': return None, -1 # Return None for y_true, -1 for neg_index

    n_samples = len(df)
    y_true = np.full(n_samples, -1, dtype=int) # Use -1 as placeholder for unassigned
    class_mapping = task_info['class_mapping']
    descriptions = task_info['pred_descriptions']
    neg_class_index = -1 # Initialize negative class index

    # Identify negative class index based on description stored in task_info
    if task_info['negative_description'] and task_info['negative_description'] in class_mapping:
        neg_class_index = class_mapping[task_info['negative_description']]
    else:
        # Fallback if negative description wasn't identified clearly earlier or missing
        for desc, index in class_mapping.items():
             if desc.startswith('not_target') or desc.startswith('no_target') or desc.lower() in ['none', 'other'] or '_neg' in desc:
                 if neg_class_index == -1: neg_class_index = index
                 # else: print(f"      Warn: Multiple potential neg classes found. Using first: {task_info.get('negative_description', 'N/A')}", file=sys.stderr) # Be less verbose

    positive_target_pattern = re.compile(r"^target__(\w+)$")
    assigned_positive = np.zeros(n_samples, dtype=bool) # Tracks if a sample has already been assigned a *positive* class

    # Assign positive classes first (priority)
    for desc, class_index in class_mapping.items():
        if class_index == neg_class_index: continue # Skip the negative class for now

        match = positive_target_pattern.match(desc)
        if match:
            target_base = match.group(1)
            target_col = f"target__{target_base}"
            if target_col not in df.columns:
                print(f"      Error: construct_y_true needs '{target_col}' but missing.", file=sys.stderr)
                return None, -1 # Indicate error

            # Assign this class only if the target is 1 AND it hasn't been assigned another positive class yet
            assign_mask = (df[target_col] == 1) & (~assigned_positive)
            y_true[assign_mask] = class_index
            assigned_positive[assign_mask] = True # Mark as assigned a positive class
        # else: # Handle cases where a prediction description doesn't match target__ pattern (maybe 'Other_Positive'?)
             # This logic assumes descriptions starting with 'target__' are the primary positive classes.
             # Add specific handling here if other description patterns signify positive classes.
             # print(f"      Debug: Description '{desc}' in multiclass mapping does not start with 'target__'.", file=sys.stderr)
             # For now, we assume only 'target__*' are positive indicators in truth columns.


    # Assign negative class to remaining unassigned samples
    unassigned_mask = (y_true == -1)
    if neg_class_index != -1:
        y_true[unassigned_mask] = neg_class_index
    elif np.any(unassigned_mask):
        # If no explicit negative class, but some samples are unassigned (meaning they were 0 for all target__ cols)
        # This implies they belong to a default negative/other category if one exists in predictions,
        # otherwise it's an issue.
        # We already tried to find neg_class_index. If still -1, maybe map to lowest index? Risky.
        print(f"      Error: Samples remain unassigned but no clear negative class found (neg_idx={neg_class_index}). Cannot construct y_true.", file=sys.stderr)
        return None, -1

    # Final check - should not happen if logic is correct
    if np.any(y_true == -1):
        print(f"      Error: Failed to assign all samples a class in y_true construction.", file=sys.stderr)
        return None, -1

    return y_true, neg_class_index


# ==============================================================================
# NEW Helper Function for Data Statistics
# ==============================================================================
def calculate_data_stats(df, task_info, dataset_label="unknown"):
    """ Calculates patient and window counts for train/val sets. """
    stats = {}
    if df is None or df.empty:
        # print(f"      Info: Empty {dataset_label} dataframe, skipping stats.")
        return stats

    if "ids__uid" not in df.columns:
        print(f"      Warn: 'ids__uid' column missing in {dataset_label} df. Cannot calculate patient counts.", file=sys.stderr)
        stats[f'{dataset_label}_total_windows'] = len(df)
        stats[f'{dataset_label}_total_patients'] = np.nan
        return stats

    # Make sure 'ids__uid' is suitable for nunique (handle potential non-hashable types if necessary)
    try:
        total_patients = df['ids__uid'].nunique()
    except TypeError:
         print(f"      Warn: 'ids__uid' column in {dataset_label} df contains non-hashable types. Patient counts may be inaccurate.", file=sys.stderr)
         total_patients = np.nan # Or fallback: len(df['ids__uid'].astype(str).unique())

    stats[f'{dataset_label}_total_windows'] = len(df)
    stats[f'{dataset_label}_total_patients'] = total_patients

    # --- Per-Target Stats ---
    target_bases = task_info.get('required_target_bases', [])
    all_pos_target_cols = [] # Keep track of positive target columns present

    for target_base in target_bases:
        target_col = f"target__{target_base}"
        if target_col not in df.columns:
            # print(f"      Info: Target column '{target_col}' not found in {dataset_label} df for stats.", file=sys.stderr)
            continue # Skip stats for this target if column missing

        all_pos_target_cols.append(target_col)
        try:
            # Ensure target column is numeric-like (handle potential errors)
            target_series = pd.to_numeric(df[target_col], errors='coerce')
            pos_mask = (target_series == 1)
            neg_mask = (target_series == 0) # Assuming binary 0/1 targets

            # Patient counts (unique IDs)
            stats[f'{dataset_label}_patients_pos_{target_base}'] = df.loc[pos_mask, 'ids__uid'].nunique()
            stats[f'{dataset_label}_patients_neg_{target_base}'] = df.loc[neg_mask, 'ids__uid'].nunique()

            # Window counts (rows)
            stats[f'{dataset_label}_windows_pos_{target_base}'] = pos_mask.sum()
            stats[f'{dataset_label}_windows_neg_{target_base}'] = neg_mask.sum()

            # Check for NaNs introduced by coerce or originally present
            nan_count = target_series.isna().sum()
            if nan_count > 0:
                stats[f'{dataset_label}_windows_nan_{target_base}'] = nan_count

        except Exception as e:
            print(f"      Error calculating stats for target '{target_base}' in {dataset_label}: {e}", file=sys.stderr)
            stats[f'{dataset_label}_patients_pos_{target_base}'] = np.nan
            stats[f'{dataset_label}_patients_neg_{target_base}'] = np.nan
            stats[f'{dataset_label}_windows_pos_{target_base}'] = np.nan
            stats[f'{dataset_label}_windows_neg_{target_base}'] = np.nan


    # --- Overall Positive/Negative Stats (Any positive vs. None) ---
    if all_pos_target_cols: # Only if we found some positive target columns
        try:
            # Mask for rows where *at least one* positive target is 1
            overall_pos_mask = (df[all_pos_target_cols].sum(axis=1) > 0)
            # Mask for rows where *all* positive targets are 0 (or NaN handled by sum > 0)
            overall_neg_mask = ~overall_pos_mask

            stats[f'{dataset_label}_patients_pos_any'] = df.loc[overall_pos_mask, 'ids__uid'].nunique()
            stats[f'{dataset_label}_windows_pos_any'] = overall_pos_mask.sum()

            stats[f'{dataset_label}_patients_neg_all'] = df.loc[overall_neg_mask, 'ids__uid'].nunique()
            stats[f'{dataset_label}_windows_neg_all'] = overall_neg_mask.sum()
        except Exception as e:
            print(f"      Error calculating overall pos/neg stats in {dataset_label}: {e}", file=sys.stderr)
            stats[f'{dataset_label}_patients_pos_any'] = np.nan
            stats[f'{dataset_label}_windows_pos_any'] = np.nan
            stats[f'{dataset_label}_patients_neg_all'] = np.nan
            stats[f'{dataset_label}_windows_neg_all'] = np.nan

    return stats

# ==============================================================================
# REVISED Main Leakage Check Logic (Integrate Stats, Binary Option, Confusion Matrix)
# ==============================================================================
def check_leakage_in_file(pklz_filepath, calculate_binary_from_multi=False):
    print(f"\n--- Checking File: {os.path.basename(pklz_filepath)} ---")
    file_leakage_summary_records = []
    file_leaked_ids_records = []
    file_config_record = {'filename': os.path.basename(pklz_filepath)}
    file_has_leakage = False
    overall_cfg = {}
    per_fold_detailed_metrics_and_stats = [] # Store metrics AND stats

    # --- Accumulators for median calculation (including binary & confusion) ---
    metric_accumulators = defaultdict(list) # Use defaultdict for easier appending

    metrics_task_description = "N/A"
    consistent_task_info_across_folds = None # Store the first valid task_info

    try: data = read_pklz(pklz_filepath)
    except Exception as e:
        print(f"Error reading {pklz_filepath}: {e}", file=sys.stderr)
        file_config_record.update({'metrics_task': 'Error: Read PKLZ', 'median_val_auroc': np.nan, 'median_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': 0})
        return False, [], [], file_config_record, []

    # Config extraction
    if isinstance(data, dict) and 'cfg' in data and isinstance(data['cfg'], dict):
        overall_cfg = data['cfg']
        try: file_config_record.update(flatten_dict(overall_cfg))
        except Exception as e: print(f"  Warning: Could not flatten 'cfg': {e}", file=sys.stderr)

    # Results validation
    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        print(f"Error: Invalid/missing 'results' list in {pklz_filepath}.", file=sys.stderr)
        file_config_record.update({'metrics_task': 'Error: Invalid results', 'median_val_auroc': np.nan, 'median_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': 0})
        return False, [], [], file_config_record, []

    compressed_datasets = data["results"]
    total_folds = len(compressed_datasets)
    if not compressed_datasets:
        print(f"  Info: 'results' list is empty.")
        file_config_record.update({'metrics_task': 'No folds', 'median_val_auroc': np.nan, 'median_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': 0})
        # Return empty leakage summary for consistency if needed
        return False, [{'filename': os.path.basename(pklz_filepath),'fold_number': 'N/A','leakage_detected': False,'leaked_id_count': 0, 'error': 'No folds'}], [], file_config_record, []

    print(f"  Found {total_folds} fold(s) in 'results'.")

    for i, compressed_fold_data in enumerate(compressed_datasets):
        fold_num = i + 1
        print(f"   --- Processing Fold {fold_num}/{total_folds} ---")
        fold_summary_record = {'filename': os.path.basename(pklz_filepath),'fold_number': fold_num,'leakage_detected': False,'leaked_id_count': 0, 'error': None}
        # Initialize record for detailed CSV (metrics AND stats)
        fold_metrics_stats_record = {'filename': os.path.basename(pklz_filepath), 'fold_number': fold_num}

        train_df, val_df = None, None # Initialize
        task_info = None # Initialize for the fold

        try:
            fold_data = decompress_obj(compressed_fold_data)
            if fold_data is None or not isinstance(fold_data, dict):
                fold_summary_record['error'] = 'Decompress Error or Invalid Type'; raise ValueError("Fold data decompression failed")

            val_df = fold_data.get("val")
            train_df = fold_data.get("train") # Needed for stats

            if not isinstance(val_df, pd.DataFrame): # Train can be missing/None, but Val is essential for metrics
                fold_summary_record['error'] = 'Invalid/Missing Val DF'; raise ValueError("Validation DataFrame missing or invalid")
            if not isinstance(train_df, pd.DataFrame):
                print(f"    Warn: Train DF missing or not a DataFrame for Fold {fold_num}. Train stats skipped.", file=sys.stderr)
                # No error, but stats will be limited

            # --- Leakage Check ---
            fold_leakage_detected, leaked_count_for_fold = False, 0
            if train_df is not None and "ids__uid" in train_df.columns and "ids__uid" in val_df.columns:
                 try:
                     train_ids = set(train_df['ids__uid'].dropna().unique())
                     val_ids = set(val_df['ids__uid'].dropna().unique())
                     intersection = train_ids.intersection(val_ids)
                     if intersection:
                         fold_leakage_detected = True; file_has_leakage = True
                         leak_list = sorted(list(intersection)); leaked_count_for_fold = len(leak_list)
                         print(f"    Leakage DETECTED Fold {fold_num}! ({leaked_count_for_fold} IDs)")
                         for leaked_id in leak_list: file_leaked_ids_records.append({'filename': os.path.basename(pklz_filepath),'fold_number': fold_num,'leaked_ids__uid': leaked_id})
                 except Exception as leak_err:
                      print(f"    Warn: Error during leakage check Fold {fold_num}: {leak_err}", file=sys.stderr)
                      fold_summary_record['error'] = f"Leakage Check Error: {leak_err}"

            fold_summary_record['leakage_detected'] = fold_leakage_detected
            fold_summary_record['leaked_id_count'] = leaked_count_for_fold

            # --- Task Inference and Metrics/Stats ---
            if not SKLEARN_AVAILABLE:
                print("    Skipping metrics and stats (scikit-learn unavailable).")
                continue # Go to next fold after leakage check

            if val_df.empty:
                print(f"    Skipping metrics and stats Fold {fold_num} (validation df empty).")
                fold_summary_record['error'] = 'Val DF Empty'
                continue

            # Infer task from validation data columns
            task_info = parse_prediction_info(val_df.columns)
            if task_info is None:
                print(f"    Skipping metrics and stats Fold {fold_num} (task inference failed/cols missing).")
                fold_summary_record['error'] = 'Task Inference Failed'
                continue # Cannot proceed without task info

            # Store consistent task info from the first successful fold
            if consistent_task_info_across_folds is None:
                consistent_task_info_across_folds = task_info
                req_targets = '/'.join(task_info['required_target_bases']) if task_info['required_target_bases'] else 'Unknown'
                metrics_task_description = f"{task_info['task_type']} ({req_targets})"
                if task_info['task_type'] == 'multiclass' and calculate_binary_from_multi:
                     metrics_task_description += " [+Binary]"


            # --- Calculate Data Stats ---
            print("      Calculating data statistics...")
            train_stats = calculate_data_stats(train_df, task_info, "train")
            val_stats = calculate_data_stats(val_df, task_info, "val")
            fold_metrics_stats_record.update(train_stats)
            fold_metrics_stats_record.update(val_stats)

            # --- Calculate Performance Metrics ---
            print("      Calculating performance metrics...")
            y_true_multi, neg_class_index = None, -1 # For multiclass case

            if task_info['task_type'] == 'binary':
                target_base = task_info['required_target_bases'][0]
                target_col = f"target__{target_base}"
                if target_col not in val_df.columns:
                     print(f"    Error: Required target column '{target_col}' missing in val_df Fold {fold_num}. Skipping metrics.", file=sys.stderr)
                     fold_summary_record['error'] = f'Missing target {target_col}'
                     continue
                y_true = val_df[target_col]
                y_pred_probs = val_df[task_info['pred_cols']] # Should be single column based on parse_prediction_info logic
                metrics_label = f"Binary ({target_base})"

                current_fold_metrics = calculate_all_metrics(y_true, y_pred_probs, 'binary', metrics_label)
                print(f"        Fold {fold_num} Metrics [{metrics_label}]: AUROC={current_fold_metrics.get('auroc', np.nan):.4f}, AP={current_fold_metrics.get('ap', np.nan):.4f}, F1={current_fold_metrics.get('f1_score', np.nan):.4f}, Acc={current_fold_metrics.get('accuracy', np.nan):.4f}")
                fold_metrics_stats_record.update(current_fold_metrics) # Add metrics to the record
                # Accumulate for median calculation
                for m_key, m_val in current_fold_metrics.items():
                    if not np.isnan(m_val): metric_accumulators[m_key].append(m_val)


            elif task_info['task_type'] == 'multiclass':
                y_true_multi, neg_class_index = construct_multiclass_y_true(val_df, task_info)
                if y_true_multi is None:
                    print(f"    Error constructing multiclass y_true Fold {fold_num}. Skipping metrics.", file=sys.stderr)
                    fold_summary_record['error'] = 'Multiclass y_true construction failed'
                    continue

                y_pred_probs_multi = val_df[task_info['pred_cols']]
                metrics_label_multi = f"Multi ({'/'.join(task_info['required_target_bases'])})"

                # Calculate Multiclass Metrics
                current_fold_metrics_multi = calculate_all_metrics(y_true_multi, y_pred_probs_multi, 'multiclass', metrics_label_multi)
                print(f"        Fold {fold_num} Metrics [{metrics_label_multi}]: AUROC={current_fold_metrics_multi.get('auroc', np.nan):.4f}, AP={current_fold_metrics_multi.get('ap', np.nan):.4f}, F1={current_fold_metrics_multi.get('f1_score', np.nan):.4f}, Acc={current_fold_metrics_multi.get('accuracy', np.nan):.4f}")
                # Add with prefix to distinguish from potential binary metrics
                fold_metrics_stats_record.update({f"multi_{k}": v for k, v in current_fold_metrics_multi.items()})
                # Accumulate for median calculation
                for m_key, m_val in current_fold_metrics_multi.items():
                    if not np.isnan(m_val): metric_accumulators[f"multi_{m_key}"].append(m_val)


                # --- Optional Binary Conversion from Multiclass ---
                if calculate_binary_from_multi:
                    print(f"        Calculating derived binary metrics (Any Positive vs Negative)...")
                    # Derive binary y_true: 1 if not the negative class, 0 otherwise
                    if neg_class_index == -1:
                        print(f"        Warn: Cannot derive binary metrics Fold {fold_num} - no negative class identified for multiclass task.", file=sys.stderr)
                    else:
                        y_true_binary = (y_true_multi != neg_class_index).astype(int)

                        # Derive binary y_pred_probs: sum probabilities of all positive classes
                        positive_indices = [idx for desc, idx in task_info['class_mapping'].items() if idx != neg_class_index]
                        if not positive_indices:
                             print(f"        Warn: Cannot derive binary metrics Fold {fold_num} - no positive class indices found.", file=sys.stderr)
                        else:
                             # Ensure y_pred_probs_multi is 2D numpy array
                             y_pred_probs_multi_np = np.asarray(y_pred_probs_multi)
                             if y_pred_probs_multi_np.ndim == 1: y_pred_probs_multi_np = y_pred_probs_multi_np.reshape(-1,1) # Reshape just in case

                             # Select columns corresponding to positive classes and sum row-wise
                             if max(positive_indices) < y_pred_probs_multi_np.shape[1]:
                                 y_pred_probs_binary = y_pred_probs_multi_np[:, positive_indices].sum(axis=1)
                                 metrics_label_binary = f"Binary (Any vs Neg {neg_class_index})"

                                 current_fold_metrics_binary = calculate_all_metrics(y_true_binary, y_pred_probs_binary, 'binary', metrics_label_binary)
                                 print(f"        Fold {fold_num} Metrics [{metrics_label_binary}]: AUROC={current_fold_metrics_binary.get('auroc', np.nan):.4f}, AP={current_fold_metrics_binary.get('ap', np.nan):.4f}, F1={current_fold_metrics_binary.get('f1_score', np.nan):.4f}, Acc={current_fold_metrics_binary.get('accuracy', np.nan):.4f}")
                                 # Add with prefix
                                 fold_metrics_stats_record.update({f"binary_{k}": v for k, v in current_fold_metrics_binary.items()})
                                 # Accumulate for median calculation
                                 for m_key, m_val in current_fold_metrics_binary.items():
                                     if not np.isnan(m_val): metric_accumulators[f"binary_{m_key}"].append(m_val)
                             else:
                                 print(f"        Warn: Positive class index out of bounds for pred_probs Fold {fold_num}. Cannot derive binary metrics.", file=sys.stderr)


        except Exception as e:
             print(f"    Unexpected error processing Fold {fold_num}: {e}. Skipping fold analysis.", file=sys.stderr)
             import traceback; traceback.print_exc(file=sys.stderr)
             fold_summary_record['error'] = f"Unexpected: {e}"
             # Ensure partial records are still appended if error happens mid-fold
        finally:
             # Always append records even if errors occurred
             file_leakage_summary_records.append(fold_summary_record)
             per_fold_detailed_metrics_and_stats.append(fold_metrics_stats_record)


    # --- Aggregate Metrics using MEDIAN ---
    num_folds_metrics_calculated = len(metric_accumulators.get('auroc', [])) # Use primary AUROC as reference
    if not num_folds_metrics_calculated: # If primary binary AUROC empty, check multiclass/derived binary
        num_folds_metrics_calculated = len(metric_accumulators.get('multi_auroc', []))
    if not num_folds_metrics_calculated:
        num_folds_metrics_calculated = len(metric_accumulators.get('binary_auroc', []))


    agg_metrics = {}
    if num_folds_metrics_calculated > 0:
        print(f"  Aggregating Val Metrics (Median) [{metrics_task_description}] across {num_folds_metrics_calculated}/{total_folds} folds with results...")
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning) # For median/mean/std of empty/single lists
             for m_key, m_list in metric_accumulators.items():
                  if m_list: # Only calculate if list is not empty
                      # Use 'val_' prefix for aggregated validation metrics
                      agg_metrics[f'median_val_{m_key}'] = np.median(m_list)
                      # Optionally calculate mean/std for primary metrics
                      if m_key in ['auroc', 'ap', 'f1_score', 'accuracy', 'binary_auroc', 'binary_ap', 'binary_f1_score', 'binary_accuracy', 'multi_auroc', 'multi_ap', 'multi_f1_score', 'multi_accuracy']:
                           agg_metrics[f'mean_val_{m_key}'] = np.mean(m_list)
                           agg_metrics[f'std_val_{m_key}'] = np.std(m_list)
                  else: # Ensure keys exist even if no data
                      agg_metrics[f'median_val_{m_key}'] = np.nan
                      if m_key in ['auroc', 'ap', 'f1_score', 'accuracy', 'binary_auroc', 'binary_ap', 'binary_f1_score', 'binary_accuracy', 'multi_auroc', 'multi_ap', 'multi_f1_score', 'multi_accuracy']:
                           agg_metrics[f'mean_val_{m_key}'] = np.nan
                           agg_metrics[f'std_val_{m_key}'] = np.nan

        # Print a summary of key aggregated metrics
        primary_auroc_key = 'median_val_auroc' if 'median_val_auroc' in agg_metrics else ('median_val_multi_auroc' if 'median_val_multi_auroc' in agg_metrics else 'N/A')
        primary_f1_key = 'median_val_f1_score' if 'median_val_f1_score' in agg_metrics else ('median_val_multi_f1_score' if 'median_val_multi_f1_score' in agg_metrics else 'N/A')
        derived_binary_auroc_key = 'median_val_binary_auroc' if 'median_val_binary_auroc' in agg_metrics else 'N/A'

        print(f"    Median Primary AUROC: {agg_metrics.get(primary_auroc_key, np.nan):.4f}" if primary_auroc_key != 'N/A' else "    Median Primary AUROC: N/A")
        print(f"    Median Primary F1: {agg_metrics.get(primary_f1_key, np.nan):.4f}" if primary_f1_key != 'N/A' else "    Median Primary F1: N/A")
        if derived_binary_auroc_key != 'N/A' and calculate_binary_from_multi : print(f"    Median Derived Binary AUROC: {agg_metrics.get(derived_binary_auroc_key, np.nan):.4f}")

    else:
        if SKLEARN_AVAILABLE and total_folds > 0: print(f"  No valid metrics calculated across folds.")
        elif not SKLEARN_AVAILABLE: print(f"  Metrics calculation skipped (scikit-learn unavailable).")


    # Update the main config record for the file
    file_config_record.update({'metrics_task': metrics_task_description, 'num_folds_metrics_calculated': num_folds_metrics_calculated, 'num_total_folds': total_folds})
    file_config_record.update(agg_metrics) # Add all aggregated metrics

    if not file_has_leakage and files_processed > 0: print(f"  --- File OK: No leakage detected in {os.path.basename(pklz_filepath)} ---")
    elif file_has_leakage: print(f"  --- File LEAKAGE: Leakage detected in {os.path.basename(pklz_filepath)} ---")

    return file_has_leakage, file_leakage_summary_records, file_leaked_ids_records, file_config_record, per_fold_detailed_metrics_and_stats


# --- Main Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check leakage, infer task, calculate metrics & stats from PKLZ files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pklz_files", type=str, nargs='+', help="Path(s) to .pklz file(s).")
    parser.add_argument("--summary-csv", type=str, default="master_leakage_summary.csv", help="Leakage summary CSV (per fold).")
    parser.add_argument("--config-csv", type=str, default="master_config_metrics_summary.csv", help="Config and aggregated (median/mean) metrics summary CSV (per file).")
    parser.add_argument("--leaked-ids-csv", type=str, default="detailed_leaked_ids.csv", help="Detailed leaked IDs CSV.")
    parser.add_argument("--detailed-metrics-csv", type=str, default="detailed_fold_metrics_stats.csv", help="Detailed per-fold performance metrics and data statistics CSV.")
    # --- NEW ARGUMENT for binary conversion ---
    parser.add_argument("--binary-from-multi", action='store_true', help="If a multiclass task is detected, also calculate binary metrics (Any Positive vs. Negative).")
    args = parser.parse_args()

    all_leakage_summary_data, all_config_summary_data, all_leaked_ids_data = [], [], []
    all_detailed_fold_data = [] # Accumulator for the detailed metrics and stats
    overall_leakage_found, files_processed, files_with_leakage = False, 0, 0

    print(f"Checking {len(args.pklz_files)} file path(s)...")
    print("Attempting to infer task, calculate metrics (median/mean aggregation), and data stats.")
    if args.binary_from_multi: print("Option '--binary-from-multi' enabled.")
    if not SKLEARN_AVAILABLE: print("WARNING: scikit-learn not found. Metrics and stats WILL NOT be calculated.", file=sys.stderr)


    for filepath in args.pklz_files:
        if not os.path.isfile(filepath): print(f"\nWarn: Skip invalid path: {filepath}", file=sys.stderr); continue
        files_processed += 1
        try:
            # Pass the new argument to the checking function
            leakage_found, summary, leaked_ids, config, detailed_data_for_file = check_leakage_in_file(filepath, args.binary_from_multi)
            all_leakage_summary_data.extend(summary)
            all_leaked_ids_data.extend(leaked_ids)
            if config: all_config_summary_data.append(config)
            all_detailed_fold_data.extend(detailed_data_for_file) # Aggregate detailed data
            if leakage_found: overall_leakage_found = True; files_with_leakage += 1
        except Exception as e:
            print(f"\nCRITICAL Error processing {filepath}: {e}", file=sys.stderr); import traceback; traceback.print_exc(file=sys.stderr)
            # Add a basic error record to config summary
            all_config_summary_data.append({'filename': os.path.basename(filepath), 'processing_error': f'CRITICAL: {e}', 'metrics_task': 'CRITICAL ERROR'})

    print("\n--- Writing CSV Reports ---")

    # Leakage Summary (per fold)
    if all_leakage_summary_data:
        try:
            summary_df = pd.DataFrame.from_records(all_leakage_summary_data)
            summary_cols = ['filename', 'fold_number', 'leakage_detected', 'leaked_id_count', 'error']
            # Ensure columns exist before selecting
            summary_df = summary_df[[col for col in summary_cols if col in summary_df.columns]]
            summary_df.to_csv(args.summary_csv, index=False)
            print(f"  Leakage summary saved: {args.summary_csv}")
        except Exception as e: print(f"  Error writing {args.summary_csv}: {e}", file=sys.stderr)
    else: print(f"  No leakage summary data generated for {args.summary_csv}.")


    # Config/Aggregated Metrics Summary (per file)
    if all_config_summary_data:
         try:
            config_df = pd.DataFrame.from_records(all_config_summary_data)
            # Define preferred order, starting with key info, then metrics, then config params
            leading_cols = [
                'filename', 'metrics_task', 'processing_error',
                'num_folds_metrics_calculated', 'num_total_folds',
                # Primary Metrics (Binary or Multiclass) - Allow flexibility
                'median_val_auroc', 'mean_val_auroc', 'std_val_auroc',
                'median_val_ap', 'mean_val_ap', 'std_val_ap',
                'median_val_precision', 'median_val_recall',
                'median_val_f1_score', 'mean_val_f1_score', 'std_val_f1_score',
                'median_val_accuracy', 'mean_val_accuracy', 'std_val_accuracy',
                 # Multiclass specific (if applicable)
                'median_val_multi_auroc', 'mean_val_multi_auroc', 'std_val_multi_auroc',
                'median_val_multi_ap', 'mean_val_multi_ap', 'std_val_multi_ap',
                'median_val_multi_precision', 'median_val_multi_recall',
                'median_val_multi_f1_score', 'mean_val_multi_f1_score', 'std_val_multi_f1_score',
                'median_val_multi_accuracy', 'mean_val_multi_accuracy', 'std_val_multi_accuracy',
                # Derived Binary Metrics (if applicable)
                'median_val_binary_auroc', 'mean_val_binary_auroc', 'std_val_binary_auroc',
                'median_val_binary_ap', 'mean_val_binary_ap', 'std_val_binary_ap',
                'median_val_binary_precision', 'median_val_binary_recall',
                'median_val_binary_f1_score', 'mean_val_binary_f1_score', 'std_val_binary_f1_score',
                'median_val_binary_accuracy', 'mean_val_binary_accuracy', 'std_val_binary_accuracy',
            ]
            existing_cols = config_df.columns.tolist()
            # Get config parameter columns (usually start with 'cfg.')
            config_param_cols = sorted([col for col in existing_cols if col.startswith('cfg.')])
            # Get remaining columns not in leading or config params
            other_cols = sorted([col for col in existing_cols if col not in leading_cols and col not in config_param_cols])

            # Build final column order, only including columns that actually exist in the dataframe
            final_cols = [col for col in leading_cols if col in existing_cols] + \
                         [col for col in other_cols if col in existing_cols] + \
                         [col for col in config_param_cols if col in existing_cols]

            config_df = config_df[final_cols]
            config_df.to_csv(args.config_csv, index=False, float_format='%.5f')
            print(f"  Config/Aggregated Metrics saved: {args.config_csv}")
         except Exception as e: print(f"  Error writing {args.config_csv}: {e}", file=sys.stderr)
    else: print(f"  No config/aggregated metrics data generated for {args.config_csv}.")


    # Leaked IDs (if any)
    if all_leaked_ids_data:
         try:
             pd.DataFrame.from_records(all_leaked_ids_data).to_csv(args.leaked_ids_csv, index=False)
             print(f"  Detailed leaked IDs saved: {args.leaked_ids_csv}")
         except Exception as e: print(f"  Error writing {args.leaked_ids_csv}: {e}", file=sys.stderr)
    elif overall_leakage_found: print(f"  Warn: Leakage detected, but no detailed ID data captured. {args.leaked_ids_csv} not created.", file=sys.stderr)
    else: print(f"  No leaked IDs detected, {args.leaked_ids_csv} not created.")


    # Detailed Per-Fold Metrics and Stats
    if all_detailed_fold_data:
        try:
            detailed_df = pd.DataFrame.from_records(all_detailed_fold_data)
            # Define preferred order: file/fold info, then metrics, then stats
            metric_cols = [ # Order matters for readability
                'auroc', 'ap', 'precision', 'recall', 'f1_score', 'accuracy', 'tn', 'fp', 'fn', 'tp',
                'multi_auroc', 'multi_ap', 'multi_precision', 'multi_recall', 'multi_f1_score', 'multi_accuracy',
                'binary_auroc', 'binary_ap', 'binary_precision', 'binary_recall', 'binary_f1_score', 'binary_accuracy', 'binary_tn', 'binary_fp', 'binary_fn', 'binary_tp',
            ]
            # Dynamically find stat columns (contain _patients_ or _windows_)
            stat_cols = sorted([col for col in detailed_df.columns if '_patients_' in col or '_windows_' in col])

            leading_info_cols = ['filename', 'fold_number']
            existing_cols = detailed_df.columns.tolist()
            other_cols = sorted([col for col in existing_cols if col not in leading_info_cols and col not in metric_cols and col not in stat_cols]) # Catch any unexpected columns

            # Build final order, including only existing columns
            final_detailed_cols = [col for col in leading_info_cols if col in existing_cols] + \
                                  [col for col in metric_cols if col in existing_cols] + \
                                  [col for col in stat_cols if col in existing_cols] + \
                                  [col for col in other_cols if col in existing_cols]

            detailed_df = detailed_df[final_detailed_cols]
            detailed_df.to_csv(args.detailed_metrics_csv, index=False, float_format='%.5f')
            print(f"  Detailed per-fold metrics & stats saved: {args.detailed_metrics_csv}")
        except Exception as e:
            print(f"  Error writing {args.detailed_metrics_csv}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr) # More detail on error
    else:
        print(f"  No detailed per-fold metrics or stats data generated for {args.detailed_metrics_csv}.")


    # Final Summary
    print("\n\n===== Overall Check Summary =====")
    print(f"Files provided: {len(args.pklz_files)}")
    print(f"Files processed successfully: {files_processed}")
    print(f"Files with leakage detected: {files_with_leakage}")
    if SKLEARN_AVAILABLE: print("Attempted performance metrics & data stats calculation.")
    else: print("Performance metrics & data stats NOT calculated (scikit-learn unavailable).")
    if args.binary_from_multi: print("Derived binary metrics were calculated for multiclass tasks where possible.")


    if not overall_leakage_found and files_processed > 0:
        print("\n✅ SUCCESS: No patient ID leakage detected.")
        print(f"Check CSVs for config, metrics, and detailed stats.")
        sys.exit(0)
    elif files_processed == 0:
        print("\n⚠️ Warning: No files were processed (check paths/permissions).")
        sys.exit(2)
    else:
        print("\n❌ ERROR: Patient ID leakage DETECTED in one or more files OR critical errors occurred during processing.")
        print(f"Review the generated CSV files for details:")
        if os.path.exists(args.summary_csv): print(f"  - Leakage Summary (Per Fold): {args.summary_csv}")
        if os.path.exists(args.config_csv): print(f"  - Configs & Aggregated Metrics (Per File): {args.config_csv}")
        if os.path.exists(args.detailed_metrics_csv): print(f"  - Detailed Fold Metrics & Stats: {args.detailed_metrics_csv}")
        if os.path.exists(args.leaked_ids_csv): print(f"  - Leaked Patient IDs: {args.leaked_ids_csv}")
        sys.exit(1)