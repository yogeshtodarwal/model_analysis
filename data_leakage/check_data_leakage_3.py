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

# --- Imports (Keep as before) ---
try:
    from utils_tbox.utils_tbox import read_pklz, decompress_obj
except ImportError:
    print("Error: Could not import 'read_pklz' and 'decompress_obj' from 'utils_tbox.utils_tbox'.", file=sys.stderr)
    sys.exit(1)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
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
        if isinstance(v, MutableMapping): items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple, set)): items.append((new_key, str(v)))
        else: items.append((new_key, v))
    return dict(items)

# ==============================================================================
# REVISED Helper Function for Metric Calculation (More Metrics)
# ==============================================================================
def calculate_all_metrics(y_true, y_pred_probs, task_type='binary', target_name='unknown'):
    """Calculates AUROC, AP, Precision, Recall, F1, Accuracy."""
    metrics = {
        'auroc': np.nan, 'ap': np.nan,
        'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'accuracy': np.nan
    }
    if not SKLEARN_AVAILABLE: return metrics

    try:
        y_true_np = np.asarray(y_true)
        y_pred_probs_np = np.asarray(y_pred_probs)
        if y_pred_probs_np.ndim == 1: y_pred_probs_np = y_pred_probs_np.reshape(-1, 1)

        valid_idx_true = ~np.isnan(y_true_np)
        valid_idx_pred = ~np.isnan(y_pred_probs_np).any(axis=1)
        valid_idx = valid_idx_true & valid_idx_pred

        if not np.any(valid_idx): return metrics
        y_true_filt = y_true_np[valid_idx]
        y_pred_probs_filt = y_pred_probs_np[valid_idx, :]

        unique_classes_true = np.unique(y_true_filt)
        if len(unique_classes_true) < 2: return metrics # Cannot calculate most metrics

    except Exception as e: print(f"      Error preparing data for metrics [{target_name}]: {e}", file=sys.stderr); return metrics

    # --- AUROC and AP (from probabilities) ---
    try:
        if task_type == 'binary':
            if y_pred_probs_filt.shape[1] != 1: print(f"      Warn [{target_name}]: Binary AUROC/AP expects 1 pred col, got {y_pred_probs_filt.shape[1]}. Using first.", file=sys.stderr)
            metrics['auroc'] = roc_auc_score(y_true_filt, y_pred_probs_filt[:, 0])
            metrics['ap'] = average_precision_score(y_true_filt, y_pred_probs_filt[:, 0])
        elif task_type == 'multiclass':
            if y_pred_probs_filt.shape[1] < 2 : raise ValueError("Multiclass AUROC/AP expects >= 2 pred cols")
            metrics['auroc'] = roc_auc_score(y_true_filt, y_pred_probs_filt, multi_class='ovr', average='weighted')
            # AP for multiclass often handled per class, 'weighted' might be available in newer sklearn but not universally.
            # For simplicity, keeping AP as NaN for multiclass or using a placeholder.
            try:
                metrics['ap'] = average_precision_score(y_true_filt, y_pred_probs_filt, average='weighted')
            except TypeError: # Older sklearn might not support weighted AP for multiclass
                metrics['ap'] = np.nan
                # print(f"      Note [{target_name}]: Weighted AP for multiclass not supported by this sklearn version. AP set to NaN.", file=sys.stderr)

    except ValueError as ve: print(f"      Warn [{target_name}]: AUROC/AP calc failed: {ve}", file=sys.stderr)
    except Exception as e: print(f"      Error calc AUROC/AP [{target_name}]: {e}", file=sys.stderr)

    # --- Threshold-based Metrics (Precision, Recall, F1, Accuracy) ---
    # Need to convert probabilities to class predictions
    y_pred_class = None
    if task_type == 'binary':
        # Standard threshold for binary classification from probabilities
        threshold = 0.5
        y_pred_class = (y_pred_probs_filt[:, 0] >= threshold).astype(int)
    elif task_type == 'multiclass':
        # Highest probability determines the class
        y_pred_class = np.argmax(y_pred_probs_filt, axis=1)

    if y_pred_class is not None:
        try:
            # For precision, recall, f1, use zero_division=0 to avoid warnings and set to 0.0
            # This is important if a class has no true or predicted samples.
            avg_type = 'binary' if task_type == 'binary' else 'weighted'
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_filt, y_pred_class, average=avg_type, zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            metrics['accuracy'] = accuracy_score(y_true_filt, y_pred_class)
        except ValueError as ve: print(f"      Warn [{target_name}]: Prec/Rec/F1/Acc calc failed: {ve}", file=sys.stderr)
        except Exception as e: print(f"      Error calc Prec/Rec/F1/Acc [{target_name}]: {e}", file=sys.stderr)

    return metrics

# ==============================================================================
# REVISED Function to Parse Prediction Columns (Handle new binary case)
# ==============================================================================
def parse_prediction_info(df_columns):
    """
    Analyzes prediction columns to infer model name, task structure, and target columns.
    """
    pred_col_pattern = re.compile(r"^pred__([^_].*?)__(.+)$")
    predictions = defaultdict(list)
    found_match = False
    for col in df_columns:
        match = pred_col_pattern.match(col)
        if match:
            found_match = True
            try: model_name, description = match.groups(); predictions[model_name].append({'full_col': col, 'desc': description})
            except ValueError: pass

    if not found_match:
        # print("      DEBUG: No columns matched prediction regex pattern: ^pred__([^_].*?)__(.+)$")
        return None

    if len(predictions) > 1: model_name = max(predictions, key=lambda k: len(predictions[k])); print(f"      Warn: Multiple models found. Using '{model_name}'.", file=sys.stderr)
    else: model_name = list(predictions.keys())[0]
    model_predictions = predictions[model_name]
    model_predictions.sort(key=lambda p: p['desc']) # Sort for consistency
    descriptions = [p['desc'] for p in model_predictions]
    full_pred_cols = [p['full_col'] for p in model_predictions]

    task_info = {
        'model_name': model_name, 'pred_cols': full_pred_cols, 'pred_descriptions': descriptions,
        'required_target_bases': [], 'task_type': None, 'class_mapping': None
    }

    positive_target_pattern = re.compile(r"^target__(\w+)$")
    actual_target_bases = set()
    has_negation = False
    for desc in descriptions:
        match = positive_target_pattern.match(desc)
        if match: actual_target_bases.add(match.group(1))
        if desc.startswith('not_target') or desc.startswith('no_target') or desc.lower() == 'none' or desc.lower() == 'other' or '_neg' in desc:
            has_negation = True

    task_info['required_target_bases'] = sorted(list(actual_target_bases))
    num_descs = len(descriptions)
    num_actual_targets = len(actual_target_bases)

    # --- Inference Logic ---
    # Case 1: Single prediction column, description IS the target name (e.g., "los_or_abdominal_nec")
    if num_descs == 1 and num_actual_targets == 0: # No "target__" prefix in description
        # Assume the description itself is the base target name (without "target__")
        # Example: pred_col description is 'los_or_abdominal_nec'
        # We need the y_true column to be 'target__los_or_abdominal_nec'
        target_base_from_desc = descriptions[0]
        task_info['required_target_bases'] = [target_base_from_desc] # This will be checked for target__{desc}
        task_info['task_type'] = 'binary'
        print(f"      Inferred BINARY task (single pred col, direct desc) for target '{target_base_from_desc}'")

    elif num_descs == 1 and num_actual_targets == 1: # Single pred, e.g., pred__model__target_death
        task_info['task_type'] = 'binary'
        print(f"      Inferred BINARY task (single pred col) for target '{task_info['required_target_bases'][0]}'")

    elif num_descs == 2 and num_actual_targets == 1 and has_negation: # target__A, not_target__A
         task_info['task_type'] = 'binary'
         pos_desc = f"target__{task_info['required_target_bases'][0]}"
         pos_col = next((p['full_col'] for p in model_predictions if p['desc'] == pos_desc), None)
         if pos_col:
             task_info['pred_cols'] = [pos_col]
             print(f"      Inferred BINARY task (explicit pos/neg) for target '{task_info['required_target_bases'][0]}'. Using positive pred: {pos_col}")
         else: print(f"      Warn: Could not find positive pred col for binary task. Desc='{pos_desc}'. Skip.", file=sys.stderr); return None

    elif num_descs > 1 and num_actual_targets >= 1: # Multiclass
         task_info['task_type'] = 'multiclass'
         task_info['class_mapping'] = {desc: i for i, desc in enumerate(descriptions)}
         print(f"      Inferred MULTICLASS task involving targets {task_info['required_target_bases']} from {num_descs} prediction columns.")

    else:
         print(f"      Warning: Could not reliably infer task structure: Descs={descriptions}, ActualTargets={actual_target_bases}. Skip.", file=sys.stderr)
         return None

    # Final Validation: Check for existence of required ACTUAL target columns
    missing_targets_truth_cols = []
    if not task_info['required_target_bases']: # Should not happen if inference is good
        print(f"      Warning: Task inference succeeded but no required target bases identified. This is a bug. Skip.", file=sys.stderr)
        return None

    for target_base in task_info['required_target_bases']:
        target_col_name = f"target__{target_base}" # Construct full target column name
        if target_col_name not in df_columns:
             missing_targets_truth_cols.append(target_col_name)
    if missing_targets_truth_cols:
         print(f"      Warning: Inferred task requires missing truth column(s): {missing_targets_truth_cols}. Skip.", file=sys.stderr)
         return None

    print(f"      Task inference successful: {task_info['task_type']}")
    return task_info


# --- construct_multiclass_y_true (Keep as before) ---
def construct_multiclass_y_true(df, task_info):
    # ... (previous implementation) ...
    if task_info['task_type'] != 'multiclass': return None
    n_samples = len(df); y_true = np.full(n_samples, -1, dtype=int)
    class_mapping = task_info['class_mapping']; descriptions = task_info['pred_descriptions']
    neg_class_index = -1
    for desc, index in class_mapping.items():
        if 'not_target' in desc or 'no_target' in desc or desc.lower() == 'none' or desc.lower() == 'other' or '_neg' in desc:
             if neg_class_index != -1: pass # print(f"      Warn: Multiple neg classes. Using first.", file=sys.stderr)
             else: neg_class_index = index
    positive_target_pattern = re.compile(r"^target__(\w+)$"); assigned_positive = np.zeros(n_samples, dtype=bool)
    for desc, class_index in class_mapping.items():
        if class_index == neg_class_index: continue
        match = positive_target_pattern.match(desc)
        if match:
            target_base = match.group(1); target_col = f"target__{target_base}"
            if target_col not in df.columns: print(f"      Error: construct_y_true needs '{target_col}' but missing.", file=sys.stderr); return None
            assign_mask = (df[target_col] == 1) & (~assigned_positive) & (y_true == -1)
            y_true[assign_mask] = class_index; assigned_positive[assign_mask] = True
    if neg_class_index != -1: y_true[y_true == -1] = neg_class_index
    elif np.any(y_true == -1): lowest_index = min(class_mapping.values()); print(f"      Warn: Unassigned samples, no clear neg class. Assigning lowest idx {lowest_index}.", file=sys.stderr); y_true[y_true == -1] = lowest_index
    if np.any(y_true == -1): print(f"      Error: Failed to assign all samples class in y_true construction.", file=sys.stderr); return None
    return y_true


# ==============================================================================
# REVISED Main Leakage Check Logic (Store per-fold metrics)
# ==============================================================================
def check_leakage_in_file(pklz_filepath):
    print(f"\n--- Checking File: {pklz_filepath} ---")
    file_leakage_summary_records = []
    file_leaked_ids_records = []
    file_config_record = {'filename': os.path.basename(pklz_filepath)}
    file_has_leakage = False
    overall_cfg = {}

    # --- NEW: Store per-fold metrics for detailed CSV ---
    per_fold_detailed_metrics = []
    # --- Accumulators for median calculation ---
    fold_aurocs, fold_aps, fold_precisions, fold_recalls, fold_f1s, fold_accuracies = [], [], [], [], [], []

    metrics_task_description = "N/A"
    consistent_task_info_across_folds = None

    try: data = read_pklz(pklz_filepath)
    except Exception as e: print(f"Error reading {pklz_filepath}: {e}", file=sys.stderr); return False, [], [], {}, [] # Added empty list for per_fold_metrics

    # Config extraction... (as before)
    if isinstance(data, dict) and 'cfg' in data and isinstance(data['cfg'], dict):
        overall_cfg = data['cfg']
        try: file_config_record.update(flatten_dict(overall_cfg))
        except Exception as e: print(f"  Warning: Could not flatten 'cfg': {e}", file=sys.stderr)

    # Results validation... (as before)
    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        print(f"Error: Invalid/missing 'results' list in {pklz_filepath}.", file=sys.stderr)
        file_config_record.update({'metrics_task': 'Error: Invalid results', 'median_val_auroc': np.nan, 'median_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': 0})
        return False, [], [], file_config_record, []

    compressed_datasets = data["results"]
    if not compressed_datasets:
        print(f"  Info: 'results' list is empty.")
        file_config_record.update({'metrics_task': 'No folds', 'median_val_auroc': np.nan, 'median_val_ap': np.nan, 'num_folds_metrics_calculated': 0, 'num_total_folds': 0})
        return False, [{'filename': os.path.basename(pklz_filepath),'fold_number': 'N/A','leakage_detected': False,'leaked_id_count': 0}], [], file_config_record, []

    total_folds = len(compressed_datasets)
    print(f"  Found {total_folds} fold(s) in 'results'.")

    for i, compressed_fold_data in enumerate(compressed_datasets):
        fold_num = i + 1
        print(f"   --- Processing Fold {fold_num}/{total_folds} ---")
        fold_summary_record = {'filename': os.path.basename(pklz_filepath),'fold_number': fold_num,'leakage_detected': False,'leaked_id_count': 0}
        fold_metrics_record = {'filename': os.path.basename(pklz_filepath), 'fold_number': fold_num} # For detailed CSV

        try:
            fold_data = decompress_obj(compressed_fold_data)
            # ... (Error handling for fold_data, train_df, val_df as before) ...
            if fold_data is None or not isinstance(fold_data, dict): fold_summary_record['error'] = 'Decompress Error'; file_leakage_summary_records.append(fold_summary_record); per_fold_detailed_metrics.append(fold_metrics_record); continue
            val_df = fold_data.get("val"); train_df = fold_data.get("train")
            if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame): fold_summary_record['error'] = 'Invalid DF type'; file_leakage_summary_records.append(fold_summary_record); per_fold_detailed_metrics.append(fold_metrics_record); continue

            # Leakage Check (as before)
            fold_leakage_detected, leaked_count_for_fold = False, 0
            if "ids__uid" not in train_df.columns or "ids__uid" not in val_df.columns: pass # print(f"    Warn: Missing 'ids__uid'. Leakage skip.", file=sys.stderr)
            else:
                train_ids = set(train_df['ids__uid'].dropna().unique()); val_ids = set(val_df['ids__uid'].dropna().unique())
                intersection = train_ids.intersection(val_ids)
                if intersection:
                     fold_leakage_detected = True; file_has_leakage = True
                     leak_list = sorted(list(intersection)); leaked_count_for_fold = len(leak_list)
                     print(f"    Leakage DETECTED Fold {fold_num}! ({leaked_count_for_fold} IDs)")
                     for leaked_id in leak_list: file_leaked_ids_records.append({'filename': os.path.basename(pklz_filepath),'fold_number': fold_num,'leaked_ids__uid': leaked_id})
            fold_summary_record['leakage_detected'] = fold_leakage_detected; fold_summary_record['leaked_id_count'] = leaked_count_for_fold
            file_leakage_summary_records.append(fold_summary_record)

            # Performance Metrics
            if not SKLEARN_AVAILABLE: per_fold_detailed_metrics.append(fold_metrics_record); continue
            if val_df.empty: per_fold_detailed_metrics.append(fold_metrics_record); continue

            task_info = parse_prediction_info(val_df.columns)
            if task_info is None:
                print(f"    Skipping metrics Fold {fold_num} (task inference failed/cols missing).")
                per_fold_detailed_metrics.append(fold_metrics_record) # Add empty record for this fold
                continue

            if consistent_task_info_across_folds is None:
                consistent_task_info_across_folds = task_info
                req_targets = '/'.join(task_info['required_target_bases']) if task_info['required_target_bases'] else 'Unknown'
                metrics_task_description = f"{task_info['task_type']} ({req_targets})"

            y_true = None; y_pred_probs = val_df[task_info['pred_cols']]
            metrics_label = metrics_task_description

            if task_info['task_type'] == 'binary':
                 target_base = task_info['required_target_bases'][0]
                 y_true = val_df[f"target__{target_base}"]
                 metrics_label = f"Binary ({target_base})"
            elif task_info['task_type'] == 'multiclass':
                 y_true = construct_multiclass_y_true(val_df, task_info)
                 if y_true is None: per_fold_detailed_metrics.append(fold_metrics_record); continue
                 metrics_label = f"Multi ({'/'.join(task_info['required_target_bases'])})"

            if y_true is not None:
                 current_fold_metrics = calculate_all_metrics(y_true, y_pred_probs, task_info['task_type'], target_name=metrics_label)
                 print(f"      Fold {fold_num} Metrics [{metrics_label}]: AUROC={current_fold_metrics.get('auroc', np.nan):.4f}, AP={current_fold_metrics.get('ap', np.nan):.4f}, F1={current_fold_metrics.get('f1_score', np.nan):.4f}")
                 fold_metrics_record.update(current_fold_metrics) # Add all metrics to the fold's record

                 # Append to lists for median calculation if valid
                 if not np.isnan(current_fold_metrics.get('auroc')): fold_aurocs.append(current_fold_metrics['auroc'])
                 if not np.isnan(current_fold_metrics.get('ap')): fold_aps.append(current_fold_metrics['ap'])
                 if not np.isnan(current_fold_metrics.get('precision')): fold_precisions.append(current_fold_metrics['precision'])
                 if not np.isnan(current_fold_metrics.get('recall')): fold_recalls.append(current_fold_metrics['recall'])
                 if not np.isnan(current_fold_metrics.get('f1_score')): fold_f1s.append(current_fold_metrics['f1_score'])
                 if not np.isnan(current_fold_metrics.get('accuracy')): fold_accuracies.append(current_fold_metrics['accuracy'])
            per_fold_detailed_metrics.append(fold_metrics_record) # Add even if some metrics are NaN

        except Exception as e:
             print(f"    Unexpected error processing Fold {fold_num}: {e}. Skipping fold.", file=sys.stderr)
             import traceback; traceback.print_exc(file=sys.stderr)
             fold_summary_record['error'] = f"Unexpected: {e}"
             if fold_summary_record not in file_leakage_summary_records: file_leakage_summary_records.append(fold_summary_record)
             per_fold_detailed_metrics.append(fold_metrics_record) # Ensure a record for this fold

    # --- Aggregate Metrics using MEDIAN ---
    num_folds_metrics_calculated = len(fold_aurocs) # Use AUROC list as reference for count
    agg_metrics = {
        'median_val_auroc': np.nan, 'median_val_ap': np.nan,
        'median_val_precision': np.nan, 'median_val_recall': np.nan,
        'median_val_f1_score': np.nan, 'median_val_accuracy': np.nan,
        # Optionally keep mean/std if needed
        'mean_val_auroc': np.nan, 'std_val_auroc': np.nan
    }

    if num_folds_metrics_calculated > 0:
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning) # For std of single element
             agg_metrics['median_val_auroc'] = np.median(fold_aurocs) if fold_aurocs else np.nan
             agg_metrics['median_val_ap'] = np.median(fold_aps) if fold_aps else np.nan
             agg_metrics['median_val_precision'] = np.median(fold_precisions) if fold_precisions else np.nan
             agg_metrics['median_val_recall'] = np.median(fold_recalls) if fold_recalls else np.nan
             agg_metrics['median_val_f1_score'] = np.median(fold_f1s) if fold_f1s else np.nan
             agg_metrics['median_val_accuracy'] = np.median(fold_accuracies) if fold_accuracies else np.nan
             # Mean/Std for AUROC (example)
             agg_metrics['mean_val_auroc'] = np.mean(fold_aurocs) if fold_aurocs else np.nan
             agg_metrics['std_val_auroc'] = np.std(fold_aurocs) if fold_aurocs else np.nan

        print(f"  Aggregated Val Metrics (Median) [{metrics_task_description}] ({num_folds_metrics_calculated}/{total_folds} folds):")
        print(f"    Median AUROC: {agg_metrics['median_val_auroc']:.4f}, Median F1: {agg_metrics['median_val_f1_score']:.4f}")
    else:
        if SKLEARN_AVAILABLE and total_folds > 0: print(f"  No valid metrics calculated across folds.")

    file_config_record.update({'metrics_task': metrics_task_description, 'num_folds_metrics_calculated': num_folds_metrics_calculated, 'num_total_folds': total_folds})
    file_config_record.update(agg_metrics) # Add all aggregated metrics

    if not file_has_leakage and files_processed > 0: print(f"  SUCCESS: No leakage in {pklz_filepath}.")
    return file_has_leakage, file_leakage_summary_records, file_leaked_ids_records, file_config_record, per_fold_detailed_metrics


# --- Main Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check leakage, infer task, calculate metrics from PKLZ files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pklz_files", type=str, nargs='+', help="Path(s) to .pklz file(s).")
    parser.add_argument("--summary-csv", type=str, default="master_leakage_summary.csv", help="Leakage summary CSV.")
    parser.add_argument("--config-csv", type=str, default="master_config_summary.csv", help="Config and aggregated (median) metrics summary CSV.")
    parser.add_argument("--leaked-ids-csv", type=str, default="detailed_leaked_ids.csv", help="Detailed leaked IDs CSV.")
    # --- NEW ARGUMENT for detailed metrics ---
    parser.add_argument("--detailed-metrics-csv", type=str, default="detailed_fold_metrics.csv", help="Detailed per-fold performance metrics CSV.")
    args = parser.parse_args()

    all_leakage_summary_data, all_config_summary_data, all_leaked_ids_data = [], [], []
    all_detailed_fold_metrics = [] # Accumulator for the new detailed metrics
    overall_leakage_found, files_processed, files_with_leakage = False, 0, 0

    print(f"Checking {len(args.pklz_files)} file path(s)...")
    print("Attempting to infer task and calculate metrics (median aggregation).")

    for filepath in args.pklz_files:
        if not os.path.isfile(filepath): print(f"\nWarn: Skip invalid path: {filepath}", file=sys.stderr); continue
        files_processed += 1
        try:
            leakage_found, summary, leaked_ids, config, detailed_metrics_for_file = check_leakage_in_file(filepath) # Unpack new list
            all_leakage_summary_data.extend(summary)
            all_leaked_ids_data.extend(leaked_ids)
            if config: all_config_summary_data.append(config)
            all_detailed_fold_metrics.extend(detailed_metrics_for_file) # Aggregate detailed metrics
            if leakage_found: overall_leakage_found = True; files_with_leakage += 1
        except Exception as e:
            print(f"\nCRITICAL Error processing {filepath}: {e}", file=sys.stderr); import traceback; traceback.print_exc(file=sys.stderr)
            all_config_summary_data.append({'filename': os.path.basename(filepath), 'processing_error': str(e), 'metrics_task': 'CRITICAL ERROR'})

    print("\n--- Writing CSV Reports ---")
    # Leakage Summary (as before)
    if all_leakage_summary_data:
        try:
            summary_df = pd.DataFrame.from_records(all_leakage_summary_data); summary_cols = ['filename', 'fold_number', 'leakage_detected', 'leaked_id_count', 'error']
            summary_df = summary_df[[col for col in summary_cols if col in summary_df.columns]]; summary_df.to_csv(args.summary_csv, index=False)
            print(f"  Leakage summary: {args.summary_csv}")
        except Exception as e: print(f"  Error writing {args.summary_csv}: {e}", file=sys.stderr)

    # Config/Aggregated Metrics Summary (stores medians now)
    if all_config_summary_data:
         try:
            config_df = pd.DataFrame.from_records(all_config_summary_data)
            leading_cols = ['filename', 'metrics_task',
                            'median_val_auroc', 'median_val_ap', 'median_val_precision',
                            'median_val_recall', 'median_val_f1_score', 'median_val_accuracy',
                            'mean_val_auroc', 'std_val_auroc', # Example of keeping mean/std for one metric
                            'num_folds_metrics_calculated', 'num_total_folds', 'processing_error']
            existing_cols = config_df.columns.tolist()
            other_cols = sorted([col for col in existing_cols if col not in leading_cols])
            final_cols = [col for col in leading_cols if col in existing_cols] + [col for col in other_cols if col not in leading_cols]
            config_df = config_df[final_cols]; config_df.to_csv(args.config_csv, index=False, float_format='%.5f')
            print(f"  Config/Aggregated (Median) Metrics: {args.config_csv}")
         except Exception as e: print(f"  Error writing {args.config_csv}: {e}", file=sys.stderr)

    # Leaked IDs (as before)
    if all_leaked_ids_data:
         try: pd.DataFrame.from_records(all_leaked_ids_data).to_csv(args.leaked_ids_csv, index=False); print(f"  Detailed leaked IDs: {args.leaked_ids_csv}")
         except Exception as e: print(f"  Error writing {args.leaked_ids_csv}: {e}", file=sys.stderr)
    elif overall_leakage_found: print(f"  Warn: Leakage detected, but no detailed ID data. {args.leaked_ids_csv} not created.", file=sys.stderr)

    # --- NEW: Detailed Per-Fold Metrics CSV ---
    if all_detailed_fold_metrics:
        try:
            detailed_metrics_df = pd.DataFrame.from_records(all_detailed_fold_metrics)
            # Define expected order, ensure filename and fold are first
            metric_cols_order = ['filename', 'fold_number', 'auroc', 'ap', 'precision', 'recall', 'f1_score', 'accuracy']
            # Include any other metrics that might be added to calculate_all_metrics
            other_metric_cols = sorted([col for col in detailed_metrics_df.columns if col not in metric_cols_order])
            final_metric_cols = [col for col in metric_cols_order if col in detailed_metrics_df.columns] + \
                                [col for col in other_metric_cols if col in detailed_metrics_df.columns and col not in metric_cols_order]
            detailed_metrics_df = detailed_metrics_df[final_metric_cols]
            detailed_metrics_df.to_csv(args.detailed_metrics_csv, index=False, float_format='%.5f')
            print(f"  Detailed per-fold metrics: {args.detailed_metrics_csv}")
        except Exception as e:
            print(f"  Error writing {args.detailed_metrics_csv}: {e}", file=sys.stderr)
    else:
        print(f"  No detailed per-fold metrics data generated for {args.detailed_metrics_csv}.")


    # Final Summary (as before)
    print("\n\n===== Overall Check Summary =====") # (rest of summary as before)
    print(f"Files provided: {len(args.pklz_files)}"); print(f"Files processed: {files_processed}"); print(f"Files with leakage detected: {files_with_leakage}")
    if SKLEARN_AVAILABLE: print("Attempted performance metrics calculation (median aggregation).")
    else: print("Performance metrics NOT calculated (scikit-learn unavailable).")
    if not overall_leakage_found and files_processed > 0:
        print("\nSUCCESS: No patient ID leakage detected."); print(f"Check CSVs for config/metrics."); sys.exit(0)
    elif files_processed == 0: print("\nWarning: No files were processed."); sys.exit(2)
    else:
        print("\nERROR: Patient ID leakage DETECTED or critical errors occurred."); print(f"Review CSV files:");
        if os.path.exists(args.summary_csv): print(f"  - Leakage Summary: {args.summary_csv}")
        if os.path.exists(args.config_csv): print(f"  - Configs & Aggregated Metrics: {args.config_csv}")
        if os.path.exists(args.detailed_metrics_csv): print(f"  - Detailed Fold Metrics: {args.detailed_metrics_csv}")
        if os.path.exists(args.leaked_ids_csv): print(f"  - Leaked IDs: {args.leaked_ids_csv}")
        sys.exit(1)