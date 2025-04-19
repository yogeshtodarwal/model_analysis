import os
import gc
import socket
import argparse
from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from datetime import timedelta
from functools import partial
from parse import parse
from multiprocessing import Pool
import sys
import warnings

# Plotting Libraries
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages

# Scikit-learn Metrics
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, roc_auc_score, precision_recall_fscore_support,
    average_precision_score, multilabel_confusion_matrix
)

# Custom Utils (Assume these are in the PYTHONPATH or same directory)
# If not, adjust the import paths as needed
try:
    from utils_tbox.utils_tbox import read_pklz, decompress_obj, write_pklz
    from utils_plots.utils_plots import plot_ci, better_lookin, linestyle_tuple
    # from utils_models.nf import NFLOWclassifier, parse_nflow_name # Likely not needed for analysis script if models aren't re-run/sampled
    from utils_results_analysis.plots import plot_roc
    from utils_results_analysis.utils_results_analysis import custom_confusion_matrices, topkmulticlass # Ensure custom_confusion_matrices is correct
except ImportError as e:
    print(f"Warning: Could not import custom utility modules: {e}")
    print("Please ensure 'utils_tbox', 'utils_plots', 'utils_models', 'utils_results_analysis' are accessible.")
    # Define dummy functions if needed for the script to run partially
    def read_pklz(f):
        with open(f, 'rb') as file:
            # Try decompressing first if it's a compressed pickle
            try:
                import zlib
                return pkl.loads(zlib.decompress(file.read()))
            except: # Fallback to standard pickle
                file.seek(0)
                return pkl.load(file)

    def write_pklz(f, obj):
        import zlib
        with open(f, 'wb') as file:
            file.write(zlib.compress(pkl.dumps(obj)))

    def decompress_obj(obj):
        # Assuming obj is already decompressed if read_pklz handles it
        # Or if it's passed as compressed bytes
        try:
            import zlib
            return pkl.loads(zlib.decompress(obj))
        except:
            return obj # Return as is if not compressed bytes

    def plot_ci(ax, data, **kwargs): ax.plot(data, **kwargs) # Dummy
    def better_lookin(ax, **kwargs): ax.grid(True) # Dummy
    linestyle_tuple = [(0, ())] # Dummy
    def plot_roc(ax, results, title, **kwargs): print(f"Plotting ROC for {title}") # Dummy
    # Redefine custom_confusion_matrices based on the notebook code
    # Corrected custom_confusion_matrices function definition
    def custom_confusion_matrices(y_true, y_pred, labels, only_false=False, score=False,topk=None, criteria="exact_match",larger_label_space=False):
        # --- Function body starts here ---
        n_classes = y_pred.shape[1] #if not larger_label_space else 2

        # Check if y_pred is already binary or probabilities
        if y_pred.dtype != bool and not np.all((y_pred == 0) | (y_pred == 1)):
            # Assuming probabilities, apply threshold
            thresh = 1 / n_classes if n_classes > 0 else 0.5
            y_pred_bin = y_pred >= thresh
        else:
            y_pred_bin = y_pred.astype(bool) # Ensure boolean

        # Ensure y_true is boolean
        if y_true.dtype != bool and not np.all((y_true == 0) | (y_true == 1)):
            # If y_true contains probabilities/scores, threshold it (e.g., at 0.5)
            # Adjust this threshold if your true labels are encoded differently
            y_true_bin = y_true >= 0.5
        else:
            y_true_bin = y_true.astype(bool)


        label_leak = []
        scores=[]
        for i in range(n_classes):
            idx_true_i = y_true_bin[:,i] == True # Use boolean indexing
            n_class_i = idx_true_i.sum()

            if n_class_i == 0:
                label_leak.append([0] * n_classes)
                if score: scores.append(np.nan) # Or 0? NaN indicates no samples for this class
                continue

            # Get predictions for samples truly belonging to class i
            y_pred_bin_i = y_pred_bin[idx_true_i]
            y_true_bin_i = y_true_bin[idx_true_i]

            # False classifications for class i are the rows where not all labels are equal
            if not larger_label_space:
                # Count mismatches for samples where true label is i
                # Ensure comparison happens correctly even with multi-label truths
                # A correct classification means the prediction matches the true label *for that specific class i pattern*
                # Simplified: check if the prediction row vector matches the true row vector
                is_correct_classif = np.all(y_pred_bin_i == y_true_bin_i, axis=1)
                idx_false_classif_i = ~is_correct_classif
                #print(f"Class {i}: n_class_i={n_class_i}, n_false={idx_false_classif_i.sum()}")
            else:
                # In the case where the prediction was binary (e.g., class 0 vs any other class)
                # Compare relevant columns (e.g., column 0 and column 1 of pred vs col 0 and col i of true)
                pred_cols_to_compare = y_pred_bin_i[:, [0, 1]] # Assumes binary prediction is always [class 0, other]
                # Need to handle cases where i might be out of bounds for y_true_bin_i if shapes differ unexpectedly
                if i < y_true_bin_i.shape[1]:
                    true_cols_to_compare = y_true_bin_i[:, [0, i]]
                    is_correct_classif = np.all(pred_cols_to_compare == true_cols_to_compare, axis=1)
                    idx_false_classif_i = ~is_correct_classif
                else:
                    # Handle error: class index i is out of bounds for true labels
                    print(f"Warning: Class index {i} out of bounds for true labels in larger_label_space mode.")
                    idx_false_classif_i = np.ones(y_pred_bin_i.shape[0], dtype=bool) # Mark all as false classification

                #print(f"Class {i} (Larger Space): n_class_i={n_class_i}, n_false={idx_false_classif_i.sum()}")


            if only_false:
                # Data to sum: predictions for the falsely classified samples of class i
                data_to_sum=y_pred_bin_i[idx_false_classif_i]
            else:
                # Data to sum: all predictions for samples of class i
                data_to_sum=y_pred_bin_i

            if score:
                if topk is None:
                    # Standard Accuracy for this class
                    # Handle n_class_i = 0 case above
                    accuracy_i = (n_class_i - idx_false_classif_i.sum()) / n_class_i if n_class_i > 0 else np.nan
                    scores.append(accuracy_i)
                else:
                    # This part needs the original implementation of topkmulticlass
                    # Placeholder:
                    print(f"Warning: topkmulticlass not implemented - skipping Top-k score for class {i}")
                    # Attempt to call if available, otherwise append NaN
                    try:
                        # Ensure inputs to topkmulticlass are appropriate (e.g., scores, not binary)
                        # The notebook passed `data_to_sum.astype(float)` which might be just 0s/1s
                        # This likely needs the original probabilities/scores, not y_pred_bin_i
                        # Reverting to placeholder as the required inputs aren't clear here
                        # score_val = topkmulticlass(y_true[idx_true_i],
                        #                            y_pred[idx_true_i][idx_false_classif_i if only_false else slice(None)], # Pass original scores
                        #                            k=topk, criteria=criteria)
                        # scores.append(score_val)
                        scores.append(np.nan) # Placeholder
                    except NameError:
                        scores.append(np.nan) # Placeholder if topkmulticlass is not defined

            # Sum the predicted labels for the selected subset (either all class i samples or only misclassified ones)
            if data_to_sum.shape[0] > 0: # Ensure there's data to sum
                label_leak.append(data_to_sum.sum(0).tolist())
            else:
                label_leak.append([0] * n_classes) # Append zeros if no data

        label_leak = np.array(label_leak)

        out_idx = [f"True={l}" for l in labels]
        # Adjust column names based on the actual shape of label_leak columns
        col_names = [f"Pred={l}" for l in labels[:label_leak.shape[1]]]


        # Ensure label_leak has the correct shape even if some classes had no samples
        # Check row count
        if label_leak.shape[0] < n_classes:
            full_leak = np.full((n_classes, label_leak.shape[1]), 0) # Match column count of calculated leak
            # Fill in the calculated rows - this needs careful index mapping if classes were skipped
            # Simplified: Assume rows correspond directly for now
            # This assumption might be wrong if classes were entirely missing from y_true
            rows_to_fill = min(label_leak.shape[0], n_classes)
            full_leak[:rows_to_fill, :] = label_leak[:rows_to_fill, :]
            label_leak = full_leak

        # Check column count (can happen if y_pred had fewer columns than labels list)
        if label_leak.shape[1] < len(labels):
            print(f"Warning: Mismatch between predicted columns ({label_leak.shape[1]}) and number of labels ({len(labels)}). Adjusting.")
            full_leak_cols = np.full((label_leak.shape[0], len(labels)), 0)
            cols_to_fill = min(label_leak.shape[1], len(labels))
            full_leak_cols[:, :cols_to_fill] = label_leak[:, :cols_to_fill]
            label_leak = full_leak_cols
            col_names = [f"Pred={l}" for l in labels] # Use full labels for columns now


        out = pd.DataFrame(data=label_leak.astype(int),
                            columns=col_names,
                            index=out_idx)

        if score:
            if (topk is None):
                score_name = "Accuracy" # Removed (%)
            else:
                score_name = f"Top-{topk}" # Removed (%)

            # Ensure scores array matches number of classes
            if len(scores) < n_classes:
                full_scores = np.full(n_classes, np.nan)
                # This assumes scores correspond directly to the first len(scores) classes
                full_scores[:len(scores)] = scores
                scores = full_scores
            elif len(scores) > n_classes: # Should not happen, but safety check
                scores = scores[:n_classes]


            out = pd.concat([out, pd.DataFrame(data=scores, index=out_idx, columns=[score_name])], axis=1)

        if larger_label_space:
            # Select columns: Pred Class 0, Pred 'Any Other', Score
            cols_to_keep_indices = []
            if "Pred=" + labels[0] in out.columns: # Check if Pred=Class 0 exists
                cols_to_keep_indices.append(out.columns.get_loc("Pred=" + labels[0]))
            if len(labels) > 1 and "Pred=" + labels[1] in out.columns: # Check if Pred for the *second* label exists (often represents 'any other' in binary pred)
                cols_to_keep_indices.append(out.columns.get_loc("Pred=" + labels[1]))
            if score and score_name in out.columns: # Check if score column exists
                cols_to_keep_indices.append(out.columns.get_loc(score_name)) # Score column index

            # Check if indices were found
            if not cols_to_keep_indices:
                print("Warning: Could not find expected columns for larger_label_space formatting.")
            else:
                out = out.iloc[:, cols_to_keep_indices]
                new_colnames = []
                if "Pred=" + labels[0] in out.columns:
                    new_colnames.append(out.columns[0]) # Keep original name like Pred=none
                if len(out.columns) > len(new_colnames) : # If there's a second column (presumed 'any')
                    new_colnames.append("Pred=any")
                if score and len(out.columns) > len(new_colnames): # If score column exists
                    new_colnames.append(score_name) # Score column name
                out.columns = new_colnames

        return out
    # --- End of function ---


# --- Helper Functions from Notebook ---

def get_patients_scores(ytrue, ypred, th=0.5):
    """Calculates binary classification metrics."""
    res = dict(tn=0, fp=0, fn=0, tp=0)
    ytrue = np.array(ytrue).astype(int)
    ypred = np.array(ypred)

    if ytrue.ndim > 1 and ytrue.shape[1] > 1:
        ytrue = ytrue[:, 1] # Assume second column is positive class if multiclass input
    if ypred.ndim > 1 and ypred.shape[1] > 1:
        ypred = ypred[:, 1] # Assume second column is positive class score

    tn, fp, fn, tp = confusion_matrix(ytrue, ypred >= th, labels=[0, 1]).ravel()
    res["tn"], res["fp"], res["fn"], res["tp"] = float(tn), float(fp), float(fn), float(tp)

    tot_neg = res["tn"] + res["fp"]
    res["fpr"] = res["fp"] / tot_neg if tot_neg > 0 else 0
    res["tnr"] = 1 - res["fpr"]
    res["spec"] = res["tnr"] # Specificity = TNR

    res["p"], res["r"], res["f1score"], _ = precision_recall_fscore_support(
        ytrue, ypred >= th, average="binary", zero_division=0
    )
    res["sen"] = res["r"] # Sensitivity = Recall = TPR

    res["bAcc"] = 0.5 * (res["sen"] + res["spec"]) if (res["sen"] is not None and res["spec"] is not None) else 0

    # Check if AUROC can be calculated
    if len(np.unique(ytrue)) < 2:
        res["auroc"] = 0 # Or np.nan? Let's use 0 for consistency with original code.
    else:
        try:
            res["auroc"] = roc_auc_score(ytrue, ypred) # labels=[0,1] is default
        except ValueError:
             res["auroc"] = 0 # Handle cases where y_pred might be constant

    # Likelihood Ratios
    res["lr-"] = (1 - res["sen"]) / res["spec"] if res["spec"] != 0 else np.inf
    res["lr+"] = res["sen"] / (1 - res["spec"]) if res["spec"] != 1 else np.inf

    return res

def nice_format(df, ndigits=2, short=True):
    """Formats pandas Series/DataFrame with median (IQR or Q1-Q3)."""
    if isinstance(df, pd.Series):
        df = df.to_frame().T # Handle Series input

    med = df.median(0).round(ndigits).astype(str)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    if short:
        iqr = (q3 - q1).round(ndigits).astype(str)
        out = med + " (" + iqr + ")"
    else:
        out = med + " (" + q1.round(ndigits).astype(str) + " - " + q3.round(ndigits).astype(str) + ")"
    return out

def rename_model(m):
    """Provides a nicer name string for model filenames."""
    parsed = parse("nflow{}.4.[{}_{}].{}.50", m)
    if parsed is not None:
        h, n1, n2, t = parsed
        t1 = "G" if t == "fit3" else "D"
        t2 = "D" if t == "fit3" else "G"
        return f"NF h={h}, {t1}({n1}) -> {t2}({n2})" # Clarified name
    parsed = parse("gmm{}.diag",m)
    if parsed is not None:
        return "NB" # Naive Bayes / Gaussian Mixture Model
    # Add other parsing rules if needed
    if 'xgboost' in m: return "XGBoost"
    if 'logistic' in m: return "Logistic Reg."
    return m # Return original name if no rule matches

def binarize_multiclass(y):
    """Converts multi-class target/prediction array to binary (class 0 vs. any other)."""
    if y is None or y.shape[1] <= 1:
        return y # Return as is if already binary or invalid

    # Check if probabilities or one-hot encoding
    is_probs = not np.all((y == 0) | (y == 1)) # Check if not strictly 0s and 1s

    if is_probs:
        # Sum probabilities of non-zero classes
        if y.shape[1] > 1:
            prob_other = y[:, 1:].sum(axis=1)
            prob_zero = y[:, 0]
            # Ensure probabilities sum to 1 (or close) across the new binary classes
            total = prob_zero + prob_other
            # Avoid division by zero; if total is 0, keep probs as 0
            prob_zero = np.divide(prob_zero, total, out=np.zeros_like(prob_zero), where=total!=0)
            prob_other = np.divide(prob_other, total, out=np.zeros_like(prob_other), where=total!=0)
            out = np.stack([prob_zero, prob_other], axis=1)
        else: # Should not happen based on initial check, but handle defensively
            out = y
    else:
        # Assume one-hot encoding or multi-label binary
        if y.shape[1] > 1:
            is_other = (y[:, 1:].sum(axis=1) > 0).astype(float)
            is_zero = y[:, 0].astype(float) # Should be mutually exclusive in one-hot
            # Handle multi-label: if both are present, how to binarize?
            # Assuming priority for 'other': if any 'other' is 1, classify as 'other'
            is_zero[is_other == 1] = 0 # If 'other' is true, 'zero' must be false in binary case
            out = np.stack([is_zero, is_other], axis=1)
        else:
            out = y

    return out

def combine_tt(x):
    """Selects the time-to-event value with the minimum absolute value across different event types for each time point."""
    """Input: (n_timelines, T), Output: (T,)"""
    if x.ndim == 1: return x # Only one timeline
    if x.shape[0] == 0: return np.array([]) # No timelines
    return x[np.argmin(np.abs(x), axis=0), np.arange(x.shape[1])]


def _aggregate_scores_run(d_tuple, theset="val", th=None, class0_name="not_target__healthy"):
    """Helper function to process results for a single model file."""
    irun, d = d_tuple # Unpack index and data
    if d is None: # Handle cases where data loading failed or was skipped
        return None

    # Initialize results dictionaries for this run
    aurocs_results = {}
    prs_results = {}
    aurocs_scores = {}
    patwise_predictions = {}
    pos_pat = {}
    all_pat = {}
    Cfgs = {}

    try:
        # Basic check for essential keys
        if "cfg" not in d or "results" not in d:
             print(f"Warning: Skipping run {irun} due to missing 'cfg' or 'results'.")
             return None

        cfg = d["cfg"]
        # --- Configuration Extraction ---
        labeling_cfg = cfg.get("labeling", {})
        patients_cfg = cfg.get("patients", {})
        feats_cfg = cfg.get("feats", {})
        model_cfg = cfg.get("model", {})

        # Check for restriction (example from notebook)
        # Adapt this if the restriction logic is different
        # if labeling_cfg.get("restrict_bw") != 24:
        #     print(f"Skipping run {irun} due to restrict_bw != 24.")
        #     return None # Skip this run based on config

        # --- Model Configuration ---
        # Use .get() for safer access
        run_cfg_dict = {
            "pop": patients_cfg.get("fname", "N/A"),
            "features": feats_cfg.get("feat_mode", "N/A"),
            "demos": feats_cfg.get("demos", "N/A"),
            "rback": labeling_cfg.get("restrict_bw", "N/A"), # Example, might need adjustment
            "model_name": model_cfg.get("name", f"unknown_model_{irun}")
        }
        model_name = run_cfg_dict["model_name"]

        # Create unique key for this configuration
        k = ", ".join([f"{key}={val}" for key, val in run_cfg_dict.items()])

        # Initialize lists for this key
        aurocs_scores[k] = []
        prs_results[k] = []
        aurocs_results[k] = []
        patwise_predictions[k] = {}
        all_pat[k] = []
        pos_pat[k] = []

        # Process each result fold within the file
        for i_res, _r in enumerate(d.get("results", [])):
            try:
                r = decompress_obj(_r)
                if not isinstance(r, dict) or theset not in r:
                     print(f"Warning: Skipping result {i_res} in run {irun} due to invalid format or missing set '{theset}'.")
                     continue

                df_set = r[theset] # DataFrame for the current set (train/val/test)

                # --- Identify Target and Prediction Columns ---
                pred_targets = sorted([c for c in df_set.columns if c.startswith("pred__")])
                if not pred_targets:
                     print(f"Warning: No prediction columns found for result {i_res}, run {irun}. Skipping fold.")
                     continue

                # Infer true targets based on prediction columns
                # Be robust to different naming conventions (e.g., 'pred__model__target_name' or 'pred__target_name')
                true_targets = []
                for pt in pred_targets:
                     # Try removing known prefixes
                     tt = pt.replace(f"pred__{model_name}__", "")
                     if tt == pt: # If model name wasn't in the prefix
                         tt = pt.replace("pred__", "")
                     # Check if the inferred target column exists
                     if tt in df_set.columns:
                         true_targets.append(tt)
                     else:
                         # Fallback: search for columns starting with "target__" or matching class0_name
                         potential_matches = [c for c in df_set.columns if c.startswith("target__") or c == class0_name]
                         # This part is tricky, needs a reliable way to map pred to true if names differ significantly
                         # For now, we assume a standard naming convention was followed or use a fixed list if available
                         # If 'true_targets' list was passed as argument, use that instead.
                         print(f"Warning: Could not reliably map prediction '{pt}' to a true target column. Check naming.")


                if not true_targets:
                    # If mapping failed, try using a predefined list (adjust as needed)
                    predefined_targets = [class0_name] + [c for c in df_set.columns if c.startswith("target__")]
                    if len(predefined_targets) == len(pred_targets) and all(t in df_set.columns for t in predefined_targets):
                        true_targets = predefined_targets
                        print(f"Info: Using predefined targets for run {irun}, result {i_res}")
                    else:
                        print(f"Error: Cannot determine true target columns for run {irun}, result {i_res}. Skipping fold.")
                        continue


                # Ensure target columns exist
                if not all(tt in df_set.columns for tt in true_targets):
                     missing_cols = [tt for tt in true_targets if tt not in df_set.columns]
                     print(f"Warning: Missing true target columns {missing_cols} for result {i_res}, run {irun}. Skipping fold.")
                     continue


                # --- Get True and Predicted Values ---
                y_true_multi = df_set[true_targets].values
                y_pred_multi = df_set[pred_targets].values

                # --- Binarize for Standard Metrics ---
                # Assumes the first column is 'negative' or 'class 0'
                y_true_bin = binarize_multiclass(y_true_multi)
                y_pred_bin = binarize_multiclass(y_pred_multi)

                if y_true_bin is None or y_pred_bin is None or y_true_bin.shape[0] == 0:
                     print(f"Warning: Skipping result {i_res}, run {irun} due to empty or invalid binarized data.")
                     continue

                y_true_flat = y_true_bin[:, 1] # Positive class labels
                y_pred_scores = y_pred_bin[:, 1] # Positive class scores/probabilities

                # --- Calculate Metrics ---
                if len(np.unique(y_true_flat)) >= 2: # Need at least two classes for ROC/PR
                    # AUROC
                    fpr, tpr, roc_thresholds = roc_curve(y_true_flat, y_pred_scores)
                    auroc_val = roc_auc_score(y_true_flat, y_pred_scores)
                    aurocs_results[k].append((fpr, tpr, roc_thresholds, auroc_val))

                    # Find optimal threshold for scoring (e.g., maximizing TPR*(1-FPR)) or use fixed 0.5
                    if th is None: # Find optimal from ROC
                       imax = np.argmax(tpr * (1 - fpr))
                       opt_th = roc_thresholds[imax] if len(roc_thresholds) > imax else 0.5
                    else: # Use fixed threshold
                       opt_th = th
                    current_scores = get_patients_scores(y_true_flat, y_pred_scores, th=opt_th)
                    aurocs_scores[k].append(current_scores)

                    # Precision-Recall
                    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_flat, y_pred_scores)
                    # Note: AUROC is appended here in the original code, which is confusing.
                    # Should probably append AUPRC (Average Precision)
                    auprc_val = average_precision_score(y_true_flat, y_pred_scores)
                    prs_results[k].append((precisions, recalls, pr_thresholds, auprc_val)) # Appending AUPRC

                else:
                    # Handle cases with only one class present in this fold
                    print(f"Warning: Only one class present in fold {i_res}, run {irun}. Cannot calculate AUROC/AUPRC.")
                    # Append dummy values or skip? Appending NaNs allows keeping track
                    aurocs_results[k].append((np.array([0, 1]), np.array([0, 1]), np.array([]), np.nan)) # Dummy ROC
                    prs_results[k].append((np.array([]), np.array([]), np.array([]), np.nan)) # Dummy PR
                    # Calculate scores with a fixed threshold (0.5) if possible, might result in 0/inf metrics
                    aurocs_scores[k].append(get_patients_scores(y_true_flat, y_pred_scores, th=0.5))


                # --- Process Patient-wise Predictions ---
                lbl = [c.replace("target__", "").replace(class0_name, "none") for c in true_targets]
                thevalpats = df_set["ids__uid"].unique()

                for ids__uid in thevalpats:
                    pat_idx = df_set["ids__uid"] == ids__uid
                    if not np.any(pat_idx): continue # Skip if patient ID not found (shouldn't happen)

                    ytrue_pat_multi = df_set.loc[pat_idx, true_targets].values
                    ypred_pat_multi = df_set.loc[pat_idx, pred_targets].values

                    # Binarize patient data
                    ytrue_pat_bin = binarize_multiclass(ytrue_pat_multi)
                    ypred_pat_bin = binarize_multiclass(ypred_pat_multi)

                    # Extract time variables (adjust column names as needed)
                    pna_days_col = "feats__pna_days" # Example name
                    pna_h_col = "tl__pna_h" # Example name from notebook
                    tt_cols = [s for s in df_set.columns if s.startswith("time_to")]

                    pna_days = df_set.loc[pat_idx, pna_days_col].values if pna_days_col in df_set.columns else np.full(pat_idx.sum(), np.nan)
                    pna_h = df_set.loc[pat_idx, pna_h_col].values if pna_h_col in df_set.columns else np.full(pat_idx.sum(), np.nan)
                    time_to = df_set.loc[pat_idx, tt_cols].values if tt_cols else np.full((pat_idx.sum(), 0), np.nan)


                    # Extract log likelihood if available
                    log_px_col = f"log_px__{model_name}" # Requires model name consistency
                    log_px = df_set.loc[pat_idx, log_px_col].values if log_px_col in df_set.columns else np.full(pat_idx.sum(), np.nan)

                    pat_data = {
                        "ytrue": ytrue_pat_bin, # Use binarized version for consistency
                        "ypred": ypred_pat_bin, # Use binarized version
                        "ytrue_multi": ytrue_pat_multi, # Keep original multi-class if needed later
                        "ypred_multi": ypred_pat_multi,
                        "pna_days": pna_days,
                        "pna_h": pna_h,
                        "time_to": time_to,
                        "model_name": model_name,
                        "lbl": lbl, # Labels correspond to the binarized structure [none, positive]
                        "log_px": log_px
                    }

                    if ids__uid not in patwise_predictions[k]:
                        patwise_predictions[k][ids__uid] = []
                    patwise_predictions[k][ids__uid].append(pat_data)

                    # Track all unique patient IDs and positive patient IDs for this fold
                    if ids__uid not in all_pat[k]:
                        all_pat[k].append(ids__uid)
                    # Check if patient is positive based on binarized 'true' label (presence of class > 0)
                    if np.any(ytrue_pat_bin[:, 1] > 0):
                         if ids__uid not in pos_pat[k]:
                             pos_pat[k].append(ids__uid)

            except Exception as e_fold:
                print(f"Error processing fold {i_res} in run {irun} for config {k}: {e_fold}")
                import traceback
                traceback.print_exc()
                continue # Skip to next fold


        # Store the configuration for this key 'k'
        Cfgs[k] = run_cfg_dict

        # Return results for this file/configuration
        return {
            "aurocs_results": aurocs_results,
            "prs_results": prs_results,
            "aurocs_scores": aurocs_scores,
            "patwise_predictions": patwise_predictions,
            "pos_pat": pos_pat,
            "all_pat": all_pat,
            "Cfgs": Cfgs,
        }

    except Exception as e_run:
        print(f"Error processing run {irun} (config key might be {k}): {e_run}")
        import traceback
        traceback.print_exc()
        return None # Indicate failure for this run

def aggregate_scores_run(data_list, theset="val", th=None, n_jobs=4, class0_name="not_target__healthy"):
    """Aggregates scores across multiple model result files using multiprocessing."""
    if not data_list:
        return {}

    # Prepare data tuples (index, data) for mapping
    data_tuples = list(enumerate(data_list))

    # Create partial function with fixed arguments
    func = partial(_aggregate_scores_run, theset=theset, th=th, class0_name=class0_name)

    print(f"Starting aggregation with {n_jobs} processes for {len(data_tuples)} files...")
    results_list = []
    # Use Pool for parallel processing
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(func, data_tuples)
    print("Aggregation finished.")

    # Filter out None results (from skipped or failed runs)
    results_list = [res for res in results_list if res is not None]

    if not results_list:
        print("Warning: No valid results were aggregated.")
        return {}

    # Combine results from all processes/files
    combined_results = {}
    # Get keys from the first valid result (assuming all results have the same top-level keys)
    keys_to_combine = results_list[0].keys()

    for k in keys_to_combine:
        combined_results[k] = {}
        for res_dict in results_list:
            if k in res_dict:
                # Merge dictionaries: update existing keys, add new ones
                for sub_key, sub_value in res_dict[k].items():
                    if sub_key not in combined_results[k]:
                        combined_results[k][sub_key] = sub_value
                    else:
                        # Logic for combining depends on the key type
                        if isinstance(sub_value, list):
                            combined_results[k][sub_key].extend(sub_value)
                        elif isinstance(sub_value, dict):
                             # For patwise_predictions, merge patient lists
                             if k == "patwise_predictions":
                                 for pat_id, pat_data_list in sub_value.items():
                                     if pat_id not in combined_results[k][sub_key]:
                                         combined_results[k][sub_key][pat_id] = []
                                     combined_results[k][sub_key][pat_id].extend(pat_data_list)
                             else: # For Cfgs, just update (last one wins for same key)
                                 combined_results[k][sub_key].update(sub_value)
                        # Add other merge logic if needed
    return combined_results

# Note: The original reformat_res seemed overly complex or specific to a previous structure.
# The new aggregate_scores_run directly combines results into the desired format.

def density_plot(time_to_event_a, risks_a, title="Risk Density Plot", xlabel="Time (hours)", ylabel="Risk Score",
                 font_size=12, xlim_l=(-np.inf, np.inf), nh=4, nbins_=10, bwidth=0.1, bstart=0):
    """Creates a 2D density plot of risk scores over time."""
    if time_to_event_a is None or risks_a is None or len(time_to_event_a) == 0:
        print("Warning: No data provided for density plot.")
        return go.Figure() # Return empty figure

    dall = pd.DataFrame({
        "x": time_to_event_a.flatten(),
        "y": risks_a.flatten()
    })
    dall.dropna(inplace=True) # Remove rows with NaNs

    # Filter by time limits
    dall = dall[(dall["x"] >= xlim_l[0]) & (dall["x"] <= xlim_l[1])]
    if dall.empty:
        print("Warning: No data remains after applying xlim_l for density plot.")
        return go.Figure()

    # Bin time
    dall["x_binned"] = (dall["x"] // nh) * nh

    # Define risk bins
    risk_bins = np.linspace(bstart, bstart + nbins_ * bwidth, nbins_ + 1)

    # Group by binned time and calculate histogram for risk within each time bin
    z_list = []
    x_bins_used = []
    grouped = dall.groupby("x_binned")
    for name, group in grouped:
        hist, _ = np.histogram(group["y"], bins=risk_bins, density=True)
        z_list.append(hist)
        x_bins_used.append(name) # Store the time bin center/start

    if not z_list:
        print("Warning: No data groups found after binning for density plot.")
        return go.Figure()

    z = np.array(z_list).T # Transpose so rows are risk bins, columns are time bins
    x_labels = [f"{t:.0f}" for t in x_bins_used] # Time labels
    y_labels = [f"{risk_bins[i]:.1f}-{risk_bins[i+1]:.1f}" for i in range(nbins_)] # Risk labels

    fig = px.imshow(z, x=x_labels, y=y_labels, template="none",
                    color_continuous_scale='RdBu_r', origin='lower', aspect="auto",
                    labels={'x': xlabel, 'y': ylabel, 'color': 'Density'},
                    title=title)

    fig.update_layout(font={"size": font_size})
    fig.update_xaxes(title_font={"size": font_size + 2}, tickangle=0)
    fig.update_yaxes(title_font={"size": font_size + 2})
    fig.update_coloraxes(colorbar_title_font={"size": font_size + 2})

    return fig

def plot_example_pat(pat_data_list, ids__uid, xlim_l=None, layout_d=None, width=None, height=None, xname="pna_days"):
    """Plots risk prediction trajectory for a single patient from aggregated data."""
    if not pat_data_list:
        print(f"No data found for patient {ids__uid}")
        return go.Figure()

    # Assuming we use the first fold's data if multiple exist per patient
    # Or potentially average/median across folds - check requirements
    pat_d = pat_data_list[0] # Use first available fold result for the patient

    # Extract data, handling potential missing keys
    pos_pat_x = pat_d.get(xname)
    pos_pat_y = pat_d.get("ypred") # Binarized prediction scores [class0, class1]
    pos_pat_ytrue = pat_d.get("ytrue") # Binarized true labels

    if pos_pat_x is None or pos_pat_y is None or pos_pat_ytrue is None:
        print(f"Missing required data fields ({xname}, ypred, ytrue) for patient {ids__uid}")
        return go.Figure()

    # Use the score for the positive class (column 1)
    pos_pat_y_score = pos_pat_y[:, 1]
    pos_pat_ytrue_label = pos_pat_ytrue[:, 1]

    # Determine xlabel based on xname
    if xname == "pna_days":
        xlabel = "Postnatal Age (Days)"
    elif "time_to" in xname:
         xlabel = "Time to Event (Hours)" # Assuming hours, adjust if needed
         # Convert 'time_to' array (potentially multi-event) if needed
         if pat_d['time_to'].ndim > 1 and pat_d['time_to'].shape[1] > 0:
             pos_pat_x = combine_tt(pat_d['time_to'].T) # Use the closest event time
         elif pat_d['time_to'].ndim == 1:
             pos_pat_x = pat_d['time_to']
         else: # No time_to data
              print(f"Warning: No valid 'time_to' data for x-axis in patient {ids__uid}")
              return go.Figure()

         # Convert hours to days if needed for consistency? Or keep as hours?
         # The original code seems to plot 'time_to' in hours. Let's stick to that.
         xlabel = "Time to Event (Hours)"

    else:
        xlabel = xname # Default


    # Filtering by xlim_l
    if xlim_l:
        idx = (pos_pat_x >= xlim_l[0]) & (pos_pat_x <= xlim_l[1])
        pos_pat_x = pos_pat_x[idx]
        pos_pat_y_score = pos_pat_y_score[idx]
        pos_pat_ytrue_label = pos_pat_ytrue_label[idx]

    if len(pos_pat_x) == 0:
        print(f"No data points remain for patient {ids__uid} after applying xlim_l.")
        return go.Figure()

    # Smoothing (optional, based on original notebook)
    df=pd.DataFrame(data=np.stack([pos_pat_x, pos_pat_y_score], axis=1), columns=["x", "y"])
    # Ensure x is numeric before trying timedelta conversion if needed
    if pd.api.types.is_numeric_dtype(df['x']):
        try:
             # Convert x-axis units (e.g., days or hours) to timedelta for resampling
             # This might need adjustment based on the unit of xname
             time_unit = 'D' if 'days' in xlabel else 'h' # Heuristic
             df["x_td"] = pd.to_timedelta(df["x"], unit=time_unit)
             df.set_index("x_td", inplace=True)
             # Resample, interpolate, smooth
             df_smooth = df.resample("10min").mean().interpolate("linear").rolling("4H").mean().reset_index() # Example: 4-hour rolling avg

             # Convert back to original units
             total_seconds_in_unit = 24*3600 if time_unit == 'D' else 3600
             df_smooth["x_smooth"] = df_smooth["x_td"].dt.total_seconds() / total_seconds_in_unit
             pospat_x_smooth, pospat_y_smooth = df_smooth[["x_smooth", "y"]].values.T
        except Exception as e_smooth:
             print(f"Warning: Smoothing failed for patient {ids__uid}. Plotting raw data. Error: {e_smooth}")
             pospat_x_smooth, pospat_y_smooth = None, None # Fallback to raw
    else:
        print(f"Warning: x-axis ('{xname}') is not numeric. Cannot perform time-based smoothing.")
        pospat_x_smooth, pospat_y_smooth = None, None


    # Plotting
    fig = go.Figure()

    # Smoothed line
    if pospat_x_smooth is not None and pospat_y_smooth is not None:
        fig.add_trace(
            go.Scatter(x=pospat_x_smooth, y=pospat_y_smooth, name="Smoothed Risk", mode='lines', line=dict(color="black", width=3))
        )

    # Raw data points
    fig.add_trace(
        go.Scatter(x=pos_pat_x, y=pos_pat_y_score, name="Raw Risk", mode='markers',
                   marker=dict(color='blue', size=8, opacity=0.6))
    )

    # True label indicator (e.g., vertical lines or shaded regions)
    # Find changes in true label to plot spans or lines
    change_points = np.where(np.diff(pos_pat_ytrue_label, prepend=np.nan))[0]
    for i, idx in enumerate(change_points):
         is_positive_interval = pos_pat_ytrue_label[idx] == 1
         start_x = pos_pat_x[idx]
         end_x = pos_pat_x[change_points[i+1]-1] if i + 1 < len(change_points) else pos_pat_x[-1]
         if is_positive_interval:
             fig.add_vrect(
                 x0=start_x, x1=end_x,
                 fillcolor="green", opacity=0.2, layer="below", line_width=0,
                 name="True Positive Period" if i == 0 else None, # Show legend only once
                 showlegend=(i==0)
             )
    # Handle case where the entire duration has the same label
    if len(change_points) <= 1 and len(pos_pat_ytrue_label) > 0 and pos_pat_ytrue_label[0] == 1:
         fig.add_vrect(
             x0=pos_pat_x[0], x1=pos_pat_x[-1],
             fillcolor="green", opacity=0.2, layer="below", line_width=0,
             name="True Positive Period", showlegend=True
         )


    fig.update_layout(
        title=f"Patient {ids__uid[:8]} Risk Trajectory",
        xaxis_title=f"<b>{xlabel}</b>",
        yaxis_title="<b>Predicted Risk (Positive Class)</b>",
        yaxis_range=[0, 1],
        template="plotly_white", # Use a standard template
        width=width, height=height,
        legend_title_text='Legend'
    )
    if layout_d: # Apply custom layout updates
        fig.update_layout(**layout_d)
    if xlim_l:
         fig.update_xaxes(range=xlim_l)

    return fig


# --- Main Execution ---

def main(results_dir, model_prefix, output_dir):
    """Runs the full analysis pipeline."""

    print(f"Starting analysis for models matching '{model_prefix}*' in '{results_dir}'")
    print(f"Outputs will be saved to '{output_dir}'")

    # --- Configuration ---
    save = True # Enable saving outputs
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Plotting settings (adjust as needed)
    model_colors = px.colors.qualitative.Plotly # Use a standard palette
    font_size = 12
    layout_d = {"plot_bgcolor": "#fff", "font": {"size": font_size}}
    yaxes_d = {"gridcolor": "#ddd"}
    fontdict = {'fontsize': font_size}
    width = 800
    height = 600
    save_opt_plt = {'bbox_inches': 'tight', 'dpi': 300}
    save_opt_px = {'scale': 2} # Scale for Plotly image export

    # Rename dictionary for plots/tables
    rename_d = {
         "data/firstclinical.csv": "Cohort", # Example names, adjust
         "data/firstclinical_vlbw.csv": "VLBW",
         "bw.sex": "BW & Sex",
         "bw.sex.pna": "BW, Sex & PNA",
         "pna": "PNA",
         "": "None",
         0: "HRC", # Example feature mode names, adjust
         2: "All",
         "pop": "Population",
         "features": "Features",
         "demos": "Demographics",
         "fpr": "FPR", "spec": "Spec.", "p": "Prec.", "r": "Recall",
         "f1score": "F1-score", "auroc": "AUROC", "bAcc": "Balanced Acc.",
         "sen": "Sensitivity", "tnr": "Specificity"
    }

    # --- Find and Load Data ---
    search_pattern = os.path.join(results_dir, f"{model_prefix}*.pklz")
    all_files = glob(search_pattern)
    print(f"Found {len(all_files)} model result files.")
    if not all_files:
        print("No files found matching the pattern. Exiting.")
        sys.exit(1)

    # Load data (consider loading only necessary parts if memory is an issue)
    # This might take time and memory
    print("Loading data from files...")
    # Wrap read_pklz in error handling
    loaded_data = []
    for f in all_files:
        try:
            loaded_data.append(read_pklz(f))
        except Exception as e:
            print(f"Warning: Failed to load or read file {f}: {e}")
            loaded_data.append(None) # Add placeholder to maintain list length if needed by aggregation index

    # Filter out None entries before aggregation
    valid_data = [d for d in loaded_data if d is not None]
    if not valid_data:
         print("Error: No valid data could be loaded. Exiting.")
         sys.exit(1)
    print(f"Successfully loaded {len(valid_data)} files.")


    # --- Aggregate Results ---
    # Define the name of the 'negative' or 'healthy' class column used in results
    # This needs to be consistent with how the results were saved.
    class0_name_in_results = "not_target__healthy" # Adjust if necessary

    # Use caching for aggregated results
    agg_cache_file = os.path.join(output_dir, f"aggregated_results_{model_prefix}.pklz")
    if os.path.exists(agg_cache_file):
        print(f"Loading aggregated results from cache: {agg_cache_file}")
        try:
            agg_res = read_pklz(agg_cache_file)
            # Basic validation of cached structure
            if not all(k in agg_res for k in ["aurocs_results", "prs_results", "aurocs_scores", "Cfgs", "patwise_predictions"]):
                 raise ValueError("Cached file has unexpected structure.")
            print("Loaded from cache.")
        except Exception as e:
            print(f"Warning: Failed to load or validate cache file ({e}). Re-aggregating...")
            agg_res = aggregate_scores_run(valid_data, theset="val", th=0.5, n_jobs=4, class0_name=class0_name_in_results)
            if save and agg_res:
                print(f"Saving aggregated results to cache: {agg_cache_file}")
                #write_pklz(agg_cache_file, agg_res)
    else:
        print("Aggregating results (this may take time)...")
        agg_res = aggregate_scores_run(valid_data, theset="val", th=0.5, n_jobs=4, class0_name=class0_name_in_results)
        if save and agg_res:
            print(f"Saving aggregated results to cache: {agg_cache_file}")
            #write_pklz(agg_cache_file, agg_res)

    # Check if aggregation produced results
    if not agg_res or not agg_res.get("Cfgs"):
        print("Error: Aggregation failed or yielded no results. Exiting.")
        sys.exit(1)

    print(f"Aggregated results for {len(agg_res['Cfgs'])} model configurations.")

    # --- Analysis 1: Confusion Matrices & Numerical Summaries ---
    print("\n--- Analysis 1: Numerical Summaries and Tables ---")
    all_scores_list = []
    config_keys = list(agg_res["Cfgs"].keys()) # Get stable list of keys

    for k in config_keys:
        scores_for_k = agg_res["aurocs_scores"].get(k, [])
        cfg_for_k = agg_res["Cfgs"].get(k, {})

        if not scores_for_k:
             print(f"Warning: No scores found for config: {k}")
             continue

        df_k = pd.DataFrame.from_records(scores_for_k) # Use from_records for list of dicts
        # Add config details to each row
        for cfg_key, cfg_val in cfg_for_k.items():
            df_k[cfg_key] = cfg_val
        all_scores_list.append(df_k)

    if not all_scores_list:
        print("Error: No scores dataframes could be created. Cannot proceed with numerical summaries.")
        sys.exit(1)

    dfout = pd.concat(all_scores_list, ignore_index=True)

    # Apply renaming for readability
    dfout_renamed = dfout.copy()
    dfout_renamed = dfout_renamed.rename(columns=rename_d)
    for col in ['Population', 'Features', 'Demographics', 'model_name', 'rback']: # Columns to potentially rename values in
        if col in dfout_renamed.columns:
             # Check if mapping exists for the column values
             # Example: dfout_renamed['Population'] = dfout_renamed['Population'].replace(rename_d)
             # This requires rename_d to have entries like 'data/file.csv': 'Cohort'
             # Be careful with direct replace on all columns
             if col == 'model_name':
                  dfout_renamed[col] = dfout_renamed[col].apply(rename_model)
             else:
                  # Apply renaming only if keys exist in rename_d
                  # Create a specific mapping for the column's current values
                  value_map = {k: v for k, v in rename_d.items() if k in dfout_renamed[col].unique()}
                  if value_map:
                       dfout_renamed[col] = dfout_renamed[col].replace(value_map)


    # --- Save Detailed Numerical Results Table ---
    if save:
        excel_path = os.path.join(tables_dir, f"detailed_results_{model_prefix}.xlsx")
        try:
            dfout_renamed.to_excel(excel_path, index=False)
            print(f"Saved detailed results table to {excel_path}")
        except Exception as e:
            print(f"Error saving detailed results to Excel: {e}")

    # --- Create Summary Table (Median and IQR) ---
    grouping_cols = ['Population', 'Features', 'Demographics', 'rback', 'Model Name'] # Use renamed columns
    # Filter grouping_cols to only those present in dfout_renamed
    grouping_cols = [col for col in grouping_cols if col in dfout_renamed.columns]

    # Define metrics to summarize
    metrics_to_summarize = ['AUROC', 'F1-score', 'Balanced Acc.', 'Prec.', 'Recall', 'Sensitivity', 'Specificity']
    # Filter metrics to only those present
    metrics_to_summarize = [m for m in metrics_to_summarize if m in dfout_renamed.columns]


    if grouping_cols and metrics_to_summarize:
         # Calculate median and IQR grouped by configuration
         summary_table = dfout_renamed.groupby(grouping_cols)[metrics_to_summarize].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)])
         summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values] # Flatten multi-index
         summary_table = summary_table.rename(columns=lambda x: x.replace('<lambda_0>', 'iqr')) # Nicer IQR name
         summary_table = summary_table.round(3)
         summary_table.sort_values(by=[('F1-score_median' if 'F1-score_median' in summary_table.columns else metrics_to_summarize[0]+'_median')], ascending=False, inplace=True)


         # --- Save Summary Table to PDF ---
         if save:
             summary_pdf_path = os.path.join(tables_dir, f"summary_results_{model_prefix}.pdf")
             try:
                 summary_table_reset = summary_table.reset_index()
                 fig_table, ax_table = plt.subplots(figsize=(max(15, summary_table_reset.shape[1]*0.8), max(4, summary_table_reset.shape[0]*0.3))) # Dynamic sizing
                 ax_table.axis('tight')
                 ax_table.axis('off')
                 # Adjust column widths if needed, e.g., make model name wider
                 colWidths = [0.15] * len(summary_table_reset.columns) # Example equal widths
                 # Find model name index to potentially adjust width
                 try:
                     model_col_idx = list(summary_table_reset.columns).index('Model Name')
                     colWidths[model_col_idx] = 0.3 # Make model name wider
                 except ValueError: pass # If 'Model Name' isn't the exact column name

                 the_table = ax_table.table(cellText=summary_table_reset.values,
                                            colLabels=summary_table_reset.columns,
                                            colWidths=colWidths,
                                            cellLoc='center', loc='center')

                 the_table.auto_set_font_size(False)
                 the_table.set_fontsize(8) # Adjust font size as needed
                 the_table.scale(1, 1.5) # Adjust scale
                 ax_table.set_title(f"Median (IQR) Performance Summary ({model_prefix})", pad=20)

                 with PdfPages(summary_pdf_path) as pp:
                     pp.savefig(fig_table, bbox_inches='tight')
                 plt.close(fig_table)
                 print(f"Saved summary table to {summary_pdf_path}")
             except Exception as e:
                 print(f"Error saving summary table to PDF: {e}")
                 # Save as Excel as fallback
                 try:
                     summary_table.to_excel(summary_pdf_path.replace('.pdf', '.xlsx'))
                     print(f"Saved summary table as Excel fallback: {summary_pdf_path.replace('.pdf', '.xlsx')}")
                 except Exception as e_excel:
                      print(f"Error saving summary table to Excel fallback: {e_excel}")
    else:
         print("Warning: Could not perform summary grouping. Check grouping columns and metrics.")


    # --- Analysis 2: ROC/PR Curves ---
    print("\n--- Analysis 2: ROC and PR Curves ---")
    # Use aggregated results (`agg_res`) which contain lists of (fpr, tpr, ...) tuples per config
    aurocs_data = agg_res.get("aurocs_results", {})
    prs_data = agg_res.get("prs_results", {})
    configs = agg_res.get("Cfgs", {})

    # Select models/configs to plot (e.g., based on performance or specific names)
    # Example: Plot all, or filter based on config keys
    configs_to_plot = list(configs.keys()) # Plot all available configs

    # Plot ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    plot_roc_data = []
    plot_labels = []
    plot_colors = []
    assigned_colors = {}
    color_idx = 0

    for i, k in enumerate(configs_to_plot):
        roc_list = aurocs_data.get(k)
        if roc_list: # Check if data exists
             model_name = configs[k].get('model_name', 'Unknown')
             nice_name = rename_model(model_name)
             plot_roc_data.append(roc_list)
             plot_labels.append(f"{nice_name} (AUC={np.nanmean([r[3] for r in roc_list]):.3f})")

             # Assign colors based on model base name or type
             base_model_name = nice_name.split(' ')[0] # Simple grouping by first word
             if base_model_name not in assigned_colors:
                  assigned_colors[base_model_name] = model_colors[color_idx % len(model_colors)]
                  color_idx += 1
             plot_colors.append(assigned_colors[base_model_name])


    if plot_roc_data:
        plot_roc(ax_roc, plot_roc_data, "Validation ROC", opts=plot_labels, colors=plot_colors, lw=2, alpha=0.8, fontdict=fontdict)
        better_lookin(ax_roc, legend_bbox=(1.05, 1), grid=True)
        ax_roc.set_title(f"Receiver Operating Characteristic ({model_prefix})")

        if save:
            roc_path = os.path.join(figures_dir, f"roc_curves_{model_prefix}.pdf")
            try:
                fig_roc.savefig(roc_path, **save_opt_plt)
                print(f"Saved ROC curves to {roc_path}")
            except Exception as e:
                print(f"Error saving ROC curves: {e}")
        plt.close(fig_roc)
    else:
        print("No ROC data available to plot.")


    # Plot PR Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
    plot_pr_data = []
    plot_labels_pr = []
    plot_colors_pr = []

    for i, k in enumerate(configs_to_plot):
         pr_list = prs_data.get(k)
         if pr_list: # Check if data exists
             model_name = configs[k].get('model_name', 'Unknown')
             nice_name = rename_model(model_name)
             plot_pr_data.append(pr_list)
             # Use AUPRC (Average Precision) in label, which is the 4th element saved
             plot_labels_pr.append(f"{nice_name} (AUPRC={np.nanmean([p[3] for p in pr_list]):.3f})")

             # Reuse colors assigned in ROC plot
             base_model_name = nice_name.split(' ')[0]
             plot_colors_pr.append(assigned_colors.get(base_model_name, model_colors[i % len(model_colors)])) # Fallback color

    if plot_pr_data:
        plot_roc(ax_pr, plot_pr_data, "Validation PR", opts=plot_labels_pr, colors=plot_colors_pr, lw=2, alpha=0.8, fontdict=fontdict, pr=True) # Set pr=True
        better_lookin(ax_pr, legend_bbox=(1.05, 1), grid=True)
        ax_pr.set_title(f"Precision-Recall Curve ({model_prefix})")

        if save:
            pr_path = os.path.join(figures_dir, f"pr_curves_{model_prefix}.pdf")
            try:
                fig_pr.savefig(pr_path, **save_opt_plt)
                print(f"Saved PR curves to {pr_path}")
            except Exception as e:
                print(f"Error saving PR curves: {e}")
        plt.close(fig_pr)
    else:
        print("No PR data available to plot.")


    # --- Analysis 3: Feature/Demographic Comparison (Violin Plots) ---
    print("\n--- Analysis 3: Feature/Demographic Comparisons ---")
    # Use the `dfout_renamed` DataFrame created earlier

    # Example: AUROC vs Features (assuming 'Features' column exists after renaming)
    if 'Features' in dfout_renamed.columns and 'AUROC' in dfout_renamed.columns:
        try:
            fig_feat = px.violin(dfout_renamed, y="AUROC", x="Features",
                                 points="all", box=True,
                                 color="Features", # Color by feature set
                                 title=f"AUROC vs. Features ({model_prefix})",
                                 template="plotly_white", width=width, height=height,
                                 labels={"AUROC": "<b>AUROC</b>", "Features": "<b>Features Used</b>"})
            fig_feat.update_layout(**layout_d)
            fig_feat.update_yaxes(**yaxes_d, range=[0,1], tickprefix="<b>", ticksuffix ="</b>")
            fig_feat.update_xaxes(tickprefix="<b>", ticksuffix ="</b>")
            # fig.update_traces(**violin_lines_d) # If defined

            if save:
                feat_path = os.path.join(figures_dir, f"violin_auroc_vs_features_{model_prefix}.pdf")
                try:
                     fig_feat.write_image(feat_path, **save_opt_px)
                     print(f"Saved AUROC vs Features violin plot to {feat_path}")
                except Exception as e:
                     print(f"Error saving AUROC vs Features plot: {e}. You might need 'kaleido' installed (`pip install kaleido`).")
            # fig_feat.show() # Don't show in script mode
        except Exception as e_fig:
             print(f"Could not generate AUROC vs Features plot: {e_fig}")
    else:
         print("Skipping AUROC vs Features plot (missing required columns 'Features' or 'AUROC').")


    # Example: AUROC vs Demographics (assuming 'Demographics' column exists)
    if 'Demographics' in dfout_renamed.columns and 'AUROC' in dfout_renamed.columns:
        try:
            fig_demo = px.violin(dfout_renamed, y="AUROC", x="Demographics",
                                 points="all", box=True,
                                 color="Demographics",
                                 title=f"AUROC vs. Demographics ({model_prefix})",
                                 template="plotly_white", width=width, height=height,
                                 labels={"AUROC": "<b>AUROC</b>", "Demographics": "<b>Demographic Features Used</b>"})
            fig_demo.update_layout(**layout_d)
            fig_demo.update_yaxes(**yaxes_d, range=[0,1], tickprefix="<b>", ticksuffix ="</b>")
            fig_demo.update_xaxes(tickprefix="<b>", ticksuffix ="</b>")

            if save:
                 demo_path = os.path.join(figures_dir, f"violin_auroc_vs_demos_{model_prefix}.pdf")
                 try:
                     fig_demo.write_image(demo_path, **save_opt_px)
                     print(f"Saved AUROC vs Demographics violin plot to {demo_path}")
                 except Exception as e:
                     print(f"Error saving AUROC vs Demographics plot: {e}. You might need 'kaleido' installed.")
            # fig_demo.show()
        except Exception as e_fig:
             print(f"Could not generate AUROC vs Demographics plot: {e_fig}")

    else:
         print("Skipping AUROC vs Demographics plot (missing required columns 'Demographics' or 'AUROC').")


    # --- Analysis 4: Population/Patient Risk Visualization ---
    print("\n--- Analysis 4: Population/Patient Risk Visualizations ---")
    patwise_preds = agg_res.get("patwise_predictions", {})
    pos_pats_map = agg_res.get("pos_pat", {}) # Map from config key to list of positive patient IDs

    # Select a key configuration to plot (e.g., the best performing one based on summary table)
    # Or loop through a few representative ones
    # For demonstration, pick the first config key that has patient data
    plot_key = None
    for k in config_keys:
         if k in patwise_preds and patwise_preds[k]:
             plot_key = k
             break

    if plot_key:
        print(f"Generating population/patient plots for config: {plot_key[:80]}...") # Print truncated key
        pat_data_dict = patwise_preds[plot_key] # Dictionary {pat_id: [data_fold1, data_fold2,...]}
        pos_pats_for_key = pos_pats_map.get(plot_key, [])

        # Aggregate patient data across folds for the chosen config key if needed
        # Currently, the aggregation step already combines lists per patient.
        # Let's flatten the data for population plots

        all_pat_x = []
        all_pat_y = []
        all_pat_tte = []

        # Use the first fold's data for each patient for simplicity in population plots
        # More advanced: average predictions across folds if available
        for pat_id, data_list in pat_data_dict.items():
             if data_list:
                 d = data_list[0] # Use first fold's data
                 # Choose time axis: 'pna_days' or 'time_to'
                 # Using 'pna_days' for general population plot
                 time_axis = d.get('pna_days')
                 risk_scores = d.get('ypred')[:, 1] if d.get('ypred') is not None and d.get('ypred').shape[1] > 1 else None

                 if time_axis is not None and risk_scores is not None:
                      all_pat_x.extend(time_axis)
                      all_pat_y.extend(risk_scores)

                 # Extract time-to-event for positive patients
                 if pat_id in pos_pats_for_key:
                     tte = d.get('time_to')
                     if tte is not None and tte.ndim > 0: # Check if tte exists and is not empty
                         # Combine multiple event times if necessary
                         tte_combined = combine_tt(tte.T) if tte.ndim > 1 else tte.flatten()
                         risk_scores_tte = d.get('ypred')[:, 1] if d.get('ypred') is not None and d.get('ypred').shape[1] > 1 else None
                         if risk_scores_tte is not None and len(tte_combined) == len(risk_scores_tte):
                              all_pat_tte.extend(list(zip(tte_combined, risk_scores_tte)))


        all_pat_x = np.array(all_pat_x)
        all_pat_y = np.array(all_pat_y)
        all_pat_tte = np.array(all_pat_tte) # Array of (tte, risk) tuples

        # --- Population Density Plot (Risk vs PNA Days) ---
        if len(all_pat_x) > 0:
            config_short_name = rename_model(configs[plot_key]['model_name']).replace(" ", "_")
            fig_density_pna = density_plot(all_pat_x, all_pat_y,
                                           title=f"Population Risk Density vs PNA ({config_short_name})",
                                           xlabel="Postnatal Age (Days)", ylabel="Risk Score", font_size=font_size)
            if save:
                density_pna_path = os.path.join(figures_dir, f"density_risk_vs_pna_{config_short_name}_{model_prefix}.pdf")
                try:
                    fig_density_pna.write_image(density_pna_path, **save_opt_px)
                    print(f"Saved Risk vs PNA density plot to {density_pna_path}")
                except Exception as e:
                     print(f"Error saving Risk vs PNA density plot: {e}.")
            # fig_density_pna.show()

        # --- Population Density Plot (Risk vs Time To Event for Positive Patients) ---
        if len(all_pat_tte) > 0:
             tte_values = all_pat_tte[:, 0]
             risk_values_tte = all_pat_tte[:, 1]
             config_short_name = rename_model(configs[plot_key]['model_name']).replace(" ", "_")

             fig_density_tte = density_plot(tte_values, risk_values_tte,
                                            title=f"Positive Patient Risk Density vs TTE ({config_short_name})",
                                            xlabel="Time to Event (Hours)", ylabel="Risk Score", font_size=font_size,
                                            xlim_l=(-168, 48)) # Example: focus on -7 days to +2 days
             if save:
                 density_tte_path = os.path.join(figures_dir, f"density_risk_vs_tte_{config_short_name}_{model_prefix}.pdf")
                 try:
                     fig_density_tte.write_image(density_tte_path, **save_opt_px)
                     print(f"Saved Risk vs TTE density plot to {density_tte_path}")
                 except Exception as e:
                     print(f"Error saving Risk vs TTE density plot: {e}.")
             # fig_density_tte.show()

        # --- Example Patient Plot ---
        # Select one positive patient to plot
        example_pat_id = pos_pats_for_key[0] if pos_pats_for_key else None
        if example_pat_id and example_pat_id in pat_data_dict:
             config_short_name = rename_model(configs[plot_key]['model_name']).replace(" ", "_")
             fig_example = plot_example_pat(pat_data_dict[example_pat_id], example_pat_id,
                                            xname="pna_days", # Plot against PNA days
                                            xlim_l=None, # No limits
                                            layout_d=layout_d, width=width, height=height)
             if save:
                 example_path = os.path.join(figures_dir, f"example_patient_{example_pat_id[:8]}_{config_short_name}_{model_prefix}.pdf")
                 try:
                     fig_example.write_image(example_path, **save_opt_px)
                     print(f"Saved example patient plot to {example_path}")
                 except Exception as e:
                     print(f"Error saving example patient plot: {e}.")
             # fig_example.show()

    else:
        print("Skipping population/patient plots (no suitable configuration or patient data found).")


    # --- Analysis 5: Alarm Rate Analysis ---
    print("\n--- Analysis 5: Alarm Rate Analysis ---")
    # Requires looping through folds again or using the detailed 'patwise_predictions'
    # This was complex in the notebook, focusing on specific model 'gmm1.diag'
    # Replicating requires careful handling of patient status and time alignment.
    # Simplified version: Calculate overall FP/TP rates from the summary scores.

    # Or replicate the notebook's approach for a specific model:
    target_model_name_alarm = "gmm1.diag" # Example from notebook
    alarm_plot_key = None
    for k, cfg in configs.items():
        if cfg.get('model_name') == target_model_name_alarm:
             alarm_plot_key = k
             break

    if alarm_plot_key and alarm_plot_key in patwise_preds:
        print(f"Generating alarm rate plots for model: {target_model_name_alarm}")
        pat_data_dict_alarm = patwise_preds[alarm_plot_key]
        pos_pats_alarm = pos_pats_map.get(alarm_plot_key, [])
        all_pats_alarm = list(pat_data_dict_alarm.keys()) # All patients with data for this model

        alarm_data_list = []
        for pat_id, data_list in pat_data_dict_alarm.items():
            if data_list:
                 d = data_list[0] # Use first fold
                 is_pos = pat_id in pos_pats_alarm
                 status = "pos" if is_pos else "neg"

                 time_pna = d.get('pna_days')
                 time_tte = combine_tt(d.get('time_to').T) if d.get('time_to') is not None and d.get('time_to').ndim > 1 else d.get('time_to')

                 pred_bin = d.get('ypred')[:, 1] >= 0.5 if d.get('ypred') is not None else None # Example threshold 0.5
                 true_bin = d.get('ytrue')[:, 1] if d.get('ytrue') is not None else None

                 if pred_bin is not None and true_bin is not None:
                      alarms = pred_bin
                      tp = pred_bin & true_bin
                      fp = pred_bin & (~true_bin)

                      if time_pna is not None and len(time_pna) == len(alarms):
                           for i in range(len(time_pna)):
                                alarm_data_list.append({
                                     "patid": pat_id, "status": status,
                                     "pn_age_days": time_pna[i],
                                     "tte_days": time_tte[i]/24 if time_tte is not None and i < len(time_tte) else np.nan, # Convert tte hours to days
                                     "alarm": alarms[i], "tp": tp[i], "fp": fp[i], "true": true_bin[i]
                                })


        if alarm_data_list:
             df_alarm = pd.DataFrame(alarm_data_list)
             df_alarm.dropna(subset=['pn_age_days'], inplace=True) # Need PNA age at least

             # Plot Alarm Rate vs PNA Days
             try:
                 fig_alarm_pna, axes_alarm_pna = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                 ax_rate, ax_count = axes_alarm_pna

                 bins = np.arange(0, df_alarm['pn_age_days'].max() + 5, 5) # 5-day bins
                 df_alarm['pna_binned'] = pd.cut(df_alarm['pn_age_days'], bins, right=False, labels=bins[:-1])

                 # Calculate rates per bin
                 # Need careful aggregation: mean alarm rate per patient *within* the bin, then average across patients?
                 # Or simply mean alarm rate across all points in the bin? Notebook seemed to do the latter.
                 grouped = df_alarm.groupby(['pna_binned', 'status'])
                 binned_stats = grouped[['alarm', 'fp', 'tp']].mean().reset_index()
                 patient_counts = df_alarm.groupby(['pna_binned', 'status'])['patid'].nunique().reset_index()

                 for status, color in [("pos", "darkred"), ("neg", "darkblue")]:
                     subset_stats = binned_stats[binned_stats['status'] == status]
                     subset_counts = patient_counts[patient_counts['status'] == status]
                     if not subset_stats.empty:
                          # Plot FP rate for neg, TP+FP (total alarms) for pos
                          rate_to_plot = subset_stats['fp'] if status == 'neg' else subset_stats['alarm']
                          lbl = f"False Alarms (Neg Pts)" if status == 'neg' else f"Total Alarms (Pos Pts)"
                          ax_rate.plot(subset_stats['pna_binned'].astype(float), rate_to_plot, marker='o', linestyle='-', color=color, label=lbl)
                     if not subset_counts.empty:
                          ax_count.plot(subset_counts['pna_binned'].astype(float), subset_counts['patid'], marker='.', linestyle='--', color=color, label=f"# Patients ({status})")

                 ax_rate.set_ylabel("Alarm Rate")
                 ax_rate.set_ylim(0, 1)
                 ax_rate.set_title(f"Alarm Rate vs Postnatal Age ({target_model_name_alarm})")
                 ax_rate.legend()
                 ax_rate.grid(True)

                 ax_count.set_ylabel("Patient Count")
                 ax_count.set_xlabel("Postnatal Age (days, lower bin edge)")
                 ax_count.legend()
                 ax_count.grid(True)
                 plt.tight_layout()

                 if save:
                      alarm_pna_path = os.path.join(figures_dir, f"alarmrate_vs_pna_{target_model_name_alarm}_{model_prefix}.pdf")
                      fig_alarm_pna.savefig(alarm_pna_path, **save_opt_plt)
                      print(f"Saved Alarm Rate vs PNA plot to {alarm_pna_path}")
                 plt.close(fig_alarm_pna)

             except Exception as e:
                  print(f"Error generating Alarm Rate vs PNA plot: {e}")
                  import traceback
                  traceback.print_exc()


             # Plot Alarm Rate vs Time To Event (Pos Patients Only)
             try:
                  df_alarm_pos = df_alarm[df_alarm['status'] == 'pos'].dropna(subset=['tte_days'])
                  if not df_alarm_pos.empty:
                      fig_alarm_tte, axes_alarm_tte = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                      ax_rate_tte, ax_count_tte = axes_alarm_tte

                      tte_max = 10 # days
                      tte_min = -10 # days
                      tte_step = 1 # 1-day bins
                      bins_tte = np.arange(tte_min, tte_max + tte_step, tte_step)
                      df_alarm_pos['tte_binned'] = pd.cut(df_alarm_pos['tte_days'], bins_tte, right=False, labels=bins_tte[:-1])

                      grouped_tte = df_alarm_pos.groupby('tte_binned')
                      binned_stats_tte = grouped_tte[['alarm', 'fp', 'tp']].mean().reset_index()
                      patient_counts_tte = df_alarm_pos.groupby('tte_binned')['patid'].nunique().reset_index()

                      # Plot total alarms, TP, FP for positive patients
                      ax_rate_tte.bar(binned_stats_tte['tte_binned'].astype(float), binned_stats_tte['fp'], width=tte_step*0.9, color='orange', label='FP Alarms (in Pos Pts)', alpha=0.7)
                      ax_rate_tte.bar(binned_stats_tte['tte_binned'].astype(float), binned_stats_tte['tp'], width=tte_step*0.9, color='darkgreen', label='TP Alarms (in Pos Pts)', alpha=0.7, bottom=binned_stats_tte['fp']) # Stack TP on FP


                      ax_rate_tte.set_ylabel("Alarm Rate")
                      ax_rate_tte.set_ylim(0, 1)
                      ax_rate_tte.set_title(f"Alarm Rate vs Time to Event ({target_model_name_alarm}, Pos Pts)")
                      ax_rate_tte.legend()
                      ax_rate_tte.grid(True)

                      ax_count_tte.plot(patient_counts_tte['tte_binned'].astype(float), patient_counts_tte['patid'], marker='.', linestyle='-', color='black', label="# Patients")
                      ax_count_tte.set_ylabel("Patient Count")
                      ax_count_tte.set_xlabel("Time to Event (days, lower bin edge)")
                      ax_count_tte.legend()
                      ax_count_tte.grid(True)
                      plt.tight_layout()

                      if save:
                           alarm_tte_path = os.path.join(figures_dir, f"alarmrate_vs_tte_{target_model_name_alarm}_{model_prefix}.pdf")
                           fig_alarm_tte.savefig(alarm_tte_path, **save_opt_plt)
                           print(f"Saved Alarm Rate vs TTE plot to {alarm_tte_path}")
                      plt.close(fig_alarm_tte)

                  else:
                       print("No positive patient data with TTE for alarm rate plot.")

             except Exception as e:
                 print(f"Error generating Alarm Rate vs TTE plot: {e}")
                 import traceback
                 traceback.print_exc()

        else:
             print("No data aggregated for alarm rate plots.")
    else:
         print(f"Skipping alarm rate plots (model '{target_model_name_alarm}' not found or no patient data).")


    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on adverse event prediction model results.")
    parser.add_argument("results_dir", help="Directory containing the model result .pklz files.")
    parser.add_argument("model_prefix", help="Prefix of the model result filenames (e.g., 'nflow', 'gmm').")
    parser.add_argument("output_dir", help="Directory to save the analysis outputs (figures, tables).")

    args = parser.parse_args()

    # Run the main analysis function
    main(args.results_dir, args.model_prefix, args.output_dir)