# Required Imports (ensure these match your environment)
import os
import gc
import socket
import argparse # Keep for potential script reuse, but not used by Flask app directly
from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting BEFORE importing pyplot
import matplotlib.pyplot as plt
from datetime import timedelta
from functools import partial
from parse import parse
from multiprocessing import Pool
import sys
import warnings
import io # For handling plot image data in memory
import base64 # For embedding matplotlib plots
import traceback # For detailed error logging
import shutil # For cache management (optional)

# Plotting Libraries
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages # Keep for potential future use, but not primary output

# Scikit-learn Metrics
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, roc_auc_score, precision_recall_fscore_support,
    average_precision_score, multilabel_confusion_matrix
)

# Flask Import
from flask import Flask, request, render_template_string, flash, redirect, url_for, session

# --- Custom Utils Section ---
# Using dummy functions as provided in the original script's fallback
# Ensure these match the actual utilities if they are available in your environment.

try:
    # Attempt to import real utils if they exist
    from utils_tbox.utils_tbox import read_pklz, decompress_obj, write_pklz
    from utils_plots.utils_plots import plot_ci, better_lookin, linestyle_tuple
    from utils_results_analysis.plots import plot_roc
    from utils_results_analysis.utils_results_analysis import custom_confusion_matrices #, topkmulticlass # topkmulticlass seems unused later
    print("INFO: Successfully imported custom utility modules.")

except ImportError as e:
    print(f"WARNING: Could not import custom utility modules: {e}. Using dummy functions.")
    # Define dummy functions
    def read_pklz(f):
        # Simple pickle read, assuming no compression or handling it internally
        try:
            with open(f, 'rb') as file:
                # Try decompressing first if it's a compressed pickle
                try:
                    import zlib
                    return pkl.loads(zlib.decompress(file.read()))
                except Exception: # Fallback to standard pickle
                    file.seek(0)
                    return pkl.load(file)
        except Exception as err:
            print(f"ERROR in dummy read_pklz reading {f}: {err}")
            raise

    def write_pklz(f, obj):
        # Simple pickle write, assuming no compression needed by default
        try:
            import zlib
            with open(f, 'wb') as file:
                file.write(zlib.compress(pkl.dumps(obj)))
        except Exception as err:
            print(f"ERROR in dummy write_pklz writing {f}: {err}")
            raise

    def decompress_obj(obj):
        # Assuming obj is potentially compressed bytes
        try:
            import zlib
            return pkl.loads(zlib.decompress(obj))
        except:
            # If it's not compressed bytes or fails, return as is
            return obj

    def plot_ci(ax, data, **kwargs):
        # Dummy: just plot the data, ignore confidence interval aspect
        if data is not None and len(data) > 0:
             # Basic plot assuming data is suitable for ax.plot
             if isinstance(data, (list, np.ndarray)) and np.ndim(data) == 1:
                 ax.plot(data, **kwargs)
             elif isinstance(data, tuple) and len(data) == 2: # Maybe x, y data?
                 ax.plot(data[0], data[1], **kwargs)
             # Add more checks if needed based on expected data format
        else:
            print("WARN (dummy plot_ci): No data to plot.")


    def better_lookin(ax, legend_bbox=None, grid=True, **kwargs):
        if grid:
            ax.grid(True, linestyle='--', alpha=0.6)
        if legend_bbox and ax.get_legend_handles_labels()[1]: # Check if legend items exist
            ax.legend(bbox_to_anchor=legend_bbox, loc='upper left')
        elif ax.get_legend_handles_labels()[1]:
            ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    linestyle_tuple = [ # Define some basic linestyles
        ('solid',              (0, ())),
        ('dotted',             (0, (1, 1))),
        ('dashed',             (0, (5, 5))),
        ('dashdotted',         (0, (3, 5, 1, 5))),
        ('dashdotdotted',      (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dotted',     (0, (1, 10))),
        ('loosely dashed',     (0, (5, 10))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ]

    def plot_roc(ax, results, title, opts=None, colors=None, lw=1.5, alpha=0.8, fontdict=None, pr=False):
        """Dummy ROC/PR plotting function using matplotlib"""
        print(f"INFO (dummy plot_roc): Plotting {'PR' if pr else 'ROC'} for {title}")
        if not results:
             print("WARN (dummy plot_roc): No results data to plot.")
             return

        n_curves = len(results)
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, n_curves))
        if opts is None:
            opts = [f"Curve {i+1}" for i in range(n_curves)]

        mean_vals = [] # Store mean AUC/AUPRC

        for i, res_list in enumerate(results):
            if not res_list: continue # Skip empty results for a curve

            xs, ys = [], []
            curve_scores = []
            for r in res_list:
                if len(r) < 2 or r[0] is None or r[1] is None: continue # Basic check
                # Check if valid data exists
                if len(r[0]) > 0 and len(r[1]) > 0:
                    xs.append(r[0]) # FPR or Recall
                    ys.append(r[1]) # TPR or Precision
                if len(r) >= 4 and r[3] is not None and not np.isnan(r[3]): # AUC/AUPRC score
                    curve_scores.append(r[3])

            if not xs or not ys: continue # Skip if no valid data pairs found

            # Average curve (simple approach: average y at common x)
            # This is complex, using mean of individual curves is simpler for dummy
            # For simplicity, just plot the first fold's curve
            first_x, first_y = xs[0], ys[0]
            label = opts[i]
            if curve_scores:
                mean_score = np.nanmean(curve_scores)
                mean_vals.append(mean_score)
                label += f" ({'AUPRC' if pr else 'AUC'}={mean_score:.3f})"

            ax.plot(first_x, first_y, color=colors[i % len(colors)], lw=lw, alpha=alpha, label=label)


        # Plot baseline
        if pr:
            # Baseline is related to prevalence, hard to calculate here without y_true
            # Plot y=0.5 or similar as a simple reference
            ax.axhline(0.5, linestyle=':', color='grey', label='Baseline (0.5)')
            ax.set_xlabel("Recall", fontdict=fontdict)
            ax.set_ylabel("Precision", fontdict=fontdict)
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
        else:
            ax.plot([0, 1], [0, 1], linestyle=':', color='grey', label='Chance (AUC=0.5)')
            ax.set_xlabel("False Positive Rate", fontdict=fontdict)
            ax.set_ylabel("True Positive Rate", fontdict=fontdict)
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])

        ax.set_title(title, fontdict=fontdict)
        # legend handling moved to better_lookin

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
            # Ensure index i is valid for y_true_bin's columns
            if i >= y_true_bin.shape[1]:
                print(f"Warning (custom_confusion_matrices): Class index {i} exceeds true label dimensions ({y_true_bin.shape[1]}). Skipping.")
                label_leak.append([0] * y_pred_bin.shape[1]) # Append zeros matching pred columns
                if score: scores.append(np.nan)
                continue

            idx_true_i = y_true_bin[:,i] == True # Use boolean indexing
            n_class_i = idx_true_i.sum()

            if n_class_i == 0:
                label_leak.append([0] * y_pred_bin.shape[1]) # Match pred columns
                if score: scores.append(np.nan) # Or 0? NaN indicates no samples for this class
                continue

            # Get predictions for samples truly belonging to class i
            y_pred_bin_i = y_pred_bin[idx_true_i]
            y_true_bin_i = y_true_bin[idx_true_i]

            # False classifications for class i are the rows where not all labels are equal
            if not larger_label_space:
                # Check if the prediction pattern matches the true pattern for these samples
                # A mismatch occurs if y_pred_bin_i[r,:] != y_true_bin_i[r,:] for any row r
                is_correct_classif = np.all(y_pred_bin_i == y_true_bin_i, axis=1)
                idx_false_classif_i = ~is_correct_classif
                #print(f"Class {i}: n_class_i={n_class_i}, n_false={idx_false_classif_i.sum()}")
            else:
                 # This logic seems specific and might need careful review based on actual use case
                 # Simplified comparison for binary case (pred=[c0, c_other], true=[c0, ci])
                 if y_pred_bin_i.shape[1] >= 2 and y_true_bin_i.shape[1] > i:
                     pred_cols_to_compare = y_pred_bin_i[:, [0, 1]] # Assumes binary prediction [class 0, other]
                     true_cols_to_compare = y_true_bin_i[:, [0, i]]
                     is_correct_classif = np.all(pred_cols_to_compare == true_cols_to_compare, axis=1)
                     idx_false_classif_i = ~is_correct_classif
                 else:
                      print(f"Warning (custom_confusion_matrices): Shape mismatch or invalid index in larger_label_space for class {i}.")
                      idx_false_classif_i = np.ones(y_pred_bin_i.shape[0], dtype=bool) # Mark all as false


            if only_false:
                # Data to sum: predictions for the falsely classified samples of class i
                data_to_sum=y_pred_bin_i[idx_false_classif_i]
            else:
                # Data to sum: all predictions for samples of class i
                data_to_sum=y_pred_bin_i

            if score:
                if topk is None:
                    # Standard Accuracy for this class
                    accuracy_i = (n_class_i - idx_false_classif_i.sum()) / n_class_i if n_class_i > 0 else np.nan
                    scores.append(accuracy_i)
                else:
                    # Placeholder: Top-k logic requires 'topkmulticlass' which wasn't provided/used
                    print(f"Warning: topkmulticlass not implemented - skipping Top-k score for class {i}")
                    scores.append(np.nan) # Placeholder

            # Sum the predicted labels for the selected subset
            if data_to_sum.shape[0] > 0:
                label_leak.append(data_to_sum.sum(0).tolist())
            else:
                label_leak.append([0] * y_pred_bin.shape[1]) # Append zeros matching pred columns

        label_leak = np.array(label_leak)

        # Ensure label_leak has rows for all n_classes (even if skipped)
        if label_leak.shape[0] < n_classes:
            full_leak = np.full((n_classes, label_leak.shape[1]), 0)
            # Simple fill assuming order is maintained - this might be flawed if classes were skipped mid-way
            rows_to_fill = min(label_leak.shape[0], n_classes)
            full_leak[:rows_to_fill, :] = label_leak[:rows_to_fill, :]
            label_leak = full_leak

        # Define row/column names using provided labels
        out_idx = [f"True={l}" for l in labels]
        # Adjust column names based on the actual shape of label_leak columns (prediction shape)
        pred_cols = y_pred_bin.shape[1]
        col_names = [f"Pred={l}" for l in labels[:pred_cols]] # Use labels up to the number of predicted columns
        if pred_cols > len(labels): # If more pred cols than labels, add generic names
             col_names.extend([f"Pred=Unknown_{j}" for j in range(len(labels), pred_cols)])


        # Ensure label_leak has the correct number of columns as derived col_names
        if label_leak.shape[1] != len(col_names):
             print(f"Warning: Mismatch after processing between label_leak columns ({label_leak.shape[1]}) and derived column names ({len(col_names)}). Adjusting matrix.")
             # Pad or truncate label_leak to match col_names length
             if label_leak.shape[1] < len(col_names): # Pad with zeros
                 padded_leak = np.zeros((label_leak.shape[0], len(col_names)), dtype=label_leak.dtype)
                 padded_leak[:, :label_leak.shape[1]] = label_leak
                 label_leak = padded_leak
             else: # Truncate
                  label_leak = label_leak[:, :len(col_names)]


        out = pd.DataFrame(data=label_leak.astype(int),
                            columns=col_names,
                            index=out_idx)

        if score:
            score_name = f"Top-{topk}" if topk else "Accuracy"

            # Ensure scores array matches number of classes (out_idx)
            if len(scores) < len(out_idx):
                full_scores = np.full(len(out_idx), np.nan)
                full_scores[:len(scores)] = scores # Fill known scores
                scores = full_scores
            elif len(scores) > len(out_idx):
                scores = scores[:len(out_idx)] # Truncate if too long

            out = pd.concat([out, pd.DataFrame(data=scores, index=out_idx, columns=[score_name])], axis=1)

        if larger_label_space:
            # Specific formatting for binary comparison (class 0 vs any other)
            cols_to_keep_indices = []
            expected_cols = ["Pred=" + labels[0], "Pred=" + labels[1]] # Assumes binary pred labels[0], labels[1] exist
            if expected_cols[0] in out.columns:
                cols_to_keep_indices.append(out.columns.get_loc(expected_cols[0]))
            if expected_cols[1] in out.columns:
                 cols_to_keep_indices.append(out.columns.get_loc(expected_cols[1]))
            else: # Fallback: use the second column regardless of name if first exists
                 if cols_to_keep_indices and len(out.columns) > 1:
                      cols_to_keep_indices.append(1)


            if score and score_name in out.columns:
                cols_to_keep_indices.append(out.columns.get_loc(score_name))

            if len(cols_to_keep_indices) >= 2: # Need at least the two prediction columns
                out = out.iloc[:, cols_to_keep_indices]
                new_colnames = []
                if expected_cols[0] in out.columns: new_colnames.append(out.columns[0])
                if len(out.columns) > len(new_colnames): new_colnames.append("Pred=any") # Rename second pred col
                if score and len(out.columns) > len(new_colnames): new_colnames.append(score_name)
                out.columns = new_colnames
            else:
                 print("Warning: Could not find expected columns for larger_label_space formatting.")

        return out
    # --- End of dummy custom_confusion_matrices ---


# --- Helper Functions from Notebook (Adapted for Web App) ---

def get_patients_scores(ytrue, ypred, th=0.5):
    """Calculates binary classification metrics."""
    res = dict(tn=0, fp=0, fn=0, tp=0)
    try:
        ytrue = np.array(ytrue).astype(int)
        ypred = np.array(ypred)

        # Ensure ytrue and ypred are 1D arrays for binary metrics
        if ytrue.ndim > 1:
            if ytrue.shape[1] >= 2:
                ytrue = ytrue[:, 1] # Assume second column is positive class
            elif ytrue.shape[1] == 1:
                 ytrue = ytrue.flatten()
            else: return res # Invalid shape

        if ypred.ndim > 1:
             if ypred.shape[1] >= 2:
                 ypred = ypred[:, 1] # Assume second column is positive class score
             elif ypred.shape[1] == 1:
                 ypred = ypred.flatten()
             else: return res # Invalid shape

        # Handle case where dimensions might still mismatch after flattening (e.g., empty input)
        if ytrue.shape[0] != ypred.shape[0] or ytrue.shape[0] == 0:
             print(f"Warning (get_patients_scores): Shape mismatch or empty array (True: {ytrue.shape}, Pred: {ypred.shape}). Returning empty scores.")
             return res

        # Ensure ypred has scores, not binary predictions for roc_auc_score
        ypred_scores = ypred
        ypred_binary = ypred >= th

        # Check for edge cases before calculating metrics
        if len(np.unique(ytrue)) < 2: # Only one class present
             print("Warning (get_patients_scores): Only one class present in ytrue.")
             # Calculate confusion matrix anyway, some metrics might work
             tn, fp, fn, tp = confusion_matrix(ytrue, ypred_binary, labels=[0, 1]).ravel()
             res["auroc"] = np.nan # Cannot calculate AUROC
             # Fill others based on CM
             res["tn"], res["fp"], res["fn"], res["tp"] = float(tn), float(fp), float(fn), float(tp)
             tot_neg = res["tn"] + res["fp"]
             res["fpr"] = res["fp"] / tot_neg if tot_neg > 0 else 0
             res["tnr"] = 1 - res["fpr"]
             res["spec"] = res["tnr"]
             tot_pos = res["tp"] + res["fn"]
             res["sen"] = res["tp"] / tot_pos if tot_pos > 0 else 0
             res["r"] = res["sen"] # Recall = Sensitivity
             # Precision might be ill-defined
             pred_pos = res["tp"] + res["fp"]
             res["p"] = res["tp"] / pred_pos if pred_pos > 0 else 0 # Precision
             # F1 score
             if res["p"] + res["r"] > 0:
                 res["f1score"] = 2 * (res["p"] * res["r"]) / (res["p"] + res["r"])
             else:
                  res["f1score"] = 0.0

        else: # Both classes are present
            tn, fp, fn, tp = confusion_matrix(ytrue, ypred_binary, labels=[0, 1]).ravel()
            res["tn"], res["fp"], res["fn"], res["tp"] = float(tn), float(fp), float(fn), float(tp)

            tot_neg = res["tn"] + res["fp"]
            res["fpr"] = res["fp"] / tot_neg if tot_neg > 0 else 0
            res["tnr"] = 1 - res["fpr"]
            res["spec"] = res["tnr"] # Specificity = TNR

            # Use average='binary' which assumes pos_label=1 by default
            res["p"], res["r"], res["f1score"], _ = precision_recall_fscore_support(
                ytrue, ypred_binary, average="binary", zero_division=0, labels=[0, 1]
            )
            res["sen"] = res["r"] # Sensitivity = Recall = TPR

            try:
                # Ensure ypred_scores are actual scores/probabilities
                res["auroc"] = roc_auc_score(ytrue, ypred_scores) # labels=[0,1] is default
            except ValueError as e_auc:
                 print(f"Warning (get_patients_scores): Could not calculate AUROC: {e_auc}. Setting to NaN.")
                 res["auroc"] = np.nan # Handle cases where y_pred might be constant or invalid

        # Balanced Accuracy
        res["bAcc"] = 0.5 * (res.get("sen", 0) + res.get("spec", 0))

        # Likelihood Ratios (handle division by zero)
        sen = res.get("sen", 0)
        spec = res.get("spec", 0)
        res["lr-"] = (1 - sen) / spec if spec != 0 else np.inf
        res["lr+"] = sen / (1 - spec) if spec != 1 else np.inf

    except Exception as e:
        print(f"ERROR in get_patients_scores: {e}")
        traceback.print_exc()
        # Return default dict with zeros/NaNs if error occurs
        res = {k: 0.0 for k in ["tn", "fp", "fn", "tp", "fpr", "tnr", "spec", "p", "r", "f1score", "sen", "bAcc"]}
        res["auroc"] = np.nan
        res["lr-"] = np.inf
        res["lr+"] = np.inf

    return res


def nice_format(df, ndigits=2, short=True):
    """Formats pandas Series/DataFrame with median (IQR or Q1-Q3)."""
    if df is None or df.empty:
        return pd.Series() if isinstance(df, pd.Series) else pd.DataFrame()

    # Ensure data is numeric before calculating quantiles
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    if isinstance(numeric_df, pd.Series):
        numeric_df = numeric_df.to_frame().T # Handle Series input

    med = numeric_df.median(0)
    q1 = numeric_df.quantile(0.25, axis=0)
    q3 = numeric_df.quantile(0.75, axis=0)

    med_str = med.round(ndigits).astype(str)

    if short:
        iqr = (q3 - q1).round(ndigits).astype(str)
        out_str = med_str + " (" + iqr + ")"
    else:
        q1_str = q1.round(ndigits).astype(str)
        q3_str = q3.round(ndigits).astype(str)
        out_str = med_str + " (" + q1_str + " - " + q3_str + ")"

    # Handle potential NaNs that became "nan" strings
    out_str = out_str.replace("nan", "N/A").replace("N/A (N/A)", "N/A").replace("N/A (N/A - N/A)", "N/A")

    # Return with original index and columns if input was DataFrame
    if isinstance(df, pd.DataFrame):
        out_df = pd.DataFrame(index=df.index, columns=df.columns)
        # Fill the output DataFrame - this needs care if format changed structure
        # Assuming nice_format operates column-wise and returns a Series:
        if isinstance(out_str, pd.Series):
             out_df = out_str.to_frame().T # Simplified: assumes single row output format
             out_df.index = ["Median (IQR)"] if short else ["Median (Q1-Q3)"]
             out_df.columns = df.columns # Ensure columns match input
             return out_df
        else: # Should not happen if input is DF
            return pd.DataFrame([out_str], columns=df.columns, index=["Median (IQR)" if short else "Median (Q1-Q3)"])

    return out_str # Return Series if input was Series


def rename_model(m):
    """Provides a nicer name string for model filenames/IDs."""
    if not isinstance(m, str): return str(m) # Handle non-string input

    # Example Parsing Rules (add more specific ones as needed)
    parsed = parse("nflow{}.4.[{}_{}].{}.50", m)
    if parsed is not None:
        try:
            h, n1, n2, t = parsed
            t1 = "G" if t == "fit3" else "D"
            t2 = "D" if t == "fit3" else "G"
            return f"NF h={h}, {t1}({n1}) -> {t2}({n2})"
        except Exception: pass # Ignore parsing errors

    parsed = parse("gmm{}.diag",m)
    if parsed is not None:
        try:
            n_comp = parsed[0]
            return f"NB (GMM {n_comp})" # Naive Bayes / Gaussian Mixture Model
        except Exception: pass

    # Generic replacements
    if 'xgboost' in m.lower(): return "XGBoost"
    if 'logistic' in m.lower(): return "Logistic Reg."
    if 'gmm' in m.lower(): return "GMM" # General GMM if specific parse failed
    if 'nflow' in m.lower(): return "Normalizing Flow" # General NF

    return m # Return original name if no rule matches


def binarize_multiclass(y):
    """Converts multi-class target/prediction array to binary (class 0 vs. any other)."""
    if y is None or not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] <= 1:
        # Return as is if already binary, 1D, invalid, or empty
        # Check shape[0] > 0?
        if y is not None and y.ndim == 1: # If 1D, assume it's already class labels or scores
             # Need to decide how to convert 1D to 2D binary [class0, class1] format
             # Assume 0 is negative, >0 is positive? Or needs explicit mapping?
             # Let's assume 0 is class 0, others are class 1 for now
             is_zero = (y == 0).astype(float)
             is_other = (y != 0).astype(float)
             return np.stack([is_zero, is_other], axis=1)
        return y # Return others (like None, 2D with 1 col) as is

    try:
        # Check if probabilities (contain values not 0 or 1)
        is_probs = not np.all((y == 0) | (y == 1) | np.isnan(y)) # Ignore NaNs in check

        if is_probs:
            # Sum probabilities of non-zero classes (cols 1 onwards)
            prob_zero = y[:, 0]
            prob_other = np.nansum(y[:, 1:], axis=1) # Use nansum

            # Normalize the new binary probabilities (handle potential zero sum)
            total = prob_zero + prob_other
            # Avoid division by zero; if total is 0 or NaN, keep probs as 0 or NaN
            prob_zero_norm = np.divide(prob_zero, total, out=np.full_like(prob_zero, np.nan), where=(total != 0) & (~np.isnan(total)))
            prob_other_norm = np.divide(prob_other, total, out=np.full_like(prob_other, np.nan), where=(total != 0) & (~np.isnan(total)))

            # Handle cases where total was 0 or NaN - resulting probs should reflect this (e.g., NaN)
            prob_zero_norm[np.isnan(prob_zero)] = np.nan # Propagate original NaNs
            prob_other_norm[np.isnan(prob_other)] = np.nan # Propagate original NaNs

            out = np.stack([prob_zero_norm, prob_other_norm], axis=1)

        else:
            # Assume one-hot encoding or multi-label binary
            is_zero = y[:, 0].astype(float) # Class 0 indicator
            # Check if any other class (cols 1 onwards) is 1
            is_other = (np.nansum(y[:, 1:], axis=1) > 0).astype(float) # Use nansum, check > 0

            # Handle potential multi-label nature if needed (e.g., if both 0 and other can be true)
            # Current logic: if 'other' is true, classify as 'other' (binary)
            # is_zero[is_other == 1] = 0 # This assumes mutual exclusivity in the binary output

            out = np.stack([is_zero, is_other], axis=1)

        return out

    except Exception as e:
        print(f"ERROR in binarize_multiclass: {e}")
        traceback.print_exc()
        return None # Return None on error


def combine_tt(x):
    """Selects the time-to-event value with the minimum absolute value across different event types for each time point."""
    """Input: (n_timelines, T), Output: (T,)"""
    if x is None or not isinstance(x, np.ndarray): return np.array([])
    if x.ndim == 1: return x # Only one timeline
    if x.shape[0] == 0 or x.shape[1] == 0: return np.array([]) # No timelines or no time points

    try:
        # Calculate absolute values, ignoring NaNs for argmin selection if possible
        abs_x = np.abs(x)
        # Find the index of the minimum absolute value in each column (time point)
        # Handle columns containing only NaNs - argmin raises error
        min_indices = np.full(x.shape[1], -1, dtype=int) # Initialize with invalid index
        valid_cols = ~np.all(np.isnan(abs_x), axis=0) # Columns with at least one non-NaN value
        if np.any(valid_cols):
             min_indices[valid_cols] = np.nanargmin(abs_x[:, valid_cols], axis=0)

        # Select the element using the calculated indices
        # Need to handle the invalid indices (-1) -> result should be NaN
        result = np.full(x.shape[1], np.nan)
        valid_selection = (min_indices != -1)
        if np.any(valid_selection):
             row_idx = min_indices[valid_selection]
             col_idx = np.arange(x.shape[1])[valid_selection]
             result[valid_selection] = x[row_idx, col_idx]

        return result

    except Exception as e:
        print(f"ERROR in combine_tt: {e}")
        traceback.print_exc()
        return np.full(x.shape[1], np.nan) # Return NaNs on error


# Global list to store errors encountered during parallel processing
process_errors = []

def _aggregate_scores_run(d_tuple, theset="val", th=None, class0_name="not_target__healthy"):
    """Helper function to process results for a single model file (for multiprocessing)."""
    global process_errors
    irun, d = d_tuple # Unpack index and data
    run_results = {} # Use local dict to avoid clashes if global vars were used

    # Initialize results dictionaries specific to this run/config key
    aurocs_results_k = {}
    prs_results_k = {}
    aurocs_scores_k = {}
    patwise_predictions_k = {}
    pos_pat_k = {}
    all_pat_k = {}
    Cfgs_k = {}
    k = None # Initialize config key

    try:
        if d is None:
            process_errors.append(f"Run {irun}: Input data is None.")
            return None

        # Basic check for essential keys
        if "cfg" not in d or "results" not in d:
             process_errors.append(f"Run {irun}: Skipping due to missing 'cfg' or 'results'.")
             return None

        cfg = d["cfg"]
        # --- Configuration Extraction ---
        labeling_cfg = cfg.get("labeling", {})
        patients_cfg = cfg.get("patients", {})
        feats_cfg = cfg.get("feats", {})
        model_cfg = cfg.get("model", {})

        # --- Model Configuration ---
        run_cfg_dict = {
            "pop": patients_cfg.get("fname", "N/A"),
            "features": feats_cfg.get("feat_mode", "N/A"),
            "demos": feats_cfg.get("demos", "N/A"),
            "rback": labeling_cfg.get("restrict_bw", "N/A"),
            "model_name": model_cfg.get("name", f"unknown_model_{irun}")
        }
        model_name = run_cfg_dict["model_name"]

        # Create unique key for this configuration (sorting ensures consistency)
        k = ", ".join([f"{key}={val}" for key, val in sorted(run_cfg_dict.items())])

        # Initialize lists for this specific key 'k' within the run's local dicts
        aurocs_scores_k[k] = []
        prs_results_k[k] = []
        aurocs_results_k[k] = []
        patwise_predictions_k[k] = {}
        all_pat_k[k] = []
        pos_pat_k[k] = []

        # Process each result fold within the file
        for i_res, _r in enumerate(d.get("results", [])):
            try:
                # Decompress if necessary (handle potential errors)
                try:
                     r = decompress_obj(_r)
                except Exception as e_decomp:
                     process_errors.append(f"Run {irun}, Fold {i_res}: Decompression failed: {e_decomp}")
                     continue # Skip fold

                if not isinstance(r, dict) or theset not in r:
                     process_errors.append(f"Run {irun}, Fold {i_res}: Invalid format or missing set '{theset}'.")
                     continue

                df_set = r[theset] # DataFrame for the current set (train/val/test)
                if not isinstance(df_set, pd.DataFrame) or df_set.empty:
                     process_errors.append(f"Run {irun}, Fold {i_res}: Data for set '{theset}' is empty or not a DataFrame.")
                     continue

                # --- Identify Target and Prediction Columns ---
                # Be robust: handle potential model name prefixes in predictions
                pred_targets = sorted([c for c in df_set.columns if c.startswith("pred__")])
                if not pred_targets:
                     process_errors.append(f"Run {irun}, Fold {i_res}: No prediction columns ('pred__*') found.")
                     continue

                # --- Infer true targets based on prediction columns ---
                # This requires a consistent naming convention. Assume 'pred__TargetName' or 'pred__Model__TargetName' maps to 'TargetName'
                true_targets = []
                possible_true_cols = [c for c in df_set.columns if not c.startswith("pred__") and not c.startswith("feats__") and not c.startswith("ids__") and not c.startswith("tl__") and not c.startswith("log_px__") and not c.startswith("time_to")]

                # Prioritize direct mapping first
                for pt in pred_targets:
                     tt_inferred = pt.replace(f"pred__{model_name}__", "").replace("pred__", "")
                     if tt_inferred in df_set.columns:
                         true_targets.append(tt_inferred)
                     # Fallback: look for standard names if direct map fails
                     elif tt_inferred == class0_name.replace("target__","").replace("not_target__","") and class0_name in df_set.columns:
                          true_targets.append(class0_name)
                     elif f"target__{tt_inferred}" in df_set.columns:
                          true_targets.append(f"target__{tt_inferred}")


                # If inference failed or lengths don't match, try a simpler approach
                if len(true_targets) != len(pred_targets):
                     # Try finding one 'negative' class and assume others are positive targets
                     found_class0 = class0_name in df_set.columns
                     potential_pos_targets = sorted([c for c in possible_true_cols if c != class0_name and c.startswith("target__")])

                     if found_class0 and len(potential_pos_targets) == len(pred_targets) - 1:
                         true_targets = [class0_name] + potential_pos_targets
                         print(f"INFO: Run {irun}, Fold {i_res}: Using inferred target structure: {true_targets}")
                     else:
                         process_errors.append(f"Run {irun}, Fold {i_res}: Cannot reliably determine true targets matching predictions ({len(pred_targets)} preds found). Preds: {pred_targets}, Possible True: {possible_true_cols}")
                         continue # Cannot proceed with this fold

                # Final check: ensure all inferred true targets exist
                if not all(tt in df_set.columns for tt in true_targets):
                     missing = [tt for tt in true_targets if tt not in df_set.columns]
                     process_errors.append(f"Run {irun}, Fold {i_res}: Inferred true target columns missing: {missing}")
                     continue


                # --- Get True and Predicted Values ---
                try:
                     y_true_multi = df_set[true_targets].values
                     y_pred_multi = df_set[pred_targets].values
                except KeyError as e_key:
                     process_errors.append(f"Run {irun}, Fold {i_res}: KeyError accessing columns: {e_key}")
                     continue
                except Exception as e_val:
                     process_errors.append(f"Run {irun}, Fold {i_res}: Error getting values: {e_val}")
                     continue


                # --- Binarize for Standard Metrics ---
                y_true_bin = binarize_multiclass(y_true_multi)
                y_pred_bin = binarize_multiclass(y_pred_multi)

                if y_true_bin is None or y_pred_bin is None or y_true_bin.shape[0] == 0 or y_true_bin.shape != y_pred_bin.shape:
                     process_errors.append(f"Run {irun}, Fold {i_res}: Binarized data invalid (True: {y_true_bin.shape if y_true_bin is not None else 'None'}, Pred: {y_pred_bin.shape if y_pred_bin is not None else 'None'}). Skipping fold.")
                     continue
                if y_true_bin.shape[1] < 2:
                    process_errors.append(f"Run {irun}, Fold {i_res}: Binarized data has < 2 columns ({y_true_bin.shape}). Cannot calculate binary metrics. Skipping fold.")
                    continue

                # Use column 1 (index 1) as the positive class/score
                y_true_flat = y_true_bin[:, 1]
                y_pred_scores = y_pred_bin[:, 1]

                # --- Calculate Metrics ---
                # Check for NaNs in scores which break metrics
                nan_pred_mask = np.isnan(y_pred_scores)
                if np.all(nan_pred_mask):
                     process_errors.append(f"Run {irun}, Fold {i_res}: All predicted scores are NaN. Skipping metrics.")
                     # Append dummy/NaN results?
                     aurocs_results_k[k].append((np.array([0,1]), np.array([0,1]), np.array([]), np.nan))
                     prs_results_k[k].append((np.array([]), np.array([]), np.array([]), np.nan))
                     aurocs_scores_k[k].append(get_patients_scores(y_true_flat, y_pred_scores, th=0.5)) # Will likely return NaNs/defaults
                     continue # Don't process patients if scores are bad

                if np.any(nan_pred_mask):
                    print(f"Warning: Run {irun}, Fold {i_res}: Found {nan_pred_mask.sum()} NaN scores. Excluding them from metric calculation.")
                    y_true_flat_valid = y_true_flat[~nan_pred_mask]
                    y_pred_scores_valid = y_pred_scores[~nan_pred_mask]
                else:
                    y_true_flat_valid = y_true_flat
                    y_pred_scores_valid = y_pred_scores

                if len(np.unique(y_true_flat_valid)) >= 2: # Need at least two classes for ROC/PR
                    # AUROC
                    fpr, tpr, roc_thresholds = roc_curve(y_true_flat_valid, y_pred_scores_valid)
                    auroc_val = roc_auc_score(y_true_flat_valid, y_pred_scores_valid)
                    aurocs_results_k[k].append((fpr, tpr, roc_thresholds, auroc_val))

                    # Find optimal threshold for scoring (e.g., maximizing TPR*(1-FPR)) or use fixed 0.5
                    opt_th = th # Use fixed threshold from argument
                    if opt_th is None: # Find optimal from ROC if th not provided
                        if len(tpr) > 0 and len(fpr) > 0:
                            imax = np.argmax(tpr * (1 - fpr))
                            # Ensure imax is valid index for roc_thresholds
                            opt_th = roc_thresholds[min(imax, len(roc_thresholds)-1)] if len(roc_thresholds) > 0 else 0.5
                        else:
                            opt_th = 0.5 # Fallback


                    # Calculate scores using the chosen threshold (on potentially filtered data)
                    current_scores = get_patients_scores(y_true_flat_valid, y_pred_scores_valid, th=opt_th)
                    aurocs_scores_k[k].append(current_scores)

                    # Precision-Recall
                    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_flat_valid, y_pred_scores_valid)
                    auprc_val = average_precision_score(y_true_flat_valid, y_pred_scores_valid)
                    prs_results_k[k].append((precisions, recalls, pr_thresholds, auprc_val)) # Appending AUPRC

                else:
                    # Handle cases with only one class present in this fold (after NaN filtering)
                    process_errors.append(f"Run {irun}, Fold {i_res}: Only one class present after NaN filter. Cannot calculate AUROC/AUPRC.")
                    aurocs_results_k[k].append((np.array([0, 1]), np.array([0, 1]), np.array([]), np.nan)) # Dummy ROC
                    prs_results_k[k].append((np.array([]), np.array([]), np.array([]), np.nan)) # Dummy PR
                    # Calculate scores with a fixed threshold (0.5) if possible
                    aurocs_scores_k[k].append(get_patients_scores(y_true_flat_valid, y_pred_scores_valid, th=0.5))


                # --- Process Patient-wise Predictions ---
                # Use binarized labels [class0, positive_class]
                lbl = ["none", "positive"] # Labels for the binarized output

                # Need unique patient identifier column ('ids__uid' assumed)
                if "ids__uid" not in df_set.columns:
                     process_errors.append(f"Run {irun}, Fold {i_res}: Missing patient ID column 'ids__uid'. Cannot process patient-wise.")
                     continue # Skip patient processing for this fold

                thevalpats = df_set["ids__uid"].unique()

                for ids__uid in thevalpats:
                    pat_idx = df_set["ids__uid"] == ids__uid
                    if not np.any(pat_idx): continue # Should not happen

                    # Extract data for this patient
                    ytrue_pat_multi = df_set.loc[pat_idx, true_targets].values
                    ypred_pat_multi = df_set.loc[pat_idx, pred_targets].values

                    # Binarize patient data
                    ytrue_pat_bin = binarize_multiclass(ytrue_pat_multi)
                    ypred_pat_bin = binarize_multiclass(ypred_pat_multi)

                    # Check for validity after binarization
                    if ytrue_pat_bin is None or ypred_pat_bin is None or ytrue_pat_bin.shape != ypred_pat_bin.shape or ytrue_pat_bin.shape[1]<2:
                         process_errors.append(f"Run {irun}, Fold {i_res}, Pat {ids__uid}: Invalid binarized patient data.")
                         continue

                    # Extract time variables (handle missing columns gracefully)
                    pna_days_col = "feats__pna_days" # Example standard name
                    pna_h_col = "tl__pna_h" # Example standard name
                    tt_cols = sorted([s for s in df_set.columns if s.startswith("time_to")])

                    pna_days = df_set.loc[pat_idx, pna_days_col].values if pna_days_col in df_set.columns else np.full(pat_idx.sum(), np.nan)
                    pna_h = df_set.loc[pat_idx, pna_h_col].values if pna_h_col in df_set.columns else np.full(pat_idx.sum(), np.nan)
                    time_to = df_set.loc[pat_idx, tt_cols].values if tt_cols else np.full((pat_idx.sum(), 0), np.nan)


                    # Extract log likelihood if available
                    log_px_col = f"log_px__{model_name}" # Requires model name consistency
                    log_px = df_set.loc[pat_idx, log_px_col].values if log_px_col in df_set.columns else np.full(pat_idx.sum(), np.nan)

                    pat_data = {
                        "ytrue": ytrue_pat_bin, # Binarized [class0, positive]
                        "ypred": ypred_pat_bin, # Binarized scores [class0, positive]
                        "ytrue_multi": ytrue_pat_multi, # Original multi-class
                        "ypred_multi": ypred_pat_multi, # Original multi-class scores/preds
                        "pna_days": pna_days,
                        "pna_h": pna_h,
                        "time_to": time_to, # Shape (n_timepoints, n_events)
                        "model_name": model_name,
                        "lbl": lbl, # Corresponds to binarized structure
                        "log_px": log_px
                    }

                    # Store patient data under the config key 'k'
                    if ids__uid not in patwise_predictions_k[k]:
                        patwise_predictions_k[k][ids__uid] = []
                    patwise_predictions_k[k][ids__uid].append(pat_data) # Append data from this fold

                    # Track unique patient IDs and positive patient IDs for this fold/config
                    if ids__uid not in all_pat_k[k]:
                        all_pat_k[k].append(ids__uid)
                    # Check if patient is positive based on binarized 'true' label (presence of class > 0 at any timepoint)
                    if np.any(ytrue_pat_bin[:, 1] > 0):
                         if ids__uid not in pos_pat_k[k]:
                             pos_pat_k[k].append(ids__uid)

            except Exception as e_fold:
                error_msg = f"Run {irun}, Fold {i_res}, Config '{k}': Error processing fold: {e_fold}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                process_errors.append(error_msg)
                continue # Skip to next fold

        # Store the configuration for this key 'k'
        Cfgs_k[k] = run_cfg_dict

        # Return results collected for this file/configuration 'k'
        return {
            "aurocs_results": aurocs_results_k,
            "prs_results": prs_results_k,
            "aurocs_scores": aurocs_scores_k,
            "patwise_predictions": patwise_predictions_k,
            "pos_pat": pos_pat_k,
            "all_pat": all_pat_k,
            "Cfgs": Cfgs_k,
        }

    except Exception as e_run:
        error_msg = f"Run {irun}, Config '{k}': Major error processing run: {e_run}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        process_errors.append(error_msg)
        return None # Indicate failure for this run


def aggregate_scores_run(data_list, theset="val", th=None, n_jobs=4, class0_name="not_target__healthy"):
    """Aggregates scores across multiple model result files using multiprocessing."""
    global process_errors
    process_errors = [] # Reset errors for this aggregation run

    if not data_list:
        return {}, ["No data provided to aggregate."]

    # Prepare data tuples (index, data) for mapping
    data_tuples = list(enumerate(data_list))

    # Create partial function with fixed arguments
    func = partial(_aggregate_scores_run, theset=theset, th=th, class0_name=class0_name)

    print(f"Starting aggregation with {min(n_jobs, len(data_tuples))} processes for {len(data_tuples)} files...")
    results_list = []
    try:
        # Use Pool for parallel processing
        # Use context manager to ensure pool cleanup
        with Pool(processes=min(n_jobs, len(data_tuples))) as pool:
             # Use map_async for potentially better error handling? map is simpler.
             results_list = pool.map(func, data_tuples)
        print("Aggregation finished.")
    except Exception as e_pool:
         error_msg = f"Error during multiprocessing pool execution: {e_pool}"
         print(f"ERROR: {error_msg}")
         traceback.print_exc()
         process_errors.append(error_msg)
         # Try to continue with any results obtained before the error?
         # results_list might be incomplete or contain exceptions.

    # Filter out None results (from skipped or failed runs)
    valid_results_list = [res for res in results_list if res is not None]

    if not valid_results_list:
        print("Warning: No valid results were aggregated.")
        return {}, process_errors # Return empty dict and errors

    # Combine results from all processes/files into a single dictionary structure
    combined_results = {
        "aurocs_results": {}, "prs_results": {}, "aurocs_scores": {},
        "patwise_predictions": {}, "pos_pat": {}, "all_pat": {}, "Cfgs": {}
    }

    # Get all unique top-level keys across all valid results (should be consistent)
    # keys_to_combine = list(valid_results_list[0].keys()) # Assumes first result is representative
    keys_to_combine = combined_results.keys() # Use the predefined structure

    for res_dict in valid_results_list:
        # Check if the result dict has the expected structure
        if not all(k in res_dict for k in keys_to_combine):
             process_errors.append(f"Aggregated result has missing keys: {res_dict.keys()}. Skipping.")
             continue

        # Merge data for each category (aurocs_results, prs_results, etc.)
        for category_key in keys_to_combine:
            category_data = res_dict[category_key] # This is a dict like {'config_k1': data, 'config_k2': data}
            if not isinstance(category_data, dict):
                 process_errors.append(f"Expected dict for category '{category_key}', got {type(category_data)}. Skipping merge.")
                 continue

            for config_key, data_value in category_data.items():
                # Initialize the config_key in combined_results if not present
                if config_key not in combined_results[category_key]:
                    if category_key in ["aurocs_results", "prs_results", "aurocs_scores"]:
                        combined_results[category_key][config_key] = []
                    elif category_key in ["pos_pat", "all_pat"]:
                         combined_results[category_key][config_key] = [] # Lists of patient IDs
                    elif category_key == "patwise_predictions":
                        combined_results[category_key][config_key] = {} # Dict of {pat_id: [fold_data_list]}
                    elif category_key == "Cfgs":
                         combined_results[category_key][config_key] = {} # Dict for config details
                    # Add other initializations if needed

                # Append/Extend/Update based on the category type
                if category_key in ["aurocs_results", "prs_results", "aurocs_scores"]:
                    if isinstance(data_value, list):
                        combined_results[category_key][config_key].extend(data_value)
                    else:
                        process_errors.append(f"Expected list for {category_key}[{config_key}], got {type(data_value)}. Skipping merge.")
                elif category_key in ["pos_pat", "all_pat"]:
                     if isinstance(data_value, list):
                         # Add only unique patient IDs
                         existing_pats = set(combined_results[category_key][config_key])
                         new_pats = [p for p in data_value if p not in existing_pats]
                         combined_results[category_key][config_key].extend(new_pats)
                     else:
                         process_errors.append(f"Expected list for {category_key}[{config_key}], got {type(data_value)}. Skipping merge.")
                elif category_key == "patwise_predictions":
                     if isinstance(data_value, dict):
                         # Merge patient dictionaries (append fold data lists)
                         for pat_id, pat_fold_data_list in data_value.items():
                             if pat_id not in combined_results[category_key][config_key]:
                                 combined_results[category_key][config_key][pat_id] = []
                             if isinstance(pat_fold_data_list, list):
                                 combined_results[category_key][config_key][pat_id].extend(pat_fold_data_list)
                             else:
                                  process_errors.append(f"Expected list for patwise_predictions[{config_key}][{pat_id}], got {type(pat_fold_data_list)}. Skipping.")
                     else:
                         process_errors.append(f"Expected dict for {category_key}[{config_key}], got {type(data_value)}. Skipping merge.")
                elif category_key == "Cfgs":
                     if isinstance(data_value, dict):
                         # Update config dict (last one wins if keys overlap, should be same)
                         combined_results[category_key][config_key].update(data_value)
                     else:
                         process_errors.append(f"Expected dict for {category_key}[{config_key}], got {type(data_value)}. Skipping merge.")

    # Garbage collect after processing potentially large data
    gc.collect()

    return combined_results, process_errors


def density_plot(time_to_event_a, risks_a, title="Risk Density Plot", xlabel="Time (hours)", ylabel="Risk Score",
                 font_size=12, xlim_l=None, nh=4, nbins_=10, bwidth=0.1, bstart=0):
    """Creates a 2D density plot of risk scores over time using Plotly."""
    if time_to_event_a is None or risks_a is None or len(time_to_event_a) == 0 or len(risks_a) == 0:
        print("Warning (density_plot): No data provided.")
        # Return an empty figure object or a figure with an annotation
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)", xaxis_title=xlabel, yaxis_title=ylabel)
        fig.add_annotation(text="No data available for this plot.", showarrow=False, y=0.5, x=0.5)
        return fig

    # Ensure inputs are numpy arrays and flattened
    time_to_event_a = np.asarray(time_to_event_a).flatten()
    risks_a = np.asarray(risks_a).flatten()

    if time_to_event_a.shape != risks_a.shape:
        print(f"Warning (density_plot): Shape mismatch between time ({time_to_event_a.shape}) and risks ({risks_a.shape}).")
        # Try to use the minimum length? Or return empty?
        min_len = min(len(time_to_event_a), len(risks_a))
        time_to_event_a = time_to_event_a[:min_len]
        risks_a = risks_a[:min_len]
        if min_len == 0:
             fig = go.Figure()
             fig.update_layout(title=f"{title} (Shape Mismatch/Empty)", xaxis_title=xlabel, yaxis_title=ylabel)
             fig.add_annotation(text="Input shape mismatch or empty data.", showarrow=False, y=0.5, x=0.5)
             return fig


    dall = pd.DataFrame({
        "x": time_to_event_a,
        "y": risks_a
    })
    dall.dropna(inplace=True) # Remove rows with NaNs in either column

    # Filter by time limits if provided
    if xlim_l and len(xlim_l) == 2:
        dall = dall[(dall["x"] >= xlim_l[0]) & (dall["x"] <= xlim_l[1])]

    if dall.empty:
        print("Warning (density_plot): No data remains after filtering.")
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data After Filter)", xaxis_title=xlabel, yaxis_title=ylabel)
        fig.add_annotation(text="No data available after applying filters.", showarrow=False, y=0.5, x=0.5)
        return fig

    # Bin time axis (x)
    # Ensure nh is positive to avoid division by zero or infinite loop
    nh = max(nh, 1e-6) # Use a small positive number if nh is zero or negative
    dall["x_binned"] = (dall["x"] // nh) * nh

    # Define risk bins (y)
    # Ensure nbins_ is positive
    nbins_ = max(nbins_, 1)
    risk_bins = np.linspace(bstart, bstart + nbins_ * bwidth, nbins_ + 1)

    # Group by binned time and calculate histogram for risk within each time bin
    z_list = []
    x_bins_used = []
    # Sort by the time bin to ensure correct order in the plot
    grouped = dall.groupby("x_binned")

    # Get all unique time bins present in the data
    all_time_bins = sorted(dall["x_binned"].unique())

    for time_bin in all_time_bins:
        group = grouped.get_group(time_bin) # Get group for this specific time bin
        # Calculate histogram within the risk_bins range
        hist, _ = np.histogram(group["y"], bins=risk_bins, density=True)
        z_list.append(hist)
        x_bins_used.append(time_bin) # Store the time bin start

    if not z_list:
        print("Warning (density_plot): No data groups found after binning.")
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Binned Data)", xaxis_title=xlabel, yaxis_title=ylabel)
        fig.add_annotation(text="No data available after binning.", showarrow=False, y=0.5, x=0.5)
        return fig

    z = np.array(z_list).T # Transpose: rows=risk bins, columns=time bins
    x_labels = [f"{t:.1f}" for t in x_bins_used] # Time labels (use float format)
    # y_labels: Center of risk bins for better axis representation
    y_bin_centers = (risk_bins[:-1] + risk_bins[1:]) / 2
    y_labels_text = [f"{risk_bins[i]:.2f}-{risk_bins[i+1]:.2f}" for i in range(nbins_)] # Labels for hover/colorbar


    try:
        fig = go.Figure(data=go.Heatmap(
                        z=z,
                        x=x_labels, # Use the formatted time labels for the axis
                        y=y_bin_centers, # Use bin centers for y-axis positioning
                        colorscale='RdBu_r',
                        colorbar=dict(title='Density', tickvals=y_bin_centers[::max(1, nbins_//5)], ticktext=y_labels_text[::max(1, nbins_//5)]), # Show fewer labels on colorbar
                        # Custom hover text
                        customdata=np.array(x_bins_used), # Pass time bin starts for hover
                        hovertemplate=(f"{xlabel}: %{{customdata:.1f}} to %{{customdata:.1f}}+{nh:.1f}<br>" +
                                     f"{ylabel}: %{{y:.2f}}<br>" + # This will show bin center
                                     "Density: %{z:.3f}<extra></extra>")
                        ))


        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            yaxis=dict(tickvals=y_bin_centers[::max(1, nbins_//5)], ticktext=y_labels_text[::max(1, nbins_//5)]), # Show fewer labels on y-axis
            font={"size": font_size},
            template="plotly_white", # Use a clean template
            xaxis=dict(type='category') # Treat x-axis as categorical labels based on bins
        )
        fig.update_xaxes(title_font={"size": font_size + 2})
        fig.update_yaxes(title_font={"size": font_size + 2})
        # fig.update_coloraxes(colorbar_title_font={"size": font_size + 2}) # Already set in colorbar dict

    except Exception as e_plotly:
        print(f"ERROR generating Plotly density plot: {e_plotly}")
        traceback.print_exc()
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Plotting Error)", xaxis_title=xlabel, yaxis_title=ylabel)
        fig.add_annotation(text="Error occurred during plot generation.", showarrow=False)

    return fig


def plot_example_pat(pat_data_list, ids__uid, xlim_l=None, layout_d=None, width=None, height=None, xname="pna_days"):
    """Plots risk prediction trajectory for a single patient using Plotly."""
    if not pat_data_list:
        print(f"Warning (plot_example_pat): No data found for patient {ids__uid}")
        fig = go.Figure()
        fig.update_layout(title=f"Patient {ids__uid[:8]} Risk Trajectory (No Data)", width=width, height=height)
        fig.add_annotation(text="No data available for this patient.", showarrow=False)
        return fig

    # Use data from the first fold for simplicity
    # TODO: Consider aggregating across folds if multiple exist (e.g., average risk?)
    pat_d = pat_data_list[0]

    # Extract data, handling potential missing keys and ensuring correct shapes
    pos_pat_x_raw = pat_d.get(xname)
    pos_pat_y_bin = pat_d.get("ypred") # Binarized prediction scores [class0, class1]
    pos_pat_ytrue_bin = pat_d.get("ytrue") # Binarized true labels

    # Validate extracted data
    if pos_pat_x_raw is None or pos_pat_y_bin is None or pos_pat_ytrue_bin is None:
        print(f"Warning (plot_example_pat): Missing required data fields ({xname}, ypred, ytrue) for patient {ids__uid}")
        fig = go.Figure()
        fig.update_layout(title=f"Patient {ids__uid[:8]} Risk Trajectory (Missing Data)", width=width, height=height)
        fig.add_annotation(text="Missing required data fields.", showarrow=False)
        return fig

    # Ensure numpy arrays and correct dimensions
    pos_pat_x_raw = np.asarray(pos_pat_x_raw).flatten()
    pos_pat_y_bin = np.asarray(pos_pat_y_bin)
    pos_pat_ytrue_bin = np.asarray(pos_pat_ytrue_bin)

    if pos_pat_y_bin.ndim != 2 or pos_pat_y_bin.shape[1] < 2 or \
       pos_pat_ytrue_bin.ndim != 2 or pos_pat_ytrue_bin.shape[1] < 2 or \
       len(pos_pat_x_raw) != pos_pat_y_bin.shape[0] or \
       len(pos_pat_x_raw) != pos_pat_ytrue_bin.shape[0]:
        print(f"Warning (plot_example_pat): Data shape mismatch for patient {ids__uid}. "
              f"X: {pos_pat_x_raw.shape}, Y_pred: {pos_pat_y_bin.shape}, Y_true: {pos_pat_ytrue_bin.shape}")
        fig = go.Figure()
        fig.update_layout(title=f"Patient {ids__uid[:8]} Risk Trajectory (Data Shape Error)", width=width, height=height)
        fig.add_annotation(text="Data shape mismatch.", showarrow=False)
        return fig

    # Use the score for the positive class (column 1)
    pos_pat_y_score = pos_pat_y_bin[:, 1]
    pos_pat_ytrue_label = pos_pat_ytrue_bin[:, 1] # Binary label for positive class

    # Determine xlabel and process x-axis data based on xname
    pos_pat_x = pos_pat_x_raw # Default to raw data
    if xname == "pna_days":
        xlabel = "Postnatal Age (Days)"
    elif "time_to" in xname:
         xlabel = "Time to Nearest Event (Hours)" # Assuming hours
         # 'time_to' might be multi-dimensional (n_timepoints, n_events)
         time_to_data = pat_d.get('time_to')
         if time_to_data is not None and isinstance(time_to_data, np.ndarray) and time_to_data.ndim > 0:
             if time_to_data.ndim > 1:
                 pos_pat_x = combine_tt(time_to_data.T) # Transpose to (n_events, n_timepoints) before combining
             else:
                 pos_pat_x = time_to_data.flatten()

             # Ensure x-axis length still matches y-axis after processing
             if len(pos_pat_x) != len(pos_pat_y_score):
                  print(f"Warning (plot_example_pat): Length mismatch after processing 'time_to' for patient {ids__uid}. X:{len(pos_pat_x)}, Y:{len(pos_pat_y_score)}")
                  # Attempt to align? Or fallback? Fallback to raw x for now.
                  pos_pat_x = pos_pat_x_raw # Fallback
                  xlabel = f"Original X-axis ({xname})" # Change label if fallback
                  if len(pos_pat_x) != len(pos_pat_y_score): # Check again after fallback
                       print(f"ERROR (plot_example_pat): Cannot align X and Y axes for patient {ids__uid}. Aborting plot.")
                       fig = go.Figure()
                       fig.update_layout(title=f"Patient {ids__uid[:8]} Risk Trajectory (Axis Alignment Error)", width=width, height=height)
                       fig.add_annotation(text="Cannot align X and Y axes.", showarrow=False)
                       return fig
         else:
              print(f"Warning (plot_example_pat): No valid 'time_to' data found for x-axis for patient {ids__uid}. Using raw '{xname}'.")
              xlabel = f"Original X-axis ({xname})" # Use raw xname if time_to fails
    else:
        xlabel = xname # Default label

    # Create DataFrame for easier handling, filtering, and potential smoothing
    df_pat = pd.DataFrame({
        'x': pos_pat_x,
        'risk': pos_pat_y_score,
        'true_label': pos_pat_ytrue_label
    })
    df_pat.dropna(subset=['x', 'risk', 'true_label'], inplace=True) # Drop rows with NaNs in essential columns
    df_pat.sort_values(by='x', inplace=True) # Ensure data is sorted by time/x-axis

    # Filtering by xlim_l
    if xlim_l and len(xlim_l) == 2:
        df_pat = df_pat[(df_pat['x'] >= xlim_l[0]) & (df_pat['x'] <= xlim_l[1])]

    if df_pat.empty:
        print(f"Warning (plot_example_pat): No data points remain for patient {ids__uid} after filtering/cleaning.")
        fig = go.Figure()
        fig.update_layout(title=f"Patient {ids__uid[:8]} Risk Trajectory (No Data After Filter)", width=width, height=height)
        fig.add_annotation(text="No data points remain after filtering.", showarrow=False)
        return fig


    # --- Plotting ---
    fig = go.Figure()

    # 1. True Label Indicator (Shaded Regions or Background Color Changes)
    # Find consecutive blocks where true_label is 1
    df_pat['label_change'] = df_pat['true_label'].diff().fillna(0) != 0
    label_intervals = df_pat[df_pat['label_change']].index.tolist()
    # Add start and end indices
    if 0 not in label_intervals: label_intervals.insert(0, 0)
    if df_pat.index[-1] not in label_intervals: label_intervals.append(df_pat.index[-1] + 1) # Use index+1 for end

    for i in range(len(label_intervals) - 1):
        start_idx = label_intervals[i]
        end_idx = label_intervals[i+1] -1 # End index of the interval
        interval_df = df_pat.loc[start_idx : end_idx]

        if not interval_df.empty:
            # Get the label for this interval (use the first point's label)
            interval_label = interval_df['true_label'].iloc[0]
            start_x = interval_df['x'].iloc[0]
            end_x = interval_df['x'].iloc[-1]

            if interval_label == 1: # If it's a positive period
                fig.add_vrect(
                    x0=start_x, x1=end_x,
                    fillcolor="rgba(0, 255, 0, 0.15)", # Light green background
                    layer="below", line_width=0,
                    name="True Positive Period" if 'True Positive Period' not in [t.name for t in fig.data] else None, # Show legend only once
                    showlegend=('True Positive Period' not in [t.name for t in fig.data])
                )

    # 2. Raw Risk Data Points
    fig.add_trace(
        go.Scatter(
            x=df_pat['x'], y=df_pat['risk'],
            name="Risk Score", mode='lines+markers', # Show both lines and markers
            line=dict(color="rgba(0, 0, 150, 0.7)", width=1.5), # Darker blue line
            marker=dict(color='rgba(0, 0, 200, 0.5)', size=5, opacity=0.6), # Slightly transparent blue markers
            hovertemplate = f"<b>{xlabel}:</b> %{{x:.2f}}<br><b>Risk:</b> %{{y:.3f}}<extra></extra>"
            )
    )

    # --- Layout and Final Touches ---
    fig.update_layout(
        title=f"<b>Patient {ids__uid[:8]} Risk Trajectory</b>",
        xaxis_title=f"<b>{xlabel}</b>",
        yaxis_title="<b>Predicted Risk (Positive Class)</b>",
        yaxis_range=[-0.05, 1.05], # Give slight padding
        template="plotly_white",
        width=width, height=height,
        legend_title_text='Legend',
        hovermode="x unified" # Show hover info for all traces at a given x
    )
    if layout_d and isinstance(layout_d, dict): # Apply custom layout updates
        fig.update_layout(**layout_d)
    if xlim_l and len(xlim_l) == 2:
         fig.update_xaxes(range=xlim_l)

    return fig


# --- Matplotlib Plot to Base64 ---
def plt_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    if fig is None:
        return None
    try:
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100) # Adjust dpi as needed
        img.seek(0)
        base64_str = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close(fig) # Close the figure to free memory
        return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        print(f"ERROR converting matplotlib figure to base64: {e}")
        plt.close(fig) # Ensure figure is closed even on error
        return None

# --- Flask App Definition ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flash messages and session

# Define cache directory within the app's instance path
CACHE_DIR = os.path.join(app.instance_path, 'analysis_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Core Analysis Function (Adapted from original main) ---
def run_analysis(results_dir, model_prefix, use_cache=True):
    """
    Runs the full analysis pipeline based on user inputs.
    Returns a dictionary containing results (HTML tables, plot data/HTML)
    and a list of errors/warnings.
    """
    analysis_results = {
        "summary_table_html": None,
        "detailed_df": None, # Keep full df for potential download later
        "roc_plot": None, # Will store base64 string for matplotlib plot
        "pr_plot": None, # Will store base64 string for matplotlib plot
        "violin_feature_plot_html": None, # Plotly HTML
        "violin_demo_plot_html": None, # Plotly HTML
        "density_pna_plot_html": None, # Plotly HTML
        "density_tte_plot_html": None, # Plotly HTML
        "example_patient_plot_html": None, # Plotly HTML
        "alarm_pna_plot": None, # Matplotlib base64
        "alarm_tte_plot": None, # Matplotlib base64
        "config_summary": {}, # Store summary of configurations found
        "patient_ids_positive": [], # List of positive patient IDs for selection
        "selected_config_key": None, # Store the key used for patient/density plots
        "selected_patient_id": None, # Store the patient ID plotted
    }
    errors = []
    warnings_list = [] # Use separate list for non-critical warnings

    print(f"Starting analysis for: Directory='{results_dir}', Prefix='{model_prefix}'")
    warnings_list.append(f"Analysis started for: Directory='{results_dir}', Prefix='{model_prefix}'")

    # --- Configuration ---
    # Plotting settings (can be customized further)
    model_colors = px.colors.qualitative.Plotly
    font_size = 10 # Smaller font for web display
    layout_d = {"plot_bgcolor": "#fff", "font": {"size": font_size}}
    yaxes_d = {"gridcolor": "#ddd"}
    fontdict = {'fontsize': font_size}
    width = 600 # Smaller default width for web layout
    height = 450 # Smaller default height
    save_opt_px = {'scale': 1.5} # Scale for Plotly image export (if needed, less relevant for HTML)

    # Rename dictionary (keep as defined in original script)
    rename_d = {
         "data/firstclinical.csv": "Cohort", "data/firstclinical_vlbw.csv": "VLBW",
         "bw.sex": "BW & Sex", "bw.sex.pna": "BW, Sex & PNA", "pna": "PNA", "": "None",
         0: "HRC", 2: "All", "pop": "Population", "features": "Features", "demos": "Demographics",
         "fpr": "FPR", "spec": "Spec.", "p": "Prec.", "r": "Recall", "f1score": "F1-score",
         "auroc": "AUROC", "bAcc": "Balanced Acc.", "sen": "Sensitivity", "tnr": "Specificity"
         # Add more mappings if needed based on actual config values
    }
    # Add reverse mapping for column names if needed
    rename_metrics_rev = {v: k for k, v in rename_d.items() if k in ["fpr", "spec", "p", "r", "f1score", "auroc", "bAcc", "sen", "tnr"]}


    # --- Find and Load Data ---
    if not os.path.isdir(results_dir):
        errors.append(f"Error: Results directory not found or is not a directory: {results_dir}")
        return analysis_results, errors, warnings_list

    search_pattern = os.path.join(results_dir, f"{model_prefix}*.pklz")
    all_files = sorted(glob(search_pattern)) # Sort for consistency
    warnings_list.append(f"Found {len(all_files)} model result files matching pattern.")
    if not all_files:
        errors.append("No result files found matching the pattern. Cannot proceed.")
        return analysis_results, errors, warnings_list

    # --- Aggregate Results (with Caching) ---
    # Define cache file path based on inputs
    # Sanitize inputs for filename (replace path separators, etc.)
    sanitized_dir = results_dir.replace(os.sep, '_').replace(':','').replace('/','_').replace('\\','_')
    sanitized_prefix = model_prefix.replace('*','_star_').replace('?','_q_')
    cache_filename = f"agg_results_{sanitized_dir}_{sanitized_prefix}.pklz"
    agg_cache_file = os.path.join(CACHE_DIR, cache_filename)

    agg_res = None
    agg_errors = []

    if use_cache and os.path.exists(agg_cache_file):
        cache_age = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(os.path.getmtime(agg_cache_file))).total_seconds() / 3600
        warnings_list.append(f"Found cache file: {agg_cache_file} (Age: {cache_age:.1f} hours)")
        print(f"Attempting to load aggregated results from cache: {agg_cache_file}")
        try:
            agg_res = read_pklz(agg_cache_file)
            # Basic validation of cached structure
            if not isinstance(agg_res, dict) or not all(k in agg_res for k in ["aurocs_results", "prs_results", "aurocs_scores", "Cfgs", "patwise_predictions"]):
                 warnings_list.append("Cache file has unexpected structure. Re-aggregating...")
                 agg_res = None # Force re-aggregation
                 os.remove(agg_cache_file) # Remove invalid cache
            else:
                 warnings_list.append("Successfully loaded aggregated results from cache.")
        except Exception as e:
            warnings_list.append(f"Failed to load or validate cache file ({e}). Re-aggregating...")
            agg_res = None # Force re-aggregation
            try: os.remove(agg_cache_file)
            except OSError: pass

    if agg_res is None:
        warnings_list.append("Cache not used or invalid. Loading raw data files...")
        print("Loading data from files...")
        loaded_data = []
        files_to_load = all_files[:50] # Limit number of files initially for web app responsiveness? Or load all? Let's load all for now.
        # files_to_load = all_files
        if len(all_files) > 50:
             warnings_list.append(f"WARNING: Loading all {len(all_files)} files. This might take time...")

        for f in files_to_load: # Use the potentially limited list
            try:
                loaded_data.append(read_pklz(f))
            except Exception as e:
                errors.append(f"Error loading file {os.path.basename(f)}: {e}. Skipping file.")
                loaded_data.append(None) # Add placeholder

        valid_data = [d for d in loaded_data if d is not None]
        warnings_list.append(f"Successfully loaded {len(valid_data)} out of {len(files_to_load)} files.")

        if not valid_data:
             errors.append("No valid data could be loaded. Cannot proceed.")
             return analysis_results, errors, warnings_list

        print("Aggregating results (this may take time)...")
        warnings_list.append("Aggregating results... This may take a while.")
        # Define the name of the 'negative' or 'healthy' class column used in results
        class0_name_in_results = "not_target__healthy" # Adjust if necessary based on data generation
        n_cores = max(1, os.cpu_count() // 2) # Use half the cores
        agg_res, agg_errors = aggregate_scores_run(valid_data, theset="val", th=0.5, n_jobs=n_cores, class0_name=class0_name_in_results)

        # Add aggregation errors/warnings to the main list
        if agg_errors:
            warnings_list.extend([f"Aggregation Warning/Error: {e}" for e in agg_errors])

        # Save to cache if aggregation was successful
        if agg_res and agg_res.get("Cfgs"): # Check if results look valid
            try:
                print(f"Saving aggregated results to cache: {agg_cache_file}")
                write_pklz(agg_cache_file, agg_res)
                warnings_list.append("Saved aggregated results to cache.")
            except Exception as e_cache:
                warnings_list.append(f"Warning: Failed to save results to cache file {agg_cache_file}: {e_cache}")
        else:
            errors.append("Aggregation failed or yielded no results.")
            # No need to return yet, maybe some partial results exist, let later steps handle it

    # --- Check Aggregated Results ---
    if not agg_res or not agg_res.get("Cfgs"):
        errors.append("Error: Aggregation failed or yielded no configurations. Cannot proceed with analysis.")
        return analysis_results, errors, warnings_list

    warnings_list.append(f"Aggregated results found for {len(agg_res['Cfgs'])} model configurations.")
    analysis_results["config_summary"] = {k: v['model_name'] for k, v in agg_res['Cfgs'].items()}


    # --- Analysis 1: Numerical Summaries ---
    print("Analysis 1: Numerical Summaries and Tables")
    all_scores_list = []
    config_keys = list(agg_res.get("Cfgs", {}).keys()) # Get stable list of keys

    for k in config_keys:
        scores_for_k = agg_res.get("aurocs_scores", {}).get(k, [])
        cfg_for_k = agg_res.get("Cfgs", {}).get(k, {})

        if not scores_for_k:
             warnings_list.append(f"No scores found for config: {k[:50]}...") # Show partial key
             continue

        try:
             # Ensure scores_for_k is a list of dicts suitable for DataFrame
             if all(isinstance(item, dict) for item in scores_for_k):
                 df_k = pd.DataFrame.from_records(scores_for_k)
                 # Add config details to each row
                 for cfg_key, cfg_val in cfg_for_k.items():
                     df_k[cfg_key] = cfg_val
                 all_scores_list.append(df_k)
             else:
                  warnings_list.append(f"Scores data for config {k[:50]}... is not a list of dictionaries. Skipping.")
        except Exception as e_df:
             errors.append(f"Error creating DataFrame for scores (config {k[:50]}...): {e_df}")
             continue # Skip this config if DataFrame creation fails


    if not all_scores_list:
        errors.append("No scores dataframes could be created. Cannot generate summary tables.")
        # Don't return yet, other analyses might still work
    else:
        try:
            dfout = pd.concat(all_scores_list, ignore_index=True)
            analysis_results["detailed_df"] = dfout.copy() # Store the raw detailed df

            # Apply renaming for readability
            dfout_renamed = dfout.copy()
            # Rename columns first
            dfout_renamed = dfout_renamed.rename(columns=rename_d)
            # Rename specific column values if applicable
            for col in ['Population', 'Features', 'Demographics', 'model_name', 'rback']:
                if col in dfout_renamed.columns:
                     if col == 'model_name':
                          dfout_renamed[col] = dfout_renamed[col].apply(rename_model)
                          dfout_renamed.rename(columns={'model_name': 'Model Name'}, inplace=True) # Rename the column itself
                     else:
                          # Apply renaming based on rename_d keys matching values in the column
                          value_map = {k: v for k, v in rename_d.items() if k in dfout_renamed[col].unique()}
                          if value_map:
                               dfout_renamed[col] = dfout_renamed[col].replace(value_map)

            # --- Create Summary Table (Median and IQR) ---
            # Use renamed columns if they exist, otherwise original
            grouping_cols_options = ['Population', 'Features', 'Demographics', 'rback', 'Model Name']
            grouping_cols = [col for col in grouping_cols_options if col in dfout_renamed.columns]

            metrics_options = ['AUROC', 'F1-score', 'Balanced Acc.', 'Prec.', 'Recall', 'Sensitivity', 'Specificity', 'FPR'] # Add FPR
            metrics_to_summarize = [m for m in metrics_options if m in dfout_renamed.columns]


            if grouping_cols and metrics_to_summarize:
                 # Calculate median and IQR grouped by configuration
                 # Handle potential non-numeric data gracefully in aggregation
                 numeric_cols = dfout_renamed[metrics_to_summarize].apply(pd.to_numeric, errors='coerce').columns
                 if not numeric_cols.empty:
                      summary_table = dfout_renamed.groupby(grouping_cols)[numeric_cols].agg(
                          ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]
                      )
                      summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values] # Flatten multi-index
                      summary_table = summary_table.rename(columns=lambda x: x.replace('<lambda_0>', 'iqr')) # Nicer IQR name
                      summary_table = summary_table.round(3)

                      # Sort by a primary metric (e.g., F1 median), handle missing columns
                      sort_col = 'F1-score_median' if 'F1-score_median' in summary_table.columns else (numeric_cols[0]+'_median' if len(numeric_cols)>0 else None)
                      if sort_col and sort_col in summary_table.columns:
                          summary_table.sort_values(by=sort_col, ascending=False, inplace=True)
                      else:
                          warnings_list.append(f"Could not sort summary table (sort column '{sort_col}' not found).")


                      # Convert to HTML for display
                      analysis_results["summary_table_html"] = summary_table.reset_index().to_html(classes='table table-striped table-hover table-sm', index=False, border=0, na_rep='N/A')
                      warnings_list.append("Generated summary table.")
                 else:
                      warnings_list.append("No numeric metric columns found for summary table.")

            else:
                 warnings_list.append("Could not perform summary grouping (missing grouping columns or metrics).")

        except Exception as e_summary:
            errors.append(f"Error generating numerical summary: {e_summary}")
            traceback.print_exc()


    # --- Analysis 2: ROC/PR Curves (Matplotlib) ---
    print("Analysis 2: ROC and PR Curves")
    try:
        aurocs_data = agg_res.get("aurocs_results", {})
        prs_data = agg_res.get("prs_results", {})
        configs = agg_res.get("Cfgs", {})

        # Select models/configs to plot (use all available)
        configs_to_plot = list(configs.keys())

        # Plot ROC Curves
        fig_roc, ax_roc = plt.subplots(figsize=(7, 7)) # Slightly smaller fig size
        plot_roc_data = []
        plot_labels = []
        plot_colors = []
        assigned_colors = {}
        color_idx = 0

        # Limit number of curves plotted directly for clarity? Maybe top N?
        # For now, plot all available.
        max_curves_to_plot = 15 # Limit to avoid overly cluttered plots
        if len(configs_to_plot) > max_curves_to_plot:
            warnings_list.append(f"Plotting only top {max_curves_to_plot} ROC/PR curves (by avg AUC) due to high number of configs ({len(configs_to_plot)}).")
            # Sort configs by mean AUC (need to calculate first)
            auc_means = {}
            for k in configs_to_plot:
                 roc_list = aurocs_data.get(k)
                 if roc_list:
                      auc_means[k] = np.nanmean([r[3] for r in roc_list if len(r)>3 and r[3] is not None])
            # Sort keys by descending AUC, handle NaNs
            sorted_keys = sorted(configs_to_plot, key=lambda k: auc_means.get(k, -1), reverse=True)
            configs_to_plot = sorted_keys[:max_curves_to_plot]


        for i, k in enumerate(configs_to_plot):
            roc_list = aurocs_data.get(k)
            if roc_list: # Check if data exists
                model_name = configs.get(k, {}).get('model_name', f'Unknown_{i}')
                nice_name = rename_model(model_name)
                mean_auc = np.nanmean([r[3] for r in roc_list if len(r)>3 and r[3] is not None])
                if not np.isnan(mean_auc):
                     label_suffix = f"(AUC={mean_auc:.3f})"
                else:
                     label_suffix = "(AUC=N/A)"

                plot_roc_data.append(roc_list)
                plot_labels.append(f"{nice_name} {label_suffix}")

                # Assign colors based on model base name or type
                base_model_name = nice_name.split(' ')[0].split('(')[0] # Simple grouping by first word before bracket
                if base_model_name not in assigned_colors:
                    assigned_colors[base_model_name] = model_colors[color_idx % len(model_colors)]
                    color_idx += 1
                plot_colors.append(assigned_colors[base_model_name])

        if plot_roc_data:
            plot_roc(ax_roc, plot_roc_data, f"ROC Curves ({model_prefix})", opts=plot_labels, colors=plot_colors, lw=1.5, alpha=0.8, fontdict=fontdict)
            better_lookin(ax_roc, legend_bbox=(1.05, 1), grid=True) # Legend outside
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
            analysis_results["roc_plot"] = plt_to_base64(fig_roc) # Convert to base64
            warnings_list.append("Generated ROC plot.")
        else:
            warnings_list.append("No ROC data available to plot.")
            plt.close(fig_roc)

        # Plot PR Curves
        fig_pr, ax_pr = plt.subplots(figsize=(7, 7))
        plot_pr_data = []
        plot_labels_pr = []
        plot_colors_pr = [] # Use the same color mapping

        for i, k in enumerate(configs_to_plot): # Use the same limited/sorted list
            pr_list = prs_data.get(k)
            if pr_list: # Check if data exists
                model_name = configs.get(k, {}).get('model_name', f'Unknown_{i}')
                nice_name = rename_model(model_name)
                mean_auprc = np.nanmean([p[3] for p in pr_list if len(p)>3 and p[3] is not None]) # 4th element is AUPRC
                if not np.isnan(mean_auprc):
                     label_suffix = f"(AUPRC={mean_auprc:.3f})"
                else:
                     label_suffix = "(AUPRC=N/A)"

                plot_pr_data.append(pr_list)
                plot_labels_pr.append(f"{nice_name} {label_suffix}")

                # Reuse colors assigned in ROC plot
                base_model_name = nice_name.split(' ')[0].split('(')[0]
                plot_colors_pr.append(assigned_colors.get(base_model_name, model_colors[i % len(model_colors)])) # Fallback color

        if plot_pr_data:
            # Use the same plot_roc function with pr=True
            plot_roc(ax_pr, plot_pr_data, f"Precision-Recall Curves ({model_prefix})", opts=plot_labels_pr, colors=plot_colors_pr, lw=1.5, alpha=0.8, fontdict=fontdict, pr=True) # Set pr=True
            better_lookin(ax_pr, legend_bbox=(1.05, 1), grid=True)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
            analysis_results["pr_plot"] = plt_to_base64(fig_pr) # Convert to base64
            warnings_list.append("Generated PR plot.")
        else:
            warnings_list.append("No PR data available to plot.")
            plt.close(fig_pr)

    except Exception as e_rocpr:
         errors.append(f"Error generating ROC/PR plots: {e_rocpr}")
         traceback.print_exc()
         # Ensure figures are closed if they exist
         if 'fig_roc' in locals() and plt.fignum_exists(fig_roc.number): plt.close(fig_roc)
         if 'fig_pr' in locals() and plt.fignum_exists(fig_pr.number): plt.close(fig_pr)


    # --- Analysis 3: Feature/Demographic Comparison (Plotly Violin Plots) ---
    print("Analysis 3: Feature/Demographic Comparisons")
    try:
        # Use the 'dfout_renamed' DataFrame created earlier if it exists
        if 'dfout_renamed' in locals() and isinstance(dfout_renamed, pd.DataFrame):
            # Violin Plot: AUROC vs Features
            if 'Features' in dfout_renamed.columns and 'AUROC' in dfout_renamed.columns:
                # Check if 'Features' column has more than 1 unique value for meaningful plot
                if dfout_renamed['Features'].nunique() > 1:
                    try:
                        fig_feat = px.violin(dfout_renamed, y="AUROC", x="Features",
                                             points="all", box=True,
                                             color="Features", # Color by feature set
                                             title=f"AUROC vs. Features Used ({model_prefix})",
                                             template="plotly_white", width=width, height=height,
                                             labels={"AUROC": "<b>AUROC</b>", "Features": "<b>Features Set</b>"})
                        fig_feat.update_layout(**layout_d)
                        fig_feat.update_yaxes(**yaxes_d, range=[-0.05, 1.05], tickprefix="<b>", ticksuffix ="</b>")
                        fig_feat.update_xaxes(tickprefix="<b>", ticksuffix ="</b>")
                        analysis_results["violin_feature_plot_html"] = fig_feat.to_html(full_html=False, include_plotlyjs='cdn')
                        warnings_list.append("Generated AUROC vs Features violin plot.")
                    except Exception as e_fig:
                         warnings_list.append(f"Could not generate AUROC vs Features plot: {e_fig}")
                else:
                    warnings_list.append("Skipping AUROC vs Features plot (only one unique feature set found).")
            else:
                 warnings_list.append("Skipping AUROC vs Features plot (missing columns 'Features' or 'AUROC').")

            # Violin Plot: AUROC vs Demographics
            if 'Demographics' in dfout_renamed.columns and 'AUROC' in dfout_renamed.columns:
                 if dfout_renamed['Demographics'].nunique() > 1:
                    try:
                        fig_demo = px.violin(dfout_renamed, y="AUROC", x="Demographics",
                                            points="all", box=True,
                                            color="Demographics",
                                            title=f"AUROC vs. Demographics Used ({model_prefix})",
                                            template="plotly_white", width=width, height=height,
                                            labels={"AUROC": "<b>AUROC</b>", "Demographics": "<b>Demographic Features</b>"})
                        fig_demo.update_layout(**layout_d)
                        fig_demo.update_yaxes(**yaxes_d, range=[-0.05, 1.05], tickprefix="<b>", ticksuffix ="</b>")
                        fig_demo.update_xaxes(tickprefix="<b>", ticksuffix ="</b>")
                        analysis_results["violin_demo_plot_html"] = fig_demo.to_html(full_html=False, include_plotlyjs='cdn')
                        warnings_list.append("Generated AUROC vs Demographics violin plot.")
                    except Exception as e_fig:
                         warnings_list.append(f"Could not generate AUROC vs Demographics plot: {e_fig}")
                 else:
                      warnings_list.append("Skipping AUROC vs Demographics plot (only one unique demographic set found).")
            else:
                 warnings_list.append("Skipping AUROC vs Demographics plot (missing columns 'Demographics' or 'AUROC').")
        else:
             warnings_list.append("Skipping violin plots (detailed scores DataFrame not available).")

    except Exception as e_violin:
        errors.append(f"Error during violin plot generation: {e_violin}")
        traceback.print_exc()


    # --- Analysis 4: Population/Patient Risk Visualization (Plotly) ---
    print("Analysis 4: Population/Patient Risk Visualizations")
    try:
        patwise_preds = agg_res.get("patwise_predictions", {})
        pos_pats_map = agg_res.get("pos_pat", {}) # Map from config key to list of positive patient IDs
        configs = agg_res.get("Cfgs", {})

        # Select a key configuration to plot (e.g., best F1 or AUC from summary?)
        # For simplicity, pick the first config key that has patient data and reasonable performance
        plot_key = None
        best_metric = -1

        # Use summary table if available to find a good key
        if analysis_results.get("summary_table_html"):
             try:
                  summary_df_temp = pd.read_html(io.StringIO(analysis_results["summary_table_html"]))[0]
                  # Find the row with the best F1-score median (handle potential errors)
                  if 'F1-score_median' in summary_df_temp.columns:
                      best_row = summary_df_temp.loc[summary_df_temp['F1-score_median'].astype(float).idxmax()]
                      # Reconstruct the config key from the grouping columns
                      # This is tricky, depends on how keys were made. Assume key format is consistent.
                      # Fallback: just find a key with patient data
                  else:
                       warnings_list.append("F1-score_median not in summary table for selecting best config.")
             except Exception as e_readhtml:
                  warnings_list.append(f"Could not read summary table HTML to find best config: {e_readhtml}")

        # Fallback or default: find the first key with patient data
        if plot_key is None:
             for k in config_keys:
                 if k in patwise_preds and patwise_preds[k]: # Check if key exists and has patient data
                     # Check if this key has positive patients (more interesting plots)
                     if k in pos_pats_map and pos_pats_map[k]:
                          plot_key = k
                          warnings_list.append(f"Selected config key for patient plots (first with positive patients): {k[:50]}...")
                          break
             # If no key with positive patients, take the very first with any patient data
             if plot_key is None:
                  for k in config_keys:
                      if k in patwise_preds and patwise_preds[k]:
                          plot_key = k
                          warnings_list.append(f"Selected config key for patient plots (first with any patient data): {k[:50]}...")
                          break


        if plot_key and plot_key in configs:
            analysis_results["selected_config_key"] = plot_key
            config_short_name = rename_model(configs[plot_key].get('model_name', 'Unknown')).replace(" ", "_").replace("=","").replace("(","").replace(")","").replace(",","_").replace("->","_") # Sanitize name
            warnings_list.append(f"Generating population/patient plots for config: {plot_key[:80]}...")
            pat_data_dict = patwise_preds.get(plot_key, {})
            pos_pats_for_key = pos_pats_map.get(plot_key, [])
            analysis_results["patient_ids_positive"] = pos_pats_for_key # Store positive patient IDs

            # Aggregate patient data across folds if needed (currently uses first fold)
            all_pat_x_pna = []
            all_pat_y_pna = []
            all_pat_tte = [] # List of tuples (tte, risk) for positive patients

            # Use the first fold's data for each patient for simplicity in population plots
            for pat_id, data_list in pat_data_dict.items():
                if data_list:
                    d = data_list[0] # Use first fold's data
                    # PNA vs Risk
                    time_axis_pna = d.get('pna_days')
                    risk_scores = d.get('ypred')[:, 1] if d.get('ypred') is not None and d.get('ypred').shape[1] > 1 else None

                    if time_axis_pna is not None and risk_scores is not None:
                        valid_idx = ~np.isnan(time_axis_pna) & ~np.isnan(risk_scores)
                        all_pat_x_pna.extend(time_axis_pna[valid_idx])
                        all_pat_y_pna.extend(risk_scores[valid_idx])

                    # TTE vs Risk (for positive patients)
                    if pat_id in pos_pats_for_key:
                        tte_raw = d.get('time_to')
                        if tte_raw is not None and tte_raw.ndim > 0:
                            # Combine multiple event times if necessary
                            tte_combined = combine_tt(tte_raw.T) if tte_raw.ndim > 1 else tte_raw.flatten()
                            risk_scores_tte = d.get('ypred')[:, 1] if d.get('ypred') is not None and d.get('ypred').shape[1] > 1 else None

                            if risk_scores_tte is not None and len(tte_combined) == len(risk_scores_tte):
                                valid_tte_idx = ~np.isnan(tte_combined) & ~np.isnan(risk_scores_tte)
                                all_pat_tte.extend(list(zip(tte_combined[valid_tte_idx], risk_scores_tte[valid_tte_idx])))


            # --- Population Density Plot (Risk vs PNA Days) ---
            if all_pat_x_pna and all_pat_y_pna:
                fig_density_pna = density_plot(np.array(all_pat_x_pna), np.array(all_pat_y_pna),
                                            title=f"Population Risk Density vs PNA ({config_short_name})",
                                            xlabel="Postnatal Age (Days)", ylabel="Risk Score", font_size=font_size,
                                            nh=10) # Use 10-day bins for PNA maybe? Or smaller like 5?
                analysis_results["density_pna_plot_html"] = fig_density_pna.to_html(full_html=False, include_plotlyjs='cdn')
                warnings_list.append("Generated Risk vs PNA density plot.")
            else:
                 warnings_list.append("No data available for Risk vs PNA density plot.")

            # --- Population Density Plot (Risk vs Time To Event for Positive Patients) ---
            if all_pat_tte:
                all_pat_tte_arr = np.array(all_pat_tte)
                tte_values = all_pat_tte_arr[:, 0]
                risk_values_tte = all_pat_tte_arr[:, 1]
                fig_density_tte = density_plot(tte_values, risk_values_tte,
                                            title=f"Positive Patient Risk Density vs TTE ({config_short_name})",
                                            xlabel="Time to Event (Hours)", ylabel="Risk Score", font_size=font_size,
                                            xlim_l=(-168, 48), # Focus on -7 days to +2 days
                                            nh=6) # 6-hour bins for TTE
                analysis_results["density_tte_plot_html"] = fig_density_tte.to_html(full_html=False, include_plotlyjs='cdn')
                warnings_list.append("Generated Risk vs TTE density plot.")
            else:
                warnings_list.append("No positive patient data with TTE available for density plot.")

            # --- Example Patient Plot ---
            # Select one positive patient to plot (first one in list)
            example_pat_id = pos_pats_for_key[0] if pos_pats_for_key else None
            # If no positive patients, maybe plot the first patient available?
            if not example_pat_id and pat_data_dict:
                 example_pat_id = list(pat_data_dict.keys())[0]
                 warnings_list.append("No positive patients found for example plot; plotting first available patient.")

            if example_pat_id and example_pat_id in pat_data_dict:
                analysis_results["selected_patient_id"] = example_pat_id
                fig_example = plot_example_pat(pat_data_dict[example_pat_id], example_pat_id,
                                               xname="pna_days", # Plot against PNA days by default
                                               xlim_l=None, # No limits
                                               layout_d=layout_d, width=width, height=height)
                analysis_results["example_patient_plot_html"] = fig_example.to_html(full_html=False, include_plotlyjs='cdn')
                warnings_list.append(f"Generated example patient plot for {example_pat_id[:8]}.")
            else:
                 warnings_list.append("Could not find suitable patient to generate example plot.")

        else:
            warnings_list.append("Skipping population/patient plots (no suitable configuration or patient data found).")

    except Exception as e_pop_pat:
        errors.append(f"Error generating population/patient plots: {e_pop_pat}")
        traceback.print_exc()

    # --- Analysis 5: Alarm Rate Analysis (Matplotlib) ---
    print("Analysis 5: Alarm Rate Analysis")
    try:
        # Use the same selected config key as for patient plots if available
        alarm_plot_key = analysis_results.get("selected_config_key")
        target_model_name_alarm = None
        if alarm_plot_key and alarm_plot_key in configs:
             target_model_name_alarm = configs[alarm_plot_key].get('model_name', 'Unknown')
             target_model_nice_name = rename_model(target_model_name_alarm)
             warnings_list.append(f"Generating alarm rate plots for model: {target_model_nice_name} (based on selected config)")
        else:
             # Try to find a default model like GMM if no key was selected
             target_model_name_alarm = "gmm1.diag" # Default target
             target_model_nice_name = rename_model(target_model_name_alarm)
             alarm_plot_key = None
             for k, cfg in configs.items():
                 if cfg.get('model_name') == target_model_name_alarm:
                     alarm_plot_key = k
                     warnings_list.append(f"Generating alarm rate plots for default model: {target_model_nice_name}")
                     break

        if alarm_plot_key and alarm_plot_key in patwise_preds:
            pat_data_dict_alarm = patwise_preds.get(alarm_plot_key, {})
            pos_pats_alarm = pos_pats_map.get(alarm_plot_key, [])
            # all_pats_alarm = list(pat_data_dict_alarm.keys()) # All patients for this model

            alarm_data_list = []
            for pat_id, data_list in pat_data_dict_alarm.items():
                if data_list:
                    d = data_list[0] # Use first fold
                    is_pos = pat_id in pos_pats_alarm
                    status = "pos" if is_pos else "neg"

                    time_pna = d.get('pna_days')
                    tte_raw = d.get('time_to')
                    time_tte = combine_tt(tte_raw.T) if tte_raw is not None and tte_raw.ndim > 1 else (tte_raw.flatten() if tte_raw is not None else None)

                    pred_bin_scores = d.get('ypred')[:, 1] if d.get('ypred') is not None and d.get('ypred').shape[1] > 1 else None
                    true_bin = d.get('ytrue')[:, 1] if d.get('ytrue') is not None and d.get('ytrue').shape[1] > 1 else None

                    if pred_bin_scores is not None and true_bin is not None and time_pna is not None:
                        # Ensure lengths match after potential NaNs
                        min_len = len(time_pna)
                        if len(pred_bin_scores) != min_len or len(true_bin) != min_len or (time_tte is not None and len(time_tte) != min_len):
                             warnings_list.append(f"Alarm Data: Length mismatch for patient {pat_id[:8]}. Skipping.")
                             continue

                        # Apply threshold for alarms (e.g., 0.5)
                        alarms = pred_bin_scores >= 0.5
                        tp = alarms & (true_bin == 1)
                        fp = alarms & (true_bin == 0)

                        for i in range(min_len):
                            # Check for NaN time values before appending
                            if np.isnan(time_pna[i]): continue

                            alarm_data_list.append({
                                "patid": pat_id, "status": status,
                                "pn_age_days": time_pna[i],
                                # Convert tte hours to days if available
                                "tte_days": time_tte[i]/24 if time_tte is not None and i < len(time_tte) and not np.isnan(time_tte[i]) else np.nan,
                                "alarm": alarms[i], "tp": tp[i], "fp": fp[i], "true": true_bin[i]
                            })


            if alarm_data_list:
                df_alarm = pd.DataFrame(alarm_data_list)
                df_alarm.dropna(subset=['pn_age_days'], inplace=True) # Need PNA age at least

                # Plot Alarm Rate vs PNA Days (Matplotlib)
                if not df_alarm.empty:
                    try:
                        fig_alarm_pna, axes_alarm_pna = plt.subplots(2, 1, figsize=(9, 6), sharex=True) # Smaller figure
                        ax_rate, ax_count = axes_alarm_pna

                        bins = np.arange(0, df_alarm['pn_age_days'].max() + 7, 7) # 7-day bins
                        # Ensure labels match bins correctly
                        bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
                        df_alarm['pna_binned'] = pd.cut(df_alarm['pn_age_days'], bins, right=False, labels=bin_labels) # Use labels

                        # Calculate rates per bin
                        grouped = df_alarm.groupby(['pna_binned', 'status'], observed=False) # Use observed=False with categorical
                        # Calculate mean alarm/fp/tp rate across all points in bin
                        binned_stats = grouped[['alarm', 'fp', 'tp']].mean().reset_index()
                        # Calculate number of unique patients contributing to each bin
                        patient_counts = df_alarm.groupby(['pna_binned', 'status'], observed=False)['patid'].nunique().reset_index()

                        # Merge stats and counts for plotting
                        merged_data = pd.merge(binned_stats, patient_counts, on=['pna_binned', 'status'], how='left')

                        # Get the actual bin centers or start points for plotting x-axis
                        # Using the labels directly as categories might be better for plot clarity
                        x_plot_pna = merged_data['pna_binned'].astype(str).unique() # Use bin labels as x-categories

                        for status, color in [("pos", "darkred"), ("neg", "darkblue")]:
                             subset_data = merged_data[merged_data['status'] == status]
                             if not subset_data.empty:
                                 # Plot FP rate for neg, TP rate for pos (or total alarms?)
                                 # Let's plot FP for neg, and TP for pos (more informative than total alarms for pos)
                                 rate_to_plot = subset_data['fp'] if status == 'neg' else subset_data['tp']
                                 lbl = f"False Alarm Rate (Neg Pts)" if status == 'neg' else f"True Alarm (TP) Rate (Pos Pts)"
                                 # Map bin labels back to numerical positions if needed, or plot categorical
                                 ax_rate.plot(subset_data['pna_binned'].astype(str), rate_to_plot, marker='o', markersize=4, linestyle='-', color=color, label=lbl)

                                 # Plot patient counts on second axis
                                 ax_count.plot(subset_data['pna_binned'].astype(str), subset_data['patid'], marker='.', markersize=5, linestyle='--', color=color, label=f"# Patients ({status})")

                        ax_rate.set_ylabel("Rate", fontsize=font_size-1)
                        ax_rate.set_ylim(-0.05, 1.05)
                        ax_rate.set_title(f"Alarm Rates vs PNA ({target_model_nice_name})", fontsize=font_size)
                        ax_rate.legend(fontsize=font_size-2)
                        ax_rate.grid(True, linestyle='--', alpha=0.6)
                        ax_rate.tick_params(axis='x', rotation=45, labelsize=font_size-2) # Rotate labels if needed

                        ax_count.set_ylabel("Patient Count", fontsize=font_size-1)
                        ax_count.set_xlabel("Postnatal Age (days, binned)", fontsize=font_size-1)
                        ax_count.legend(fontsize=font_size-2)
                        ax_count.grid(True, linestyle='--', alpha=0.6)
                        ax_count.tick_params(axis='x', rotation=45, labelsize=font_size-2)
                        ax_count.set_yscale('log') # Use log scale for counts if range is large? Or linear? Linear default.

                        plt.tight_layout()
                        analysis_results["alarm_pna_plot"] = plt_to_base64(fig_alarm_pna)
                        warnings_list.append("Generated Alarm Rate vs PNA plot.")

                    except Exception as e:
                        errors.append(f"Error generating Alarm Rate vs PNA plot: {e}")
                        traceback.print_exc()
                        if 'fig_alarm_pna' in locals() and plt.fignum_exists(fig_alarm_pna.number): plt.close(fig_alarm_pna)

                else:
                     warnings_list.append("No data available for Alarm Rate vs PNA plot after processing.")


                # Plot Alarm Rate vs Time To Event (Pos Patients Only, Matplotlib)
                df_alarm_pos = df_alarm[(df_alarm['status'] == 'pos') & (df_alarm['tte_days'].notna())].copy() # Ensure TTE is not NaN
                if not df_alarm_pos.empty:
                     try:
                        fig_alarm_tte, axes_alarm_tte = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                        ax_rate_tte, ax_count_tte = axes_alarm_tte

                        tte_max_days = 10 # Max days before/after event
                        tte_min_days = -10 # Min days
                        tte_step_days = 1 # 1-day bins
                        bins_tte = np.arange(tte_min_days, tte_max_days + tte_step_days, tte_step_days)
                        # Ensure labels match bins correctly
                        bin_labels_tte = [f"{bins_tte[i]} to {bins_tte[i+1]}" for i in range(len(bins_tte)-1)]
                        df_alarm_pos['tte_binned'] = pd.cut(df_alarm_pos['tte_days'], bins_tte, right=False, labels=bin_labels_tte)

                        grouped_tte = df_alarm_pos.groupby('tte_binned', observed=False)
                        binned_stats_tte = grouped_tte[['alarm', 'fp', 'tp']].mean().reset_index()
                        patient_counts_tte = df_alarm_pos.groupby('tte_binned', observed=False)['patid'].nunique().reset_index()

                        # Merge stats and counts
                        merged_data_tte = pd.merge(binned_stats_tte, patient_counts_tte, on='tte_binned', how='left')
                        x_plot_tte = merged_data_tte['tte_binned'].astype(str).unique() # Use bin labels as x-categories


                        # Plot stacked bar for TP and FP rates for positive patients
                        ax_rate_tte.bar(merged_data_tte['tte_binned'].astype(str), merged_data_tte['fp'], width=0.8, color='orange', label='False Positives (FP)', alpha=0.8)
                        ax_rate_tte.bar(merged_data_tte['tte_binned'].astype(str), merged_data_tte['tp'], width=0.8, color='darkgreen', label='True Positives (TP)', alpha=0.8, bottom=merged_data_tte['fp']) # Stack TP on FP

                        ax_rate_tte.set_ylabel("Rate", fontsize=font_size-1)
                        ax_rate_tte.set_ylim(-0.05, 1.05)
                        ax_rate_tte.set_title(f"Alarm Composition vs TTE ({target_model_nice_name}, Pos Pts)", fontsize=font_size)
                        ax_rate_tte.legend(fontsize=font_size-2)
                        ax_rate_tte.grid(True, axis='y', linestyle='--', alpha=0.6) # Grid on y-axis only
                        ax_rate_tte.tick_params(axis='x', rotation=45, labelsize=font_size-2)

                        # Plot patient counts
                        ax_count_tte.plot(merged_data_tte['tte_binned'].astype(str), merged_data_tte['patid'], marker='.', markersize=5, linestyle='-', color='black', label="# Patients")
                        ax_count_tte.set_ylabel("Patient Count", fontsize=font_size-1)
                        ax_count_tte.set_xlabel("Time to Event (days, binned)", fontsize=font_size-1)
                        ax_count_tte.legend(fontsize=font_size-2)
                        ax_count_tte.grid(True, linestyle='--', alpha=0.6)
                        ax_count_tte.tick_params(axis='x', rotation=45, labelsize=font_size-2)
                        ax_count_tte.set_yscale('log') # Log scale often useful for patient counts near event

                        plt.tight_layout()
                        analysis_results["alarm_tte_plot"] = plt_to_base64(fig_alarm_tte)
                        warnings_list.append("Generated Alarm Rate vs TTE plot.")

                     except Exception as e:
                         errors.append(f"Error generating Alarm Rate vs TTE plot: {e}")
                         traceback.print_exc()
                         if 'fig_alarm_tte' in locals() and plt.fignum_exists(fig_alarm_tte.number): plt.close(fig_alarm_tte)
                else:
                    warnings_list.append("No positive patient data with TTE available for alarm rate plot.")

            else:
                 warnings_list.append("No data aggregated for alarm rate plots.")
        else:
             warnings_list.append(f"Skipping alarm rate plots (model '{target_model_nice_name}' not found or no patient data for selected config).")

    except Exception as e_alarm:
        errors.append(f"Error during alarm rate analysis: {e_alarm}")
        traceback.print_exc()


    print("--- Analysis Complete ---")
    warnings_list.append("Analysis processing finished.")

    # Final garbage collection
    gc.collect()

    return analysis_results, errors, warnings_list


# --- Flask HTML Templates (as strings) ---

# Template for the input form
INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <title>Model Analysis Runner</title>
    <style>
        body { padding-top: 40px; padding-bottom: 40px; background-color: #f5f5f5; }
        .form-container { max-width: 500px; padding: 15px; margin: auto; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }
        .form-container .form-floating:focus-within { z-index: 2; }
        .form-container input[type="text"] { margin-bottom: 10px; border-radius: 5px; }
        .btn-primary { background-color: #0d6efd; border-color: #0d6efd; } /* Standard Bootstrap Blue */
        .alert { margin-top: 20px; }
        .spinner-border { display: none; /* Hidden by default */ margin-left: 10px; }
        #loading-message { display: none; /* Hidden by default */ margin-top: 15px; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-container text-center">
            <h1 class="h3 mb-3 fw-normal">Run Model Analysis</h1>

            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                  </div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            <form method="post" action="{{ url_for('analyze') }}" id="analysis-form">
                <div class="form-floating">
                    <input type="text" class="form-control" id="results_dir" name="results_dir" placeholder="Results Directory" required value="{{ request.form['results_dir'] if request.form else '' }}">
                    <label for="results_dir">Results Directory Path</label>
                </div>
                <div class="form-floating">
                    <input type="text" class="form-control" id="model_prefix" name="model_prefix" placeholder="Model Prefix" required value="{{ request.form['model_prefix'] if request.form else 'nflow' }}">
                    <label for="model_prefix">Model File Prefix (e.g., nflow, gmm)</label>
                </div>
                 <div class="form-check text-start my-3">
                    <input class="form-check-input" type="checkbox" value="yes" id="use_cache" name="use_cache" checked>
                    <label class="form-check-label" for="use_cache">
                        Use Cache for Aggregated Results (Recommended)
                    </label>
                </div>

                <button class="w-100 btn btn-lg btn-primary" type="submit" id="submit-button">
                    Run Analysis
                    <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                </button>
                <p id="loading-message">Processing... This may take several minutes depending on the data size.</p>
            </form>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Show spinner and message on form submit
        const form = document.getElementById('analysis-form');
        const button = document.getElementById('submit-button');
        const spinner = button.querySelector('.spinner-border');
        const loadingMessage = document.getElementById('loading-message');

        form.addEventListener('submit', function() {
            button.disabled = true; // Disable button
            spinner.style.display = 'inline-block'; // Show spinner
            loadingMessage.style.display = 'block'; // Show loading message
        });
    </script>
</body>
</html>
"""

# Template for displaying results
RESULTS_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <title>Analysis Results</title>
    <style>
        body { padding: 20px; }
        .results-section { margin-bottom: 30px; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }
        h2 { border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
        h3 { margin-top: 20px; margin-bottom: 15px; color: #333; }
        .plot-container img { max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #ddd; }
        .plotly-plot-container { width: 100%; min-height: 450px; margin: 10px auto; border: 1px solid #ddd; padding: 5px; }
        .table-responsive { max-height: 500px; overflow-y: auto; margin-top: 15px; } /* Scrollable tables */
        .table thead { position: sticky; top: 0; background-color: #f8f9fa; } /* Sticky header for tables */
        .alert { margin-bottom: 20px; }
        .log-box { max-height: 200px; overflow-y: scroll; background-color: #f1f1f1; border: 1px solid #ccc; padding: 10px; font-size: 0.8em; white-space: pre-wrap; word-wrap: break-word; margin-top: 15px; }
        .nav-tabs { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-4">Analysis Results</h1>
        <a href="{{ url_for('index') }}" class="btn btn-secondary mb-3">Run New Analysis</a>

        <!-- Display Errors First -->
        {% if errors %}
        <div class="alert alert-danger">
            <h4>Errors Occurred:</h4>
            <ul>
                {% for error in errors %}
                <li>{{ error }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <!-- Display Warnings/Logs -->
        {% if warnings %}
        <div class="alert alert-warning">
            <h4>Warnings & Logs:</h4>
            <div class="log-box">
                {% for warning in warnings %}
                    {{ warning }}<br>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if not errors %} {# Only show results sections if no fatal errors occurred #}
        <div class="results-section">
            <h2>Summary Table</h2>
            {% if results.summary_table_html %}
            <div class="table-responsive">
                {{ results.summary_table_html | safe }}
            </div>
            {% else %}
            <p>Summary table could not be generated.</p>
            {% endif %}
        </div>

        <div class="results-section">
            <h2>Performance Curves</h2>
            <ul class="nav nav-tabs" id="curveTab" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="roc-tab" data-bs-toggle="tab" data-bs-target="#roc-panel" type="button" role="tab" aria-controls="roc-panel" aria-selected="true">ROC Curves</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="pr-tab" data-bs-toggle="tab" data-bs-target="#pr-panel" type="button" role="tab" aria-controls="pr-panel" aria-selected="false">PR Curves</button>
              </li>
            </ul>
            <div class="tab-content" id="curveTabContent">
              <div class="tab-pane fade show active" id="roc-panel" role="tabpanel" aria-labelledby="roc-tab">
                <h3>ROC Curves</h3>
                {% if results.roc_plot %}
                <div class="plot-container">
                    <img src="{{ results.roc_plot }}" alt="ROC Curves">
                </div>
                {% else %}
                <p>ROC plot could not be generated.</p>
                {% endif %}
              </div>
              <div class="tab-pane fade" id="pr-panel" role="tabpanel" aria-labelledby="pr-tab">
                <h3>Precision-Recall Curves</h3>
                {% if results.pr_plot %}
                <div class="plot-container">
                    <img src="{{ results.pr_plot }}" alt="PR Curves">
                </div>
                {% else %}
                <p>PR plot could not be generated.</p>
                {% endif %}
              </div>
            </div>
        </div>

        <div class="results-section">
            <h2>Feature & Demographic Impact</h2>
             <ul class="nav nav-tabs" id="impactTab" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="feature-tab" data-bs-toggle="tab" data-bs-target="#feature-panel" type="button" role="tab" aria-controls="feature-panel" aria-selected="true">vs Features</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="demo-tab" data-bs-toggle="tab" data-bs-target="#demo-panel" type="button" role="tab" aria-controls="demo-panel" aria-selected="false">vs Demographics</button>
              </li>
            </ul>
             <div class="tab-content" id="impactTabContent">
              <div class="tab-pane fade show active" id="feature-panel" role="tabpanel" aria-labelledby="feature-tab">
                 <h3>AUROC vs Features</h3>
                {% if results.violin_feature_plot_html %}
                <div class="plotly-plot-container">
                    {{ results.violin_feature_plot_html | safe }}
                </div>
                {% else %}
                <p>AUROC vs Features plot could not be generated.</p>
                {% endif %}
              </div>
              <div class="tab-pane fade" id="demo-panel" role="tabpanel" aria-labelledby="demo-tab">
                 <h3>AUROC vs Demographics</h3>
                {% if results.violin_demo_plot_html %}
                <div class="plotly-plot-container">
                    {{ results.violin_demo_plot_html | safe }}
                </div>
                {% else %}
                <p>AUROC vs Demographics plot could not be generated.</p>
                {% endif %}
              </div>
             </div>
        </div>

        <div class="results-section">
            <h2>Population & Patient Risk</h2>
             <ul class="nav nav-tabs" id="riskTab" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="density-pna-tab" data-bs-toggle="tab" data-bs-target="#density-pna-panel" type="button" role="tab" aria-controls="density-pna-panel" aria-selected="true">Density (vs PNA)</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="density-tte-tab" data-bs-toggle="tab" data-bs-target="#density-tte-panel" type="button" role="tab" aria-controls="density-tte-panel" aria-selected="false">Density (vs TTE)</button>
              </li>
               <li class="nav-item" role="presentation">
                <button class="nav-link" id="example-pat-tab" data-bs-toggle="tab" data-bs-target="#example-pat-panel" type="button" role="tab" aria-controls="example-pat-panel" aria-selected="false">Example Patient</button>
              </li>
            </ul>
             <div class="tab-content" id="riskTabContent">
                <div class="tab-pane fade show active" id="density-pna-panel" role="tabpanel" aria-labelledby="density-pna-tab">
                    <h3>Population Risk Density vs PNA</h3>
                    {% if results.density_pna_plot_html %}
                    <div class="plotly-plot-container">
                        {{ results.density_pna_plot_html | safe }}
                    </div>
                    {% else %}
                    <p>Risk Density vs PNA plot could not be generated.</p>
                    {% endif %}
                </div>
                <div class="tab-pane fade" id="density-tte-panel" role="tabpanel" aria-labelledby="density-tte-tab">
                    <h3>Positive Patient Risk Density vs Time to Event</h3>
                    {% if results.density_tte_plot_html %}
                    <div class="plotly-plot-container">
                        {{ results.density_tte_plot_html | safe }}
                    </div>
                    {% else %}
                    <p>Risk Density vs TTE plot could not be generated.</p>
                    {% endif %}
                </div>
                <div class="tab-pane fade" id="example-pat-panel" role="tabpanel" aria-labelledby="example-pat-tab">
                    <h3>Example Patient Trajectory {% if results.selected_patient_id %}({{ results.selected_patient_id[:8] }}){% endif %}</h3>
                    {% if results.example_patient_plot_html %}
                    <div class="plotly-plot-container">
                        {{ results.example_patient_plot_html | safe }}
                    </div>
                    {% else %}
                    <p>Example patient plot could not be generated.</p>
                    {% endif %}
                    <!-- Add dropdown to select patient later? -->
                </div>
             </div>
        </div>

         <div class="results-section">
            <h2>Alarm Rate Analysis</h2>
             <ul class="nav nav-tabs" id="alarmTab" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="alarm-pna-tab" data-bs-toggle="tab" data-bs-target="#alarm-pna-panel" type="button" role="tab" aria-controls="alarm-pna-panel" aria-selected="true">vs PNA</button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="alarm-tte-tab" data-bs-toggle="tab" data-bs-target="#alarm-tte-panel" type="button" role="tab" aria-controls="alarm-tte-panel" aria-selected="false">vs TTE</button>
              </li>
            </ul>
             <div class="tab-content" id="alarmTabContent">
               <div class="tab-pane fade show active" id="alarm-pna-panel" role="tabpanel" aria-labelledby="alarm-pna-tab">
                 <h3>Alarm Rate vs PNA</h3>
                {% if results.alarm_pna_plot %}
                <div class="plot-container">
                    <img src="{{ results.alarm_pna_plot }}" alt="Alarm Rate vs PNA">
                </div>
                {% else %}
                <p>Alarm Rate vs PNA plot could not be generated.</p>
                {% endif %}
               </div>
                <div class="tab-pane fade" id="alarm-tte-panel" role="tabpanel" aria-labelledby="alarm-tte-tab">
                 <h3>Alarm Rate vs Time To Event (Positive Patients)</h3>
                {% if results.alarm_tte_plot %}
                <div class="plot-container">
                    <img src="{{ results.alarm_tte_plot }}" alt="Alarm Rate vs TTE">
                </div>
                {% else %}
                <p>Alarm Rate vs TTE plot could not be generated.</p>
                {% endif %}
                </div>
             </div>
        </div>

        {% endif %} {# End of check for no errors #}

    </div> <!-- /container -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    # Clear previous results from session if desired
    session.pop('results', None)
    session.pop('errors', None)
    session.pop('warnings', None)
    return render_template_string(INDEX_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    results_dir = request.form.get('results_dir')
    model_prefix = request.form.get('model_prefix')
    use_cache = request.form.get('use_cache') == 'yes'

    print(f"Received request: dir='{results_dir}', prefix='{model_prefix}', use_cache={use_cache}")

    # Basic input validation
    if not results_dir or not model_prefix:
        flash('Both Results Directory and Model Prefix are required.', 'danger')
        return redirect(url_for('index'))
    if not os.path.isdir(results_dir):
         flash(f'Results directory not found: {results_dir}', 'danger')
         return redirect(url_for('index'))

    # Define variables to hold results outside the try block
    results = {}
    errors = []
    warnings_list = []

    # Run the analysis function
    try:
        # Store inputs in session just to repopulate the form easily if needed (optional)
        # session['results_dir'] = results_dir
        # session['model_prefix'] = model_prefix

        results, errors, warnings_list = run_analysis(results_dir, model_prefix, use_cache=use_cache)

        # *** REMOVE storing large results in session ***
        # session['results'] = results
        # session['errors'] = errors
        # session['warnings'] = warnings_list

    except Exception as e:
        print(f"FATAL Error during analysis execution: {e}")
        traceback.print_exc()
        # Add error to the list that will be passed to the template
        errors.append(f"An unexpected error occurred during analysis: {e}")

    # Render the results template, passing the results directly
    # The template already expects variables named 'results', 'errors', 'warnings'
    return render_template_string(RESULTS_TEMPLATE,
                                  results=results,
                                  errors=errors,
                                  warnings=warnings_list)

# --- Main Execution ---
if __name__ == "__main__":
    # Note: Running with debug=True is not recommended for production
    # Consider using a production WSGI server like gunicorn or waitress
    print("Starting Flask development server...")
    print(f"Instance path (for cache): {app.instance_path}")
    # Run on 0.0.0.0 to make it accessible on the network if needed
    # Choose a port (default is 5000)
    app.run(host='0.0.0.0', port=5001, debug=True)