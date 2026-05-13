"""
evaluate.py
-----------
Evaluation metrics, reporting, and confusion matrix visualisation.

Responsibility:
    Compute and display all standard classification metrics for any
    fitted classifier / prediction pair, and save confusion matrix heatmaps.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Core Metric Computation
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the four primary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels returned by a classifier.

    Returns
    -------
    dict
        Dictionary with keys ``"accuracy"``, ``"precision"``, ``"recall"``,
        ``"f1"`` (all weighted averages except accuracy).
    """
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Return the full per-class classification report as a formatted string.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    str
        Formatted classification report (precision / recall / F1 per class).
    """
    return classification_report(y_true, y_pred, zero_division=0)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    numpy.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    """
    return confusion_matrix(y_true, y_pred)


# ===========================================================================
# Console Reporting
# ===========================================================================

def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a visually distinct section header to stdout.

    Parameters
    ----------
    title : str
        Section title text.
    width : int
        Total character width of the header line. Default 60.
    """
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_metrics(
    algorithm_name: str,
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    show_report: bool = True,
) -> None:
    """
    Print metrics for a single algorithm in a clean, readable format.

    Parameters
    ----------
    algorithm_name : str
        Human-readable name of the algorithm (e.g. ``"Naive Bayes"``).
    metrics : dict
        Metrics dict as returned by :func:`compute_metrics`.
    y_true : array-like
        Ground-truth labels (used for the classification report).
    y_pred : array-like
        Predicted labels (used for the classification report).
    show_report : bool
        Whether to print the full per-class report. Default ``True``.
    """
    print(f"--- {algorithm_name} ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")

    if show_report:
        print("\n  Classification Report:")
        report = get_classification_report(y_true, y_pred)
        # Indent each line of the report for visual clarity
        for line in report.splitlines():
            print(f"    {line}")

    print()  # Blank line separator


# ===========================================================================
# Confusion Matrix Heatmap
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
) -> None:
    """
    Save a confusion matrix heatmap as a PNG file.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list of str
        Ordered list of class label names for axes.
    title : str
        Plot title (e.g. ``"Naive Bayes — Health Tweets"``).
    output_path : str
        Filepath (including filename) where the PNG will be saved.
    """
    cm = get_confusion_matrix(y_true, y_pred)

    # Dynamically size the figure based on number of classes
    n_classes  = len(labels)
    fig_size   = max(8, n_classes * 0.9)
    font_scale = max(0.6, 1.0 - (n_classes - 5) * 0.05)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=font_scale * 9)
    plt.yticks(rotation=0,  fontsize=font_scale * 9)
    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", output_path)


# ===========================================================================
# Final Comparison Table
# ===========================================================================

def print_comparison_table(results: List[Dict]) -> None:
    """
    Print a formatted comparison table of all algorithm / dataset combinations.

    Parameters
    ----------
    results : list of dict
        Each dict must contain the keys:
        ``"algorithm"``, ``"dataset"``, ``"accuracy"``,
        ``"precision"``, ``"recall"``, ``"f1"``.
    """
    print_section_header("FINAL COMPARISON TABLE")

    # Column widths
    col_algo    = 16
    col_dataset = 18
    col_metric  = 10

    header = (
        f"{'Algorithm':<{col_algo}} | "
        f"{'Dataset':<{col_dataset}} | "
        f"{'Accuracy':>{col_metric}} | "
        f"{'Precision':>{col_metric}} | "
        f"{'Recall':>{col_metric}} | "
        f"{'F1':>{col_metric}}"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)

    for row in results:
        line = (
            f"{row['algorithm']:<{col_algo}} | "
            f"{row['dataset']:<{col_dataset}} | "
            f"{row['accuracy']:>{col_metric}.4f} | "
            f"{row['precision']:>{col_metric}.4f} | "
            f"{row['recall']:>{col_metric}.4f} | "
            f"{row['f1']:>{col_metric}.4f}"
        )
        print(line)

    print(separator)
    print()


# ===========================================================================
# Unified Evaluation Entry Point
# ===========================================================================

def evaluate_classifier(
    classifier,
    X_train,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    algorithm_name: str,
    dataset_name: str,
    label_names: List[str],
    output_dir: str = ".",
    show_report: bool = True,
) -> Dict[str, float]:
    """
    Train a classifier, evaluate it, print results, and save the confusion matrix.

    This is the single call in :mod:`main` to fully benchmark one
    algorithm on one dataset.

    Parameters
    ----------
    classifier : object
        An unfitted classifier with ``fit(X, y)`` and ``predict(X)`` methods
        (any of :class:`~src.naive_bayes.NaiveBayesClassifier`,
        :class:`~src.rocchio.RocchioClassifier`,
        :class:`~src.knn.KNNClassifier`).
    X_train : sparse matrix
        Training feature matrix.
    X_test : sparse matrix
        Test feature matrix.
    y_train : array-like
        Training labels.
    y_test : array-like
        Ground-truth test labels.
    algorithm_name : str
        Human-readable name, e.g. ``"Naive Bayes"``.
    dataset_name : str
        Human-readable dataset name, e.g. ``"Health Tweets"``.
    label_names : list of str
        Sorted list of unique class names (for confusion matrix axes).
    output_dir : str
        Directory where the confusion matrix PNG will be saved. Default ``"."``.
    show_report : bool
        Whether to print the full classification report. Default ``True``.

    Returns
    -------
    dict
        Metrics dict with keys ``"algorithm"``, ``"dataset"``,
        ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"``.
    """
    # --- Train ---
    classifier.fit(X_train, y_train)

    # --- Predict ---
    y_pred = classifier.predict(X_test)

    # --- Compute metrics ---
    metrics = compute_metrics(y_test, y_pred)

    # --- Print to console ---
    print_metrics(algorithm_name, metrics, y_test, y_pred, show_report=show_report)

    # --- Save confusion matrix heatmap ---
    safe_algo    = algorithm_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    safe_dataset = dataset_name.lower().replace(" ", "_")
    cm_filename  = f"cm_{safe_algo}_{safe_dataset}.png"
    cm_path      = os.path.join(output_dir, cm_filename)

    try:
        plot_confusion_matrix(
            y_test,
            y_pred,
            labels=label_names,
            title=f"{algorithm_name} — {dataset_name}",
            output_path=cm_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save confusion matrix for %s / %s: %s", algorithm_name, dataset_name, exc)

    # --- Return structured result for the comparison table ---
    return {
        "algorithm": algorithm_name,
        "dataset":   dataset_name,
        **metrics,
    }
