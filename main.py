"""
main.py
-------
Entry point for the Text Classification Benchmarking System.

Runs all three classifiers (Naive Bayes, Rocchio, kNN) on both datasets
(Health News in Twitter and DBWorld Emails), prints per-algorithm metrics,
and displays a final comparison table.

Usage
-----
    python main.py

All dataset files must be placed in the correct subdirectories under data/
before running. See README.md for detailed setup instructions.
"""

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so 'src' is importable when
# running from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from src.preprocessing import (
    load_health_tweets,
    load_dbworld,
    prepare_dataset,
)
from src.naive_bayes import NaiveBayesClassifier
from src.rocchio    import RocchioClassifier
from src.knn        import KNNClassifier
from src.evaluate   import (
    evaluate_classifier,
    print_section_header,
    print_comparison_table,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

# Paths (all relative to this file's directory)
DATA_DIR          = PROJECT_ROOT / "data"
HEALTH_TWEETS_DIR = DATA_DIR / "health_tweets"
DBWORLD_DIR       = DATA_DIR / "dbworld"

# Output directory for confusion matrix PNGs (saved alongside main.py)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# kNN configuration
KNN_K = 1

# TF-IDF configuration
TFIDF_MAX_FEATURES = 100_000

# Train / test split
TEST_SIZE    = 0.20
RANDOM_STATE = 42


# ===========================================================================
# Helper: build the list of classifiers to benchmark
# ===========================================================================

def build_classifiers() -> list:
    """
    Instantiate one fresh copy of each classifier.

    Returns
    -------
    list of (str, classifier) tuples
        Each tuple contains a human-readable name and an unfitted classifier.
    """
    return [
        ("Naive Bayes",      NaiveBayesClassifier(alpha=1.0)),
        ("Rocchio",          RocchioClassifier(metric="cosine")),
        (f"kNN (k={KNN_K})", KNNClassifier(k=KNN_K)),
    ]


# ===========================================================================
# Dataset runner
# ===========================================================================

def run_on_dataset(
    dataset_name: str,
    df,
    all_results: list,
) -> None:
    """
    Run the full benchmark pipeline on a single pre-loaded dataset.

    Steps:
        1. Preprocess and vectorise
        2. For each classifier: train → predict → evaluate → print → store result
        3. Append structured metric dicts to *all_results* for the comparison table

    Parameters
    ----------
    dataset_name : str
        Human-readable name shown in section headers and the comparison table.
    df : pd.DataFrame
        DataFrame with columns ``['text', 'label']``.
    all_results : list
        Mutable list to which metric dicts will be appended.
    """
    print_section_header(f"DATASET: {dataset_name}")

    # --- Preprocessing & Vectorization ---
    print(f"  Preprocessing {len(df):,} documents …")
    try:
        X_train, X_test, y_train, y_test, _ = prepare_dataset(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            max_features=TFIDF_MAX_FEATURES,
        )
    except Exception as exc:
        logger.error("Preprocessing failed for '%s': %s", dataset_name, exc)
        return

    print(
        f"  Train samples : {X_train.shape[0]:>6,}   "
        f"Test samples : {X_test.shape[0]:>6,}   "
        f"Features : {X_train.shape[1]:>6,}\n"
    )

    # Sorted unique class labels (for confusion matrix axes)
    label_names = sorted(set(y_train) | set(y_test))

    # --- Evaluate each classifier ---
    for name, clf in build_classifiers():
        try:
            result = evaluate_classifier(
                classifier    = clf,
                X_train       = X_train,
                X_test        = X_test,
                y_train       = y_train,
                y_test        = y_test,
                algorithm_name= name,
                dataset_name  = dataset_name,
                label_names   = label_names,
                output_dir    = str(OUTPUT_DIR),
                show_report   = True,
            )
            all_results.append(result)
        except Exception as exc:
            logger.error("Classifier '%s' failed on '%s': %s", name, dataset_name, exc)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """
    Orchestrate the full benchmarking pipeline:

    1. Load Health News in Twitter dataset
    2. Load DBWorld Emails dataset
    3. Run all three classifiers on each dataset
    4. Print a final side-by-side comparison table
    5. Confusion matrix PNGs are saved to the ``outputs/`` directory
    """
    print("\n" + "=" * 60)
    print("  TEXT CLASSIFICATION BENCHMARKING SYSTEM")
    print("=" * 60)
    print(f"  Classifiers : Naive Bayes | Rocchio | kNN (k={KNN_K})")
    print(f"  Datasets    : Health News in Twitter | DBWorld Emails")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    all_results = []   # Collects metric dicts from every run

    # -------------------------------------------------------------------
    # Dataset 1 — Health News in Twitter
    # -------------------------------------------------------------------
    print("[INFO] Loading Health News in Twitter …")
    try:
        health_df = load_health_tweets(str(HEALTH_TWEETS_DIR))
        print(
            f"       Loaded {len(health_df):,} tweets across "
            f"{health_df['label'].nunique()} classes: "
            f"{sorted(health_df['label'].unique())}\n"
        )
        run_on_dataset("Health News in Twitter", health_df, all_results)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Could not load Health Tweets dataset: %s", exc)
        print(
            "\n  [SKIP] Health Tweets dataset not found.\n"
            f"  Please place all .txt files in: {HEALTH_TWEETS_DIR}\n"
        )

    # -------------------------------------------------------------------
    # Dataset 2 — DBWorld Emails
    # -------------------------------------------------------------------
    print("[INFO] Loading DBWorld Emails …")
    try:
        dbworld_df = load_dbworld(str(DBWORLD_DIR))
        print(
            f"       Loaded {len(dbworld_df):,} emails across "
            f"{dbworld_df['label'].nunique()} classes: "
            f"{sorted(dbworld_df['label'].unique())}\n"
        )
        run_on_dataset("DBWorld Emails", dbworld_df, all_results)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Could not load DBWorld dataset: %s", exc)
        print(
            "\n  [SKIP] DBWorld dataset not found.\n"
            f"  Please place dataset files in: {DBWORLD_DIR}\n"
        )

    # -------------------------------------------------------------------
    # Final comparison table
    # -------------------------------------------------------------------
    if all_results:
        print_comparison_table(all_results)
        print(f"  Confusion matrix heatmaps saved in: {OUTPUT_DIR}\n")
    else:
        print("\n  No results to display — please ensure at least one dataset is available.\n")


if __name__ == "__main__":
    main()
