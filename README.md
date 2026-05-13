# Text Classification System

A modular text classification system that benchmarks **Naive Bayes**, **Rocchio**, and **k-Nearest Neighbors (kNN)** classifiers on two real-world datasets: *Health News in Twitter* and *DBWorld Emails*. Evaluation includes accuracy, precision, recall, F1-score, classification reports, and confusion matrix heatmaps.

---

## Folder Structure

```
project/
│
├── data/
│   ├── health_tweets/          # All Health News Twitter .txt files
│   │   ├── bbchealth.txt
│   │   ├── cnnhealth.txt
│   │   └── ... (all other files)
│   └── dbworld/                # DBWorld Emails dataset files
│       ├── index               # Labels file (one per line: 1 or -1 / announces / not)
│       └── ...
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Text cleaning, stopword removal, stemming, TF-IDF
│   ├── naive_bayes.py          # Naive Bayes classifier wrapper
│   ├── rocchio.py              # Rocchio (NearestCentroid) classifier wrapper
│   ├── knn.py                  # kNN classifier wrapper
│   └── evaluate.py             # Metrics computation and reporting
│
├── main.py                     # Entry point — runs everything end-to-end
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Navigate to the project directory
cd project

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Download required NLTK data (runs automatically on first execution,
#    but you can also run it manually once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Dataset Setup

### Dataset 1 — Health News in Twitter
1. Download from: https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter
2. Extract and copy **all `.txt` files** (e.g., `bbchealth.txt`, `cnnhealth.txt`, etc.) into:
   ```
   project/data/health_tweets/
   ```

### Dataset 2 — DBWorld Emails
1. Download from: https://archive.ics.uci.edu/ml/datasets/DBWorld+e-mails
2. Extract and copy **all dataset files** into:
   ```
   project/data/dbworld/
   ```
   The loader expects either:
   - An `index` file listing labels, or
   - Subdirectories named `announces/` and `not/` containing email text files.

---

## How to Run

```bash
python main.py
```

The script will:
1. Load and preprocess both datasets
2. Train all three classifiers on each dataset
3. Print per-algorithm metrics to the console
4. Display a final comparison table
5. Save confusion matrix heatmaps as PNG files in the project root

---

## Output Example

```
============================================================
DATASET: Health News in Twitter
============================================================

--- Naive Bayes ---
Accuracy  : 0.91
Precision : 0.90
Recall    : 0.91
F1-Score  : 0.90

--- Rocchio ---
...

--- kNN (k=5) ---
...

============================================================
FINAL COMPARISON TABLE
============================================================
Algorithm      | Dataset         | Accuracy | Precision | Recall | F1
...
```
