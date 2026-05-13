"""
preprocessing.py
----------------
Handles all text preprocessing and TF-IDF vectorization steps.
A single reusable pipeline is applied to both datasets.

Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation / special characters
    4. Remove digits
    5. Remove English stopwords (NLTK)
    6. Stem with PorterStemmer (NLTK)
    7. Strip extra whitespace
    8. Vectorize with TF-IDF (scikit-learn)
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import spmatrix

# ---------------------------------------------------------------------------
# Ensure required NLTK corpora are available
# ---------------------------------------------------------------------------
def _ensure_nltk_data() -> None:
    """Download required NLTK datasets if they are not already present."""
    for resource, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt",     "tokenizers/punkt"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

# Module-level singletons (initialised once, reused across calls)
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))

logger = logging.getLogger(__name__)


# ===========================================================================
# Text Cleaning
# ===========================================================================

def clean_text(text: str) -> str:
    """
    Super-enhanced cleaning: extracts domains and protects source identifiers.
    """
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()
    
    # 1. Extract URL Domains
    domains = re.findall(r"https?://(?:www\.)?([^/]+)", text_lower)
    domain_features = ["domain_" + d.replace(".", "_") for d in domains]

    # 2. Protect Source Keywords (ensure they stay as tokens)
    sources = ['bbc', 'cnn', 'nytimes', 'fox', 'reuters', 'npr', 'wsj', 'msn', 'cbc']
    found_sources = ["source_" + s for s in sources if s in text_lower]

    # 3. Standard cleaning
    text = re.sub(r"https?://\S+|www\.\S+", " ", text_lower)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # 4. Tokenize and Stem
    tokens = text.split()
    tokens = [
        _stemmer.stem(token)
        for token in tokens
        if (token not in _stop_words and len(token) > 1) or token in sources
    ]

    # 5. Combine everything
    return " ".join(tokens + domain_features + found_sources)


def clean_corpus(documents: List[str]) -> List[str]:
    """
    Apply :func:`clean_text` to every document in a list.

    Parameters
    ----------
    documents : list of str
        Raw text documents.

    Returns
    -------
    list of str
        Cleaned documents.
    """
    return [clean_text(doc) for doc in documents]


# ===========================================================================
# Dataset Loaders
# ===========================================================================

def load_health_tweets(data_dir: str) -> pd.DataFrame:
    """
    Load the Health News in Twitter dataset.

    Each `.txt` file in *data_dir* is treated as a separate class. The label
    is derived from the filename (without the `.txt` extension, e.g. 
    ``bbchealth``).  Each line in a file has the format::

        tweet_id | date_time | tweet_text

    Only the **tweet_text** (3rd column) is extracted.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the health-tweet `.txt` files.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['text', 'label']``.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist or contains no `.txt` files.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Health tweets directory not found: {data_dir}")

    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {data_dir}")

    records = []
    for filepath in txt_files:
        label = filepath.stem  # e.g. "bbchealth"
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 3:
                        tweet_text = parts[2].strip()
                        if tweet_text:
                            records.append({"text": tweet_text, "label": label})
        except OSError as exc:
            logger.warning("Could not read %s: %s", filepath, exc)

    if not records:
        raise ValueError("No valid tweet records were parsed from the health tweets files.")

    df = pd.DataFrame(records)
    logger.info("Health Tweets: loaded %d samples across %d classes.", len(df), df["label"].nunique())
    return df


from scipy.io import arff

def load_dbworld(data_dir: str) -> pd.DataFrame:
    """
    Load the DBWorld dataset from ARFF files in the 'weka' subdirectory.
    
    Reconstructs text by joining attribute names that have a value of 1.
    """
    data_path = Path(data_dir) / "weka"
    if not data_path.exists():
        raise FileNotFoundError(f"DBWorld weka directory not found: {data_path}")

    # We can use either bodies or subjects. Bodies usually has more info.
    arff_file = data_path / "dbworld_bodies.arff"
    if not arff_file.exists():
        # Fallback to subjects if bodies isn't there
        arff_file = data_path / "dbworld_subjects.arff"
        if not arff_file.exists():
            raise FileNotFoundError(f"No ARFF files found in {data_path}")

    logger.info("Loading DBWorld from ARFF: %s", arff_file.name)
    
    try:
        data, meta = arff.loadarff(arff_file)
        df_raw = pd.DataFrame(data)
        
        # Attributes names (excluding the last one which is usually the CLASS)
        attributes = meta.names()[:-1]
        class_attr = meta.names()[-1]
        
        records = []
        for _, row in df_raw.iterrows():
            # Reconstruct text: join words that are present (value 1 or b'1')
            words = [attr for attr in attributes if str(row[attr]).strip() in ("1", "1.0", "b'1'", "b'1.0'")]
            text = " ".join(words)
            
            # Label is the last column
            label = str(row[class_attr]).strip()
            # Clean up byte strings if necessary (e.g. b'0' -> 0)
            if label.startswith("b'"):
                label = label[2:-1]
            
            if text:
                records.append({"text": text, "label": f"class_{label}"})
                
        if not records:
            raise ValueError("No records could be reconstructed from the ARFF file.")
            
        df = pd.DataFrame(records)
        logger.info("DBWorld: loaded %d samples.", len(df))
        return df

    except Exception as exc:
        logger.error("Error parsing ARFF file: %s", exc)
        raise


# ===========================================================================
# Vectorization
# ===========================================================================

from sklearn.feature_selection import SelectKBest, chi2

# Expanded stopwords for health news context
_HEALTH_STOPWORDS = {"health", "news", "medical", "doctor", "new", "study", "says", "may"}
_stop_words.update(_HEALTH_STOPWORDS)

class TextVectorizer:
    """
    High-capacity vectorizer with n-grams up to 4-words and broad feature selection.
    """
    def __init__(self, max_features: int = 60_000) -> None:
        # Use 'word' analyzer for DBWorld (topics) and high n-grams for Twitter (style)
        # To hit 97% on DBWorld, word-level context is critical.
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=max_features,
            sublinear_tf=True,
            min_df=1, # Crucial for small datasets like DBWorld
            strip_accents="unicode",
        )
        self.selector = SelectKBest(chi2, k=min(15_000, max_features))
        self._fitted = False

    def fit_transform(self, documents: List[str], y: np.ndarray) -> spmatrix:
        """Fit vectorizer and selector, then transform."""
        X = self.vectorizer.fit_transform(documents)
        X_selected = self.selector.fit_transform(X, y)
        self._fitted = True
        return X_selected

    def transform(self, documents: List[str]) -> spmatrix:
        """Transform documents using fitted models."""
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted.")
        X = self.vectorizer.transform(documents)
        return self.selector.transform(X)

def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    max_features: int = 60_000,
) -> Tuple[spmatrix, spmatrix, np.ndarray, np.ndarray, TextVectorizer]:
    """Run preprocessing, split, and feature-selected vectorization."""
    logger.info("Cleaning and preparing dataset...")
    cleaned_texts = clean_corpus(df["text"].tolist())
    labels = df["label"].values

    texts_train, texts_test, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    vectorizer = TextVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(texts_train, y_train)
    X_test  = vectorizer.transform(texts_test)

    return X_train, X_test, y_train, y_test, vectorizer
