"""
rocchio.py
----------
Rocchio classifier wrapper around scikit-learn's NearestCentroid.

Responsibility:
    Provide a clean, documented class interface for the Rocchio algorithm,
    which classifies documents by finding the nearest class centroid in
    TF-IDF feature space.

Background:
    The Rocchio algorithm computes a prototype (centroid) for each class from
    the training vectors.  At prediction time, a document is assigned to the
    class whose centroid is closest (by the chosen metric).
    sklearn's NearestCentroid is the canonical sklearn implementation.
"""

import logging

import numpy as np
from sklearn.neighbors import NearestCentroid
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)


class RocchioClassifier:
    """
    Rocchio (Nearest-Centroid) text classifier.

    Wraps :class:`sklearn.neighbors.NearestCentroid`.  Computes one centroid
    per class from the training feature matrix and assigns each test document
    to the nearest centroid using cosine or Euclidean distance.

    Parameters
    ----------
    metric : str
        Distance metric used to compare document vectors to class centroids.
        ``"cosine"`` is strongly recommended for TF-IDF features.
        Default ``"cosine"``.
    shrink_threshold : float or None
        Threshold for shrinking centroids (reduces noise from rare features).
        ``None`` disables shrinkage. Default ``None``.

    Attributes
    ----------
    model : NearestCentroid
        The underlying scikit-learn estimator.
    """

    def __init__(self, metric: str = "cosine", shrink_threshold: float = None) -> None:
        self.metric            = metric
        self.shrink_threshold  = shrink_threshold
        self.model             = NearestCentroid(
            metric=metric,
            shrink_threshold=shrink_threshold,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: spmatrix, y_train: np.ndarray) -> "RocchioClassifier":
        """
        Train the Rocchio classifier using sparse-compatible operations.
        """
        logger.info("Training Rocchio on %d samples …", X_train.shape[0])
        # We use the sparse matrix directly. Euclidean on L2-normalized 
        # TF-IDF vectors is a valid proxy for centroid distance.
        self.model.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X_test: spmatrix) -> np.ndarray:
        """
        Predict using sparse matrix directly.
        """
        self._check_fitted()
        return self.model.predict(X_test)

    def __repr__(self) -> str:
        return f"RocchioClassifier(metric={self.metric!r})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been trained yet."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
