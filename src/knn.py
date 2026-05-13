"""
knn.py
------
k-Nearest Neighbours classifier wrapper around scikit-learn's KNeighborsClassifier.

Responsibility:
    Provide a clean, documented class interface for kNN text classification.
    Uses cosine similarity (via ``metric='cosine'``) which is well-suited to
    high-dimensional, sparse TF-IDF vectors.
"""

import logging

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)


class KNNClassifier:
    """
    k-Nearest Neighbours text classifier.

    Wraps :class:`sklearn.neighbors.KNeighborsClassifier`.
    Cosine distance is used by default because it captures directional
    similarity in the high-dimensional TF-IDF space better than Euclidean
    distance.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to consider. Default ``5``.
    metric : str
        Distance metric. ``"cosine"`` is recommended for TF-IDF. Default ``"cosine"``.
    weights : str
        How to weight neighbours: ``"uniform"`` (equal weight) or ``"distance"``
        (closer neighbours get higher weight). Default ``"distance"``.
    algorithm : str
        Algorithm used to compute nearest neighbours.  ``"brute"`` is required
        when ``metric='cosine'``. Default ``"brute"``.

    Attributes
    ----------
    model : KNeighborsClassifier
        The underlying scikit-learn estimator.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "cosine",
        weights: str = "distance",
        algorithm: str = "brute",
    ) -> None:
        self.k         = k
        self.metric    = metric
        self.weights   = weights
        self.algorithm = algorithm

        self.model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            algorithm=algorithm,
            n_jobs=-1,  # Use all available CPU cores
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: spmatrix, y_train: np.ndarray) -> "KNNClassifier":
        """
        Fit the kNN model (stores training vectors for lookup at prediction time).

        Parameters
        ----------
        X_train : sparse matrix or ndarray, shape (n_train, n_features)
            TF-IDF feature matrix for training samples.
        y_train : array-like, shape (n_train,)
            Class labels for training samples.

        Returns
        -------
        self
        """
        logger.info(
            "Training kNN (k=%d, metric=%s) on %d samples …",
            self.k, self.metric, X_train.shape[0],
        )
        # sklearn's KNeighborsClassifier with brute + cosine works with sparse matrices
        self.model.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X_test: spmatrix) -> np.ndarray:
        """
        Predict class labels for *X_test* using the k nearest training neighbours.

        Parameters
        ----------
        X_test : sparse matrix or ndarray, shape (n_test, n_features)
            TF-IDF feature matrix for test samples.

        Returns
        -------
        numpy.ndarray, shape (n_test,)
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()
        return self.model.predict(X_test)

    def predict_proba(self, X_test: spmatrix) -> np.ndarray:
        """
        Return class probability estimates (vote fractions among neighbours).

        Parameters
        ----------
        X_test : sparse matrix or ndarray, shape (n_test, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_test, n_classes)
        """
        self._check_fitted()
        return self.model.predict_proba(X_test)

    def __repr__(self) -> str:
        return f"KNNClassifier(k={self.k}, metric={self.metric!r})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been trained yet."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
