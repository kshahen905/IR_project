"""
naive_bayes.py
--------------
Naive Bayes text classifier wrapper around scikit-learn's MultinomialNB.

Responsibility:
    Provide a clean, documented class interface for training and predicting
    with Multinomial Naive Bayes on TF-IDF or count-vectorised features.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for text classification.

    Wraps :class:`sklearn.naive_bayes.MultinomialNB`.  Because TF-IDF values
    can be non-negative floats, a :class:`~sklearn.preprocessing.MaxAbsScaler`
    is applied internally to scale features to [0, 1] while preserving
    sparsity — this satisfies MultinomialNB's non-negativity requirement.

    Parameters
    ----------
    alpha : float
        Laplace / Lidstone smoothing parameter. Default ``1.0``.

    Attributes
    ----------
    model : MultinomialNB
        The underlying scikit-learn estimator.
    scaler : MaxAbsScaler
        Feature scaler fitted on training data.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha  = alpha
        self.model  = MultinomialNB(alpha=alpha)
        self.scaler = MaxAbsScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: spmatrix, y_train: np.ndarray) -> "NaiveBayesClassifier":
        """
        Train the Naive Bayes model.

        Parameters
        ----------
        X_train : sparse matrix, shape (n_train, n_features)
            TF-IDF feature matrix for training samples.
        y_train : array-like, shape (n_train,)
            Class labels for training samples.

        Returns
        -------
        self
        """
        logger.info("Training Naive Bayes (alpha=%.2f) on %d samples …", self.alpha, X_train.shape[0])
        # Scale to [0,1] to satisfy MultinomialNB's constraint
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._fitted = True
        return self

    def predict(self, X_test: spmatrix) -> np.ndarray:
        """
        Predict class labels for *X_test*.

        Parameters
        ----------
        X_test : sparse matrix, shape (n_test, n_features)
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
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def predict_proba(self, X_test: spmatrix) -> np.ndarray:
        """
        Return class probability estimates.

        Parameters
        ----------
        X_test : sparse matrix, shape (n_test, n_features)

        Returns
        -------
        numpy.ndarray, shape (n_test, n_classes)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)

    def __repr__(self) -> str:
        return f"NaiveBayesClassifier(alpha={self.alpha})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been trained yet."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
