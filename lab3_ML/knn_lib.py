
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomKNN(BaseEstimator, ClassifierMixin):
    """
    Реализация алгоритма k ближайших соседей, совместимая со sklearn.

    Параметры
    ---------
    n_neighbors : int   — количество ближайших соседей (k)
    metric      : str   — метрика расстояния: 'euclidean', 'manhattan', 'chebyshev'
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    # ── обучение (запоминаем обучающую выборку) ──────────────────────────────
    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        self.classes_ = np.unique(self.y_train_)
        return self

    # ── вычисление расстояний (векторизованно) ───────────────────────────────
    def _distances(self, x: np.ndarray) -> np.ndarray:
        diff = self.X_train_ - x          # (n_train, n_features)
        if self.metric == "euclidean":
            return np.sqrt(np.einsum("ij,ij->i", diff, diff))
        elif self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=1)
        elif self.metric == "chebyshev":
            return np.max(np.abs(diff), axis=1)
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}")

    # ── предсказание классов ─────────────────────────────────────────────────
    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        result = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            dist = self._distances(x)
            k_idx = np.argpartition(dist, self.n_neighbors)[: self.n_neighbors]
            k_labels = self.y_train_[k_idx]
            result[i] = np.bincount(k_labels).argmax()
        return result

    # ── вероятности классов ──────────────────────────────────────────────────
    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        probas = np.empty((len(X), n_classes))
        for i, x in enumerate(X):
            dist = self._distances(x)
            k_idx = np.argpartition(dist, self.n_neighbors)[: self.n_neighbors]
            k_labels = self.y_train_[k_idx]
            counts = np.bincount(k_labels, minlength=n_classes)
            probas[i] = counts / self.n_neighbors
        return probas
