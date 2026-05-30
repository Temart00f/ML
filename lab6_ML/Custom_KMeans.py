"""
Библиотека алгоритмов ML — Лабораторная работа 5.
Собственная реализация алгоритма кластеризации K-Means.
"""

import numpy as np


class CustomKMeans:
    """
    Реализация K-Means с подсчётом WCSS (Within-Cluster Sum of Squares).

    Parameters
    ----------
    n_clusters : int
        Количество кластеров.
    max_iter : int
        Максимальное число итераций.
    tol : float
        Порог сходимости (максимальный сдвиг центроиды).
    random_state : int or None
        Seed для воспроизводимости.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """Обучить модель на данных X (n_init перезапусков, лучший по WCSS)."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_centers = None
        best_labels = None
        best_inertia = np.inf

        for _ in range(self.n_init):
            # Случайная инициализация
            idx = rng.choice(len(X), self.n_clusters, replace=False)
            centers = X[idx].copy()

            for iteration in range(self.max_iter):
                labels = self._assign_centers(X, centers)
                new_centers = np.array([
                    X[labels == k].mean(axis=0) if np.any(labels == k)
                    else centers[k]
                    for k in range(self.n_clusters)
                ])
                shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
                centers = new_centers
                if shift < self.tol:
                    break

            inertia = float(sum(
                np.sum((X[labels == k] - centers[k]) ** 2)
                for k in range(self.n_clusters)
                if np.any(labels == k)
            ))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X):
        """Предсказать метки кластеров для X."""
        X = np.asarray(X, dtype=float)
        return self._assign(X)

    def fit_predict(self, X):
        """Обучить и вернуть метки кластеров."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_centers(self, X, centers):
        """E-шаг с явной передачей центроид."""
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def _assign(self, X):
        return self._assign_centers(X, self.cluster_centers_)

    def _wcss(self, X):
        """
        WCSS — сумма квадратов расстояний от точек до центроид своего кластера.
        Аналог sklearn KMeans.inertia_.
        """
        labels = self._assign(X)
        return float(sum(
            np.sum((X[labels == k] - self.cluster_centers_[k]) ** 2)
            for k in range(self.n_clusters)
            if np.any(labels == k)
        ))
