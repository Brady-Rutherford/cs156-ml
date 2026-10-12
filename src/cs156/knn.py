"""k-nearest neighbours, written from scratch in numpy. Import from notebooks with:

    from cs156.knn import KNN

Run it directly to see it classify Iris:

    uv run python -m cs156.knn
"""
import numpy as np


class KNN:
    """Classify a point by majority vote among its k closest training points."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """Store the training data. kNN does no work until it is asked to predict."""
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        """Return the majority label of the k nearest training points for each row of X."""
        X = np.asarray(X, dtype=float)
        # Squared Euclidean distance from every query row to every training row, shape (n_query, n_train).
        dists = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argsort(dists, axis=1)[:, : self.k]
        votes = self.y[nearest]
        # Ties go to whichever tied label sorts first, because np.unique returns labels sorted.
        return np.array([_majority(row) for row in votes])

    def score(self, X, y):
        """Fraction of rows in X whose predicted label matches y."""
        return float((self.predict(X) == np.asarray(y)).mean())


def _majority(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from cs156.data import iris

    X, y, _, target_names = iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    for k in (1, 3, 5, 15):
        acc = KNN(k=k).fit(X_train, y_train).score(X_test, y_test)
        print(f"k={k:>2}  test accuracy {acc:.3f}")
