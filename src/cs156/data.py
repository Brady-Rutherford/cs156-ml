"""Dataset loaders. Both datasets are fetched by scikit-learn; nothing is committed to git."""
from sklearn.datasets import fetch_openml, load_iris


def iris():
    """Return (X, y, feature_names, target_names) for the Iris dataset."""
    d = load_iris()
    return d.data, d.target, list(d.feature_names), list(d.target_names)


def mnist():
    """Return (X, y) for MNIST as numpy arrays. X is (70000, 784) uint8; y is int labels.

    First call downloads ~15MB and caches it under ~/scikit_learn_data.
    """
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return X.astype("uint8"), y.astype("int64")
