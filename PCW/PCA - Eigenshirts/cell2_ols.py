# OLS
# Assumes Code Cell 1 already ran, so X (n_images x 12288), y (0 = jersey, 1 = shirt) and k exist.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Same split for both cells, stratified so both classes are balanced in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

def ols_classifier(n_components):
    """PCA fitted on training data only, then OLS on the PCA scores, threshold at 0.5."""
    if n_components is None:                      # raw pixels, no PCA
        Z_train, Z_test = X_train, X_test
    else:
        pca = PCA(n_components).fit(X_train)
        Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
    ols = LinearRegression().fit(Z_train, y_train)   # minimizes sum of squared errors
    pred_train = (ols.predict(Z_train) >= 0.5).astype(int)
    pred_test = (ols.predict(Z_test) >= 0.5).astype(int)
    return accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test), pred_test

print("OLS (squared loss) classifier")
print(f"{'components':>12} {'train acc':>10} {'test acc':>10}")
for n in [2, 5, 10, 20, 50, k, None]:
    tr, te, _ = ols_classifier(n)
    name = "raw pixels" if n is None else str(n)
    print(f"{name:>12} {tr:>10.3f} {te:>10.3f}")

_, _, pred_k = ols_classifier(k)
print(f"\nConfusion matrix with k = {k} components (rows = true, cols = predicted):")
print(confusion_matrix(y_test, pred_k))
