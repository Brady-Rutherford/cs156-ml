# Run after the PCA cells in eigenshirts_colab.ipynb
#OLS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Same split for both classifiers, stratified so both classes are balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
COMPONENTS = sorted({2, 5, 10, 20, 50, k}) + [None]   # None = raw pixels, no PCA

def project(n_components):
    """PCA fitted on the training images only, so no test information leaks in."""
    if n_components is None:
        return X_train, X_test
    pca = PCA(n_components).fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)

def ols_classifier(n_components):
    Z_train, Z_test = project(n_components)
    ols = LinearRegression().fit(Z_train, y_train)   # minimizes sum of squared errors
    pred_train = (ols.predict(Z_train) >= 0.5).astype(int)
    pred_test = (ols.predict(Z_test) >= 0.5).astype(int)
    return accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test), pred_test

print("OLS (squared loss) classifier")
print(f"{'components':>12} {'train acc':>10} {'test acc':>10}")
ols_test = []
for n in COMPONENTS:
    tr, te, _ = ols_classifier(n)
    ols_test.append(te)
    print(f"{'raw pixels' if n is None else n:>12} {tr:>10.3f} {te:>10.3f}")

_, _, pred_k = ols_classifier(k)
print(f"\nConfusion matrix with k = {k} components (rows = true {NAMES}, cols = predicted):")
print(confusion_matrix(y_test, pred_k))
