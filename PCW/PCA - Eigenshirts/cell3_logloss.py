# Log Loss
# Uses X_train, X_test, y_train, y_test from the OLS cell so both models see the same split.
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

def logistic_classifier(n_components):
    """PCA fitted on training data only, then logistic regression (minimizes log loss)."""
    if n_components is None:
        Z_train, Z_test = X_train, X_test
    else:
        pca = PCA(n_components).fit(X_train)
        Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
    clf = LogisticRegression(max_iter=5000).fit(Z_train, y_train)
    p_test = clf.predict_proba(Z_test)[:, 1]
    return (accuracy_score(y_train, clf.predict(Z_train)),
            accuracy_score(y_test, clf.predict(Z_test)),
            log_loss(y_test, p_test),
            clf.predict(Z_test))

print("Logistic regression (log loss) classifier")
print(f"{'components':>12} {'train acc':>10} {'test acc':>10} {'test logloss':>13}")
for n in [2, 5, 10, 20, 50, k, None]:
    tr, te, ll, _ = logistic_classifier(n)
    name = "raw pixels" if n is None else str(n)
    print(f"{name:>12} {tr:>10.3f} {te:>10.3f} {ll:>13.3f}")

_, _, _, pred_k = logistic_classifier(k)
print(f"\nConfusion matrix with k = {k} components (rows = true, cols = predicted):")
print(confusion_matrix(y_test, pred_k))
