# Run after cell2_ols.py
#Log Loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def logistic_classifier(n_components):
    Z_train, Z_test = project(n_components)
    clf = LogisticRegression(max_iter=5000).fit(Z_train, y_train)   # minimizes log loss
    p_test = clf.predict_proba(Z_test)[:, 1]
    return (accuracy_score(y_train, clf.predict(Z_train)),
            accuracy_score(y_test, clf.predict(Z_test)),
            log_loss(y_test, p_test),
            clf.predict(Z_test))

print("Logistic regression (log loss) classifier")
print(f"{'components':>12} {'train acc':>10} {'test acc':>10} {'test logloss':>13}")
log_test = []
for n in COMPONENTS:
    tr, te, ll, _ = logistic_classifier(n)
    log_test.append(te)
    print(f"{'raw pixels' if n is None else n:>12} {tr:>10.3f} {te:>10.3f} {ll:>13.3f}")

_, _, _, pred_k = logistic_classifier(k)
print(f"\nConfusion matrix with k = {k} components (rows = true {NAMES}, cols = predicted):")
print(confusion_matrix(y_test, pred_k))

# Compare both losses
names = [("raw" if n is None else str(n)) for n in COMPONENTS]
xs = np.arange(len(names))
plt.figure(figsize=(7, 4))
plt.plot(xs, ols_test, "o-", label="OLS (squared loss)")
plt.plot(xs, log_test, "s-", label="Logistic (log loss)")
plt.xticks(xs, names)
plt.xlabel("number of PCA components")
plt.ylabel("test accuracy")
plt.title("T-shirts vs shirts: test accuracy by representation")
plt.legend()
plt.savefig("6_classifier_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
