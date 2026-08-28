"""SVM model training, including a grid search over hyperparameters."""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score


def train_baseline(X_train, y_train, kernel="rbf"):
    """Train an SVM with scikit-learn's default hyperparameters."""
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_all_kernels(X_train, y_train, X_test, y_test):
    """
    Compare the four standard kernels.

    linear  -- straight-line (hyperplane) boundary
    rbf     -- gaussian, handles curved boundaries, the usual default
    poly    -- polynomial boundary
    sigmoid -- tanh-based, rarely the best choice but worth showing
    """
    results = {}
    for kernel in ["linear", "rbf", "poly", "sigmoid"]:
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train, y_train)
        results[kernel] = {
            "model": model,
            "train_accuracy": model.score(X_train, y_train),
            "test_accuracy": model.score(X_test, y_test),
            "n_support_vectors": int(model.n_support_.sum()),
        }

    print("=" * 60)
    print("KERNEL COMPARISON")
    print("=" * 60)
    print(f"{'Kernel':<10}{'Train Acc':>12}{'Test Acc':>12}{'Support Vecs':>15}")
    for kernel, r in results.items():
        print(f"{kernel:<10}{r['train_accuracy']:>12.4f}"
              f"{r['test_accuracy']:>12.4f}{r['n_support_vectors']:>15}")
    print()
    return results


def tune_hyperparameters(X_train, y_train, cv=5):
    """
    Grid search over C, gamma and kernel using cross-validation.

    C     -- regularization. Low C gives a wide margin that tolerates some
             misclassification; high C forces a tighter fit to the training
             data and risks overfitting.
    gamma -- how far a single training point's influence reaches. Low gamma
             means far-reaching and smooth; high gamma means each point only
             affects its immediate neighbourhood, again risking overfitting.
    """
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", 1, 0.1, 0.01, 0.001],
        "kernel": ["rbf", "linear"],
    }

    grid = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Best parameters   : {grid.best_params_}")
    print(f"Best CV accuracy  : {grid.best_score_:.4f}\n")

    return grid.best_estimator_, grid


def cross_validate_model(model, X, y, cv=5):
    """Report cross-validated accuracy with a spread, not just one number."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{cv}-fold CV accuracy: {scores.mean():.4f} "
          f"(+/- {scores.std() * 2:.4f})")
    print(f"Individual folds  : {[f'{s:.4f}' for s in scores]}\n")
    return scores


def compare_scaling_effect(X_train_raw, X_test_raw,
                           X_train_scaled, X_test_scaled,
                           y_train, y_test):
    """
    Demonstrate why scaling matters for SVMs.

    Train the same model on raw and on standardized features and compare.
    This is usually one of the more striking results in the whole project.
    """
    unscaled = SVC(kernel="rbf", random_state=42).fit(X_train_raw, y_train)
    scaled = SVC(kernel="rbf", random_state=42).fit(X_train_scaled, y_train)

    acc_unscaled = unscaled.score(X_test_raw, y_test)
    acc_scaled = scaled.score(X_test_scaled, y_test)

    print("=" * 60)
    print("EFFECT OF FEATURE SCALING")
    print("=" * 60)
    print(f"Without scaling : {acc_unscaled:.4f}")
    print(f"With scaling    : {acc_scaled:.4f}")
    print(f"Improvement     : {acc_scaled - acc_unscaled:+.4f}\n")

    return acc_unscaled, acc_scaled