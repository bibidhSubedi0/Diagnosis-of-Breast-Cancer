"""Plotting functions. Every figure is saved to results/figures/."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # write files without needing a display (helps on WSL)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC

FIGURE_DIR = "results/figures"
sns.set_theme(style="whitegrid")


def _save(fig, filename):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_class_distribution(y, target_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    ax.bar([target_names[i] for i in counts.index], counts.values,
           color=["#c44e52", "#4c72b0"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center")
    _save(fig, "class_distribution.png")


def plot_feature_correlation(X, n_features=10):
    """Correlation heatmap for the first n features, to keep it readable."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = X.iloc[:, :n_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Feature Correlation (first {n_features} features)")
    _save(fig, "feature_correlation.png")


def plot_confusion_matrix(cm, target_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    _save(fig, "confusion_matrix.png")


def plot_roc_curve(roc_data):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(roc_data["fpr"], roc_data["tpr"], lw=2,
            label=f"SVM (AUC = {roc_data['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    _save(fig, "roc_curve.png")


def plot_kernel_comparison(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    kernels = list(results.keys())
    train = [results[k]["train_accuracy"] for k in kernels]
    test = [results[k]["test_accuracy"] for k in kernels]

    x = np.arange(len(kernels))
    width = 0.35
    ax.bar(x - width / 2, train, width, label="Train", color="#4c72b0")
    ax.bar(x + width / 2, test, width, label="Test", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by Kernel")
    ax.legend()
    _save(fig, "kernel_comparison.png")


def plot_scaling_effect(acc_unscaled, acc_scaled):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Unscaled", "Standardized"], [acc_unscaled, acc_scaled],
                  color=["#c44e52", "#55a868"])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Effect of Feature Scaling on SVM")
    for bar, val in zip(bars, [acc_unscaled, acc_scaled]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.4f}", ha="center")
    _save(fig, "scaling_effect.png")


def plot_decision_boundary(X_train_scaled, y_train, target_names,
                           kernel="rbf", C=1.0):
    """
    The centrepiece figure.

    The data has 30 dimensions, which cannot be drawn. PCA compresses it to
    the 2 directions carrying the most variance, and a fresh SVM is trained
    on just those two so the boundary drawn is genuinely the boundary of the
    model being shown.

    Circled points are the support vectors: the training samples sitting on
    or inside the margin. These alone determine where the boundary goes,
    which is the defining property of an SVM.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_train_scaled)

    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(X_2d, y_train)

    # Build a grid covering the plot area and classify every point on it.
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")

    y_arr = np.asarray(y_train)
    for label, name, colour in zip([0, 1], target_names, ["#c44e52", "#4c72b0"]):
        mask = y_arr == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colour, s=30,
                   edgecolors="k", linewidths=0.4, label=name)

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=140, facecolors="none", edgecolors="black", linewidths=1.2,
               label=f"Support vectors ({len(model.support_vectors_)})")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} of variance)")
    ax.set_title(f"SVM Decision Boundary — {kernel} kernel, C={C}")
    ax.legend(loc="best")
    _save(fig, f"decision_boundary_{kernel}.png")


def plot_C_effect(X_train_scaled, y_train, target_names):
    """
    Four panels showing what C actually does.

    Small C: wide margin, many support vectors, smooth boundary, underfits.
    Large C: narrow margin, fewer support vectors, wiggly boundary that
    chases individual points, overfits. Good slide for explaining the
    bias-variance tradeoff concretely.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_train_scaled)
    y_arr = np.asarray(y_train)

    C_values = [0.01, 0.1, 1, 100]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for ax, C in zip(axes.ravel(), C_values):
        model = SVC(kernel="rbf", C=C, random_state=42).fit(X_2d, y_train)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        for label, colour in zip([0, 1], ["#c44e52", "#4c72b0"]):
            mask = y_arr == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colour, s=18,
                       edgecolors="k", linewidths=0.3)
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=90, facecolors="none", edgecolors="black", linewidths=0.9)
        ax.set_title(f"C = {C}  |  {len(model.support_vectors_)} support vectors, "
                     f"train acc {model.score(X_2d, y_train):.3f}")

    fig.suptitle("Effect of the C Parameter on the Decision Boundary",
                 fontsize=14)
    fig.tight_layout()
    _save(fig, "C_parameter_effect.png")