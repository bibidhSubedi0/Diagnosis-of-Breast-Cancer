"""
SVM Classification on the Breast Cancer Wisconsin (Diagnostic) Dataset
AI Class Project

Run with:  python main.py
"""

from src.load_data import load_data, describe_data
from src.preprocess import split_data, scale_features
from src.train import (
    train_all_kernels,
    tune_hyperparameters,
    cross_validate_model,
    compare_scaling_effect,
)
from src.evaluate import evaluate_model, evaluate_with_probabilities, interpret_errors
from src.persist import save_model
from src.visualize import (
    plot_class_distribution,
    plot_feature_correlation,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_kernel_comparison,
    plot_scaling_effect,
    plot_decision_boundary,
    plot_C_effect,
)


def main():
    # 1. Load
    X, y, feature_names, target_names = load_data()
    describe_data(X, y, target_names)

    # 2. Split and scale
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 3. Show why scaling is necessary
    acc_unscaled, acc_scaled = compare_scaling_effect(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    )

    # 4. Compare kernels
    kernel_results = train_all_kernels(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # 5. Tune the best model
    best_model, grid = tune_hyperparameters(X_train_scaled, y_train)
    cross_validate_model(best_model, X_train_scaled, y_train)

    # 6. Evaluate on the held-out test set
    metrics = evaluate_model(best_model, X_test_scaled, y_test, target_names)
    roc_data = evaluate_with_probabilities(best_model, X_test_scaled, y_test)
    interpret_errors(metrics, target_names)

    # 7. Persist the tuned model so the Streamlit app can serve it
    save_model(
        best_model,
        scaler,
        feature_names,
        target_names,
        metadata={
            "best_params": grid.best_params_,
            "cv_accuracy": grid.best_score_,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "roc_auc": roc_data["auc"],
            "n_train": len(y_train),
            "n_test": len(y_test),
        },
    )

    # 8. Figures
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    plot_class_distribution(y, target_names)
    plot_feature_correlation(X)
    plot_scaling_effect(acc_unscaled, acc_scaled)
    plot_kernel_comparison(kernel_results)
    plot_confusion_matrix(metrics["confusion_matrix"], target_names)
    plot_roc_curve(roc_data)
    plot_decision_boundary(X_train_scaled, y_train, target_names, kernel="rbf")
    plot_decision_boundary(X_train_scaled, y_train, target_names, kernel="linear")
    plot_C_effect(X_train_scaled, y_train, target_names)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Final test accuracy : {metrics['accuracy']:.4f}")
    print(f"Best parameters     : {grid.best_params_}")
    print("Model written to    : models/svm_model.joblib")
    print("Figures written to  : results/figures/")


if __name__ == "__main__":
    main()