"""Model evaluation metrics."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, target_names):
    """Compute and print the full set of classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 score  : {metrics['f1']:.4f}")

    print("\nConfusion matrix:")
    print(f"{'':>14}{'Pred ' + target_names[0]:>16}{'Pred ' + target_names[1]:>16}")
    cm = metrics["confusion_matrix"]
    for i, name in enumerate(target_names):
        print(f"{'True ' + name:>14}{cm[i][0]:>16}{cm[i][1]:>16}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return metrics


def evaluate_with_probabilities(model, X_test, y_test):
    """
    ROC AUC and curve points.

    SVC does not give probabilities by default, so this uses the signed
    distance from the decision boundary instead. That works fine for ROC,
    which only needs a ranking of confidence, not calibrated probabilities.
    """
    y_scores = model.decision_function(X_test)
    auc = roc_auc_score(y_test, y_scores)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    print(f"ROC AUC: {auc:.4f}\n")
    return {"auc": auc, "fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def interpret_errors(metrics, target_names):
    """
    Spell out what the confusion matrix means in domain terms.

    Worth including in the report: for a cancer screening task, a false
    negative (calling a malignant tumour benign) is far more costly than a
    false positive. Accuracy alone hides that asymmetry, which is exactly
    why recall matters here.
    """
    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    print(f"Correctly identified {target_names[0]} : {tn}")
    print(f"Correctly identified {target_names[1]} : {tp}")
    print(f"False positives (said {target_names[1]}, was {target_names[0]}): {fp}")
    print(f"False negatives (said {target_names[0]}, was {target_names[1]}): {fn}")
    print()