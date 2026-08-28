"""Save and load the trained model so it can be served without retraining."""

import os

import joblib

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")


def save_model(model, scaler, feature_names, target_names, metadata=None,
               path=MODEL_PATH):
    """
    Write the model and everything needed to reproduce its input pipeline.

    The scaler has to travel with the model. It was fit on the training split
    only, and an SVM scored on unscaled input is meaningless -- serving the
    model without its scaler is the classic way to get silently wrong
    predictions.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "metadata": metadata or {},
    }
    joblib.dump(artifact, path)
    print(f"Model saved to      : {path}")
    return path


def load_model(path=MODEL_PATH):
    """Load the artifact written by save_model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at '{path}'. Run `python main.py` first."
        )
    return joblib.load(path)
