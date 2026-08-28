"""
Streamlit UI for the trained breast cancer SVM.

Run with:  streamlit run app.py
Requires models/svm_model.joblib, written by `python main.py`.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.load_data import load_data
from src.persist import MODEL_PATH, load_model

st.set_page_config(page_title="Breast Cancer Diagnosis (SVM)",
                   page_icon="🔬", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_dataset():
    X, y, feature_names, target_names = load_data()
    return X, y, feature_names, target_names


def predict(artifact, values):
    """Scale one sample the same way training did, then classify it."""
    # Kept as a DataFrame with the original column names: the scaler was fit
    # on one, and passing a bare array makes scikit-learn warn about it.
    row = pd.DataFrame([np.asarray(values, dtype=float)],
                       columns=artifact["feature_names"])
    scaled = artifact["scaler"].transform(row)
    label = int(artifact["model"].predict(scaled)[0])
    # SVC has no predict_proba here, so the signed distance from the boundary
    # stands in for confidence: sign gives the class, magnitude gives how far
    # from the boundary the sample sits.
    margin = float(artifact["model"].decision_function(scaled)[0])
    return label, margin


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
try:
    artifact = get_model()
except FileNotFoundError:
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Train and save one first:\n\n```bash\npython main.py\n```"
    )
    st.stop()

feature_names = artifact["feature_names"]
target_names = artifact["target_names"]   # index 0 = malignant, 1 = benign
meta = artifact["metadata"]

X, y, _, _ = get_dataset()

# --------------------------------------------------------------------------
# Sidebar: what the served model actually is
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    st.caption(f"`{MODEL_PATH}`")
    if meta.get("best_params"):
        st.write("**Hyperparameters**")
        st.json(meta["best_params"], expanded=True)
    cols = st.columns(2)
    if "test_accuracy" in meta:
        cols[0].metric("Test accuracy", f"{meta['test_accuracy']:.3f}")
    if "test_recall" in meta:
        cols[1].metric("Recall", f"{meta['test_recall']:.3f}")
    if "roc_auc" in meta:
        cols[0].metric("ROC AUC", f"{meta['roc_auc']:.3f}")
    if "test_f1" in meta:
        cols[1].metric("F1", f"{meta['test_f1']:.3f}")
    st.caption(
        "Metrics are on the held-out test split. Precision/recall/F1 score the "
        f"*{target_names[1]}* class (label 1), following scikit-learn's encoding."
    )

st.title("🔬 Breast Cancer Diagnosis")
st.caption(
    "Support Vector Machine trained on the Breast Cancer Wisconsin (Diagnostic) "
    "dataset, using 30 cell-nucleus measurements per sample. "
    "**Coursework demonstration, not a medical device.**"
)

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
mode = st.radio(
    "Input",
    ["Pick a case from the dataset", "Enter measurements manually"],
    horizontal=True,
)

if mode == "Pick a case from the dataset":
    left, right = st.columns([3, 1])
    idx = left.number_input(
        "Sample index", min_value=0, max_value=len(X) - 1, value=0, step=1,
        help="Any row of the 569-sample dataset. Its true label is shown so you "
             "can check the prediction against it.",
    )
    if right.button("Random case", use_container_width=True):
        idx = int(np.random.randint(len(X)))
        st.session_state["idx"] = idx
    idx = int(st.session_state.get("idx", idx))
    values = X.iloc[idx].tolist()
    true_label = int(y.iloc[idx])
    with st.expander("Measurements for this sample"):
        st.dataframe(
            pd.DataFrame({"feature": feature_names, "value": values}),
            hide_index=True, use_container_width=True,
        )
else:
    true_label = None
    st.caption(
        "Defaults are the dataset median for each feature. Change the ones you "
        "care about and leave the rest."
    )
    values = []
    groups = [("Mean values", 0, 10), ("Standard error", 10, 20),
              ("Worst (largest) values", 20, 30)]
    for tab, (title, start, end) in zip(st.tabs([g[0] for g in groups]), groups):
        with tab:
            cols = st.columns(2)
            for i in range(start, end):
                name = feature_names[i]
                col = X.iloc[:, i]
                values.append(cols[i % 2].number_input(
                    name,
                    min_value=float(col.min()) * 0.5,
                    max_value=float(col.max()) * 1.5,
                    value=float(col.median()),
                    format="%.5f",
                    key=f"f_{i}",
                ))

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    label, margin = predict(artifact, values)
    name = target_names[label]

    left, right = st.columns([2, 1])
    with left:
        if label == 0:
            st.error(f"### Prediction: {name.upper()}")
        else:
            st.success(f"### Prediction: {name.upper()}")
        if true_label is not None:
            actual = target_names[true_label]
            if true_label == label:
                st.caption(f"✅ Actual diagnosis: **{actual}**. Correct.")
            else:
                st.caption(f"❌ Actual diagnosis: **{actual}**. Misclassified.")
    with right:
        st.metric("Distance from boundary", f"{margin:+.3f}")
        st.caption(
            f"Negative → {target_names[0]}, positive → {target_names[1]}. "
            "Values near 0 sit close to the decision boundary and are the "
            "least certain."
        )

    if abs(margin) < 0.5:
        st.warning(
            "This sample lies close to the decision boundary, so the model is "
            "not confident either way."
        )
    if label == 1:
        st.info(
            "A false negative (calling a malignant tumour benign) is the costly "
            "error in screening, which is why recall matters more than raw "
            "accuracy on this task."
        )
