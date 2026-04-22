import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.data_loader import load_data, detect_target_column
from src.preprocessing import preprocess_data, stratified_sample
from src.eda import run_eda
from src.models import train_model, evaluate_model, predict_with_model
import os

st.set_page_config(page_title="Road Accident Severity Prediction", layout="wide")

st.title("🚦 Road Accident Severity Prediction (2019)")
st.markdown(
    """
This app trains **machine learning classifiers** to predict **accident severity**
using the Kaggle *Road Safety Data – Accidents 2019* dataset.

✔ Defensive column detection  
✔ Sampling for speed  
✔ Optional binary severity  
✔ Imbalance handling  
✔ Interactive EDA & maps  
✔ Streamlit-first UX
"""
)

# -----------------------------
# Sidebar – Global Controls
# -----------------------------
st.sidebar.header("⚙️ Global Settings")

RANDOM_STATE = st.sidebar.number_input("Random State", value=42, min_value=0, step=1)
TEST_SIZE = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
USE_SMOTE = st.sidebar.checkbox("Use SMOTE (training only)", value=False)
CLASS_WEIGHT = st.sidebar.checkbox("Use class_weight='balanced'", value=True)

# -----------------------------
# Data Loading
# -----------------------------
st.header("📂 Data Input")

uploaded_file = st.file_uploader(
    "Upload CSV (primary)",
    type=["csv"],
    key="dataset_uploader"
)

if uploaded_file:
    df = load_data(uploaded_file)
elif os.path.exists("Road Safety Data - Accidents 2019.csv"):
    df = load_data("Road Safety Data - Accidents 2019.csv")
    st.success("Loaded local dataset automatically.")
else:
    st.error("No dataset found. Please upload a CSV or place the dataset in the project root.")
    st.stop()

# -----------------------------
# Target Detection
# -----------------------------
st.header("🎯 Target Selection")

possible_targets = detect_target_column(df)

if len(possible_targets) == 0:
    st.error("No suitable target column detected.")
    st.stop()
elif len(possible_targets) == 1:
    target_col = possible_targets[0]
    st.success(f"Auto-detected target column: `{target_col}`")
else:
    target_col = st.selectbox(
        "Multiple possible targets detected. Choose one:",
        possible_targets
    )

BINARIZE = st.checkbox("Binarize target (Severe vs Slight)", value=False)

# -----------------------------
# Sampling
# -----------------------------
st.header("📉 Sampling")

use_sampling = st.checkbox("Use stratified sampling", value=True)
sample_size = st.slider("Sample size", 5000, 50000, 25000, step=5000)

if use_sampling:
    df = stratified_sample(
        df,
        target_col,
        sample_size=sample_size,
        random_state=RANDOM_STATE,
    )
    st.info(f"Using stratified sample: {df.shape}")

# -----------------------------
# EDA
# -----------------------------
st.header("📊 Exploratory Data Analysis")
run_eda(df, target_col)

# -----------------------------
# Preprocessing
# -----------------------------
st.header("🧹 Preprocessing")

X, y, preprocessor = preprocess_data(
    df,
    target_col=target_col,
    binarize=BINARIZE,
)

st.success("Preprocessing completed.")

# -----------------------------
# Model Training
# -----------------------------
st.header("🤖 Model Training")

model_choice = st.selectbox(
    "Choose model",
    ["Logistic Regression", "Random Forest", "XGBoost (if available)"],
)

if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        model, X_train, X_test, y_train, y_test, label_encoder = train_model(
            X,
            y,
            model_choice=model_choice,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            class_weight=CLASS_WEIGHT,
            use_smote=USE_SMOTE,
        )

    st.session_state["model"] = model
    st.session_state["preprocessor"] = preprocessor
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["label_encoder"] = label_encoder

    joblib.dump(model, "trained_model.joblib")
    st.success("Model trained and saved as trained_model.joblib")

# -----------------------------
# Evaluation
# -----------------------------
if "model" in st.session_state:
    st.header("📈 Evaluation")

    if st.button("📊 Evaluate Model"):
        evaluate_model(
            st.session_state["model"],
            st.session_state["X_test"],
            st.session_state["y_test"],
            st.session_state.get("label_encoder"),
        )

# -----------------------------
# Prediction
# -----------------------------
if "model" in st.session_state:
    st.header("🔮 Prediction")

    st.subheader("Batch Prediction (CSV)")
    pred_file = st.file_uploader(
        "Upload CSV for prediction",
        type=["csv"],
        key="pred"
    )

    if pred_file:
        pred_df = pd.read_csv(pred_file)
        preds = predict_with_model(
            st.session_state["model"],
            st.session_state["preprocessor"],
            pred_df,
            st.session_state.get("label_encoder"),
        )
        pred_df["prediction"] = preds
        st.dataframe(pred_df.head())

        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )

st.markdown("---")
st.markdown("✅ App ready. Designed for **robust local execution**.")
