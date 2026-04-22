import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def train_model(
    X,
    y,
    model_choice,
    test_size,
    random_state,
    class_weight,
    use_smote,
):
    # -----------------------------
    # Train / Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # -----------------------------
    # SMOTE (training only)
    # -----------------------------
    if use_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    label_encoder = None  # used only for XGBoost

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    if model_choice == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced" if class_weight else None,
        )

    # -----------------------------
    # Random Forest
    # -----------------------------
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced" if class_weight else None,
            random_state=random_state,
        )

    # -----------------------------
    # XGBoost (needs labels starting at 0)
    # -----------------------------
    else:
        if not XGB_AVAILABLE:
            st.warning("XGBoost not installed. Skipping.")
            raise RuntimeError("XGBoost unavailable")

        # 🔒 FIX: encode labels to 0,1,2,...
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            random_state=random_state,
            eval_metric="mlogloss",
        )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, label_encoder


def evaluate_model(model, X_test, y_test, label_encoder=None):
    preds = model.predict(X_test)

    # Decode predictions back to original labels if needed
    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)
        y_test = label_encoder.inverse_transform(y_test)

    st.text("Classification Report")
    st.text(classification_report(y_test, preds))

    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, preds))


def predict_with_model(model, preprocessor, df, label_encoder=None):
    X = df.copy()

    # 🔒 Force SAME dtypes as training
    for col in preprocessor.cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    for col in preprocessor.num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_transformed = preprocessor.transform(X)
    preds = model.predict(X_transformed)

    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)

    return preds

