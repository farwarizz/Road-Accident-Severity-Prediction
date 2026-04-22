import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def stratified_sample(df, target, sample_size, random_state):
    if sample_size >= len(df):
        return df

    return (
        df.groupby(target, group_keys=False)
        .apply(
            lambda x: x.sample(
                max(1, int(sample_size * len(x) / len(df))),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )


def preprocess_data(df, target_col, binarize=False):
    # -----------------------------
    # Target
    # -----------------------------
    y = df[target_col]

    if binarize:
        majority_class = y.mode()[0]
        y = y.apply(lambda x: 0 if x == majority_class else 1)

    # -----------------------------
    # Features
    # -----------------------------
    X = df.drop(columns=[target_col]).copy()

    # Identify column types
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.columns.difference(num_cols).tolist()

    # Force safe dtypes
    for col in cat_cols:
        X[col] = X[col].astype(str)

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # -----------------------------
    # Pipelines
    # -----------------------------
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )

    # -----------------------------
    # Transform
    # -----------------------------
    X_processed = preprocessor.fit_transform(X)

    # Store column info for prediction-time safety
    preprocessor.num_cols = num_cols
    preprocessor.cat_cols = cat_cols

    return X_processed, y, preprocessor
