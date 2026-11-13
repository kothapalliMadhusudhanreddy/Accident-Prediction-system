
"""
Threshold Validation Experiment
Accident Prediction System

Purpose:
1. Train a baseline Random Forest on July 1–4.
2. Evaluate probability thresholds on July 5.
3. Keep July 6–7 untouched for final testing.
"""

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

from train_model import (
    DATA_PATH,
    FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    build_preprocessor,
)


TRAIN_END_DATE = "2025-07-05"
VALIDATION_START_DATE = "2025-07-05"
VALIDATION_END_DATE = "2025-07-06"



def load_validation_data():
    """Load data and create training and validation datasets."""

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(DATA_PATH)

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce"
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Invalid timestamps found.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Missing target values found.")

    train_end = pd.Timestamp(TRAIN_END_DATE)
    validation_start = pd.Timestamp(VALIDATION_START_DATE)
    validation_end = pd.Timestamp(VALIDATION_END_DATE)

    train_df = df[
        df[TIMESTAMP_COLUMN] < train_end
    ].copy()

    validation_df = df[
        (df[TIMESTAMP_COLUMN] >= validation_start)
        & (df[TIMESTAMP_COLUMN] < validation_end)
    ].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError("Training or validation data is empty.")

    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_validation = validation_df[FEATURES].copy()
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    X_train[CATEGORICAL_FEATURES] = (
        X_train[CATEGORICAL_FEATURES].astype(object)
    )

    X_validation[CATEGORICAL_FEATURES] = (
        X_validation[CATEGORICAL_FEATURES].astype(object)
    )

    print("\nTRAINING DATA")
    print("Records:", len(X_train))
    print("Accidents:", int(y_train.sum()))
    print("Non-accidents:", int((y_train == 0).sum()))

    print("\nVALIDATION DATA")
    print("Records:", len(X_validation))
    print("Accidents:", int(y_validation.sum()))
    print("Non-accidents:", int((y_validation == 0).sum()))

    return X_train, X_validation, y_train, y_validation



def train_validation_model(X_train, y_train):
    """Train Random Forest using training data only."""

    print("\nBuilding Random Forest pipeline...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )

    print("\nTraining model on July 1–4 data...")

    model_pipeline.fit(X_train, y_train)

    print("Model training completed.")
    print("Training records:", len(X_train))

    return model_pipeline


def evaluate_thresholds(model_pipeline, X_validation, y_validation):
    """Compare prediction thresholds using validation data."""

    print("\nGenerating validation probabilities...")

    y_prob = model_pipeline.predict_proba(
        X_validation
    )[:, 1]

    print("\nVALIDATION THRESHOLD RESULTS")
    print("=" * 75)

    print(
        f"{'Threshold':<12}"
        f"{'TP':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'F1':<10}"
    )

    thresholds = [
        0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50
    ]

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        print(
            f"{threshold:<12.2f}"
            f"{tp:<8}"
            f"{fp:<8}"
            f"{fn:<8}"
            f"{precision:<12.4f}"
            f"{recall:<10.4f}"
            f"{f1:<10.4f}"
        )

    print("\nVALIDATION AVERAGE PRECISION:",
          f"{average_precision_score(y_validation, y_prob):.4f}")



def run_validation_experiment():
    """Run the complete threshold validation experiment."""

    # Step 1: Load chronological training and validation data
    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = load_validation_data()

    # Step 2: Train model on July 1–4
    model_pipeline = train_validation_model(
        X_train,
        y_train
    )

    # Step 3: Evaluate thresholds on July 5
    evaluate_thresholds(
        model_pipeline,
        X_validation,
        y_validation
    )

    print("\nVALIDATION EXPERIMENT COMPLETED.")


if __name__ == "__main__":
    run_validation_experiment()
