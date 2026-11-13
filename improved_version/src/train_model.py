"""
Machine Learning Training Pipeline
Accident Prediction System

This module handles:
1. Dataset loading
2. Feature selection
3. Train-test separation
4. Preprocessing
5. Model training
6. Model evaluation
"""

from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "cleaned_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

# Target
TARGET_COLUMN = "accident_occurred"

# Selected features
NUMERICAL_FEATURES = [
    "day_of_week",
    "hour_of_day",
    "vehicle_count_per_hr",
    "avg_speed_kmph",
    "lane_count",
    "speed_limit_kmph",
]

CATEGORICAL_FEATURES = [
    "is_peak",
    "weather",
    "lighting",
    "road_type",
    "has_signal",
    "enforcement_level",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Chronological split
TEST_START_DATE = "2025-07-06"

TIMESTAMP_COLUMN = "timestamp"


def load_and_split_data():
    """Load the cleaned dataset and split it chronologically."""

    print("\nLoading cleaned dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Check required columns
    required_columns = FEATURES + [
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert timestamp
    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce"
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Invalid timestamps found in the cleaned dataset."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Missing target values found."
        )

    # Chronological split
    test_start = pd.Timestamp(TEST_START_DATE)

    train_df = df[df[TIMESTAMP_COLUMN] < test_start].copy()
    test_df = df[df[TIMESTAMP_COLUMN] >= test_start].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURES].copy()
    y_test = test_df[TARGET_COLUMN].astype(int)

    # Convert categorical columns to object format
    # for compatibility with scikit-learn imputation.
    X_train[CATEGORICAL_FEATURES] = (
        X_train[CATEGORICAL_FEATURES].astype(object)
    )

    X_test[CATEGORICAL_FEATURES] = (
        X_test[CATEGORICAL_FEATURES].astype(object)
    )

    print("\nTRAINING DATA")
    print("Records:", len(X_train))
    print("Accidents:", int(y_train.sum()))
    print("Non-accidents:", int((y_train == 0).sum()))
    print("Start:", train_df[TIMESTAMP_COLUMN].min())
    print("End:", train_df[TIMESTAMP_COLUMN].max())

    print("\nTESTING DATA")
    print("Records:", len(X_test))
    print("Accidents:", int(y_test.sum()))
    print("Non-accidents:", int((y_test == 0).sum()))
    print("Start:", test_df[TIMESTAMP_COLUMN].min())
    print("End:", test_df[TIMESTAMP_COLUMN].max())

    print("\nChronological split completed.")

    return X_train, X_test, y_train, y_test

def build_preprocessor():
    """Build preprocessing for numerical and categorical features."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    return preprocessor

def build_model_pipeline():
    """Combine preprocessing and Random Forest classification."""

    preprocessor = build_preprocessor()

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
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return model_pipeline

def train_model(X_train, y_train):
    """Train the Random Forest model using training data only."""

    print("\nBuilding model pipeline...")

    model_pipeline = build_model_pipeline()

    print("Starting model training...")

    model_pipeline.fit(X_train, y_train)

    print("\nModel training completed.")

    print("Training records:", len(X_train))
    print("Features used:", len(FEATURES))

    return model_pipeline

def evaluate_model(model_pipeline, X_test, y_test):
    """Evaluate the trained model on the chronological test period.

    Note: This period was examined in earlier experiments and
    is not an untouched holdout dataset."""
    print("\nEvaluating model on test data...")

    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    avg_precision = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nMODEL EVALUATION RESULTS")
    print("------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")

    print("\nCONFUSION MATRIX")
    print(cm)

    print("\nDETAILED RESULTS")
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    print("\nFALSE ALARMS:", fp)
    print("MISSED ACCIDENTS:", fn)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Accident", "Accident"],
        zero_division=0,
    ))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_precision": avg_precision,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

def save_model(model_pipeline):
    """Save the complete trained model pipeline."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "accident_prediction_pipeline.joblib"

    joblib.dump(model_pipeline, model_path)

    print("\nMODEL SAVED SUCCESSFULLY")
    print("Model path:", model_path)

    return model_path

def run_training_pipeline():
    """Execute the complete model training and evaluation workflow."""

    # Step 1: Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Step 2: Train model
    model_pipeline = train_model(X_train, y_train)

    # Step 3: Evaluate model
    evaluation_results = evaluate_model(
        model_pipeline,
        X_test,
        y_test
    )

    # Step 4: Save model
    model_path = save_model(model_pipeline)

    print("\nTRAINING PIPELINE COMPLETED")
    print("Saved model:", model_path)

    return model_pipeline, evaluation_results
if __name__ == "__main__":
    run_training_pipeline()