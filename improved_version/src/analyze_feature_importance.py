"""
Feature Importance and Leakage Investigation
Accident Prediction System
"""

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.inspection import permutation_importance

from train_model import (
    load_and_split_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def analyze_feature_importance():

    print("\nFEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Load baseline model
    project_root = Path(__file__).resolve().parents[2]

    model_path = (
        project_root
        / "improved_version"
        / "models"
        / "accident_prediction_pipeline.joblib"
    )

    print("\nLoading trained baseline model...")

    model = joblib.load(model_path)

    # Load test data
    X_train, X_test, y_train, y_test = load_and_split_data()

    print("\nCalculating Random Forest feature importance...")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTOP 20 FEATURE IMPORTANCE")
    print(importance_df.head(20).to_string(index=False))

    # Permutation importance on original test data
    print("\nCalculating permutation importance...")
    print("This may take a little time...")

    permutation_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": permutation_result.importances_mean,
        "importance_std": permutation_result.importances_std
    })

    permutation_df = permutation_df.sort_values(
        by="importance_mean",
        ascending=False
    )

    print("\nPERMUTATION IMPORTANCE")
    print(permutation_df.to_string(index=False))

    # Save reports
    report_dir = (
        project_root
        / "improved_version"
        / "reports"
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(
        report_dir / "random_forest_feature_importance.csv",
        index=False
    )

    permutation_df.to_csv(
        report_dir / "permutation_feature_importance.csv",
        index=False
    )

    print("\nReports saved in:")
    print(report_dir)

    print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED.")


if __name__ == "__main__":
    analyze_feature_importance()