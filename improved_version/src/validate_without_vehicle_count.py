"""
Validation Experiment Without Vehicle Count
Accident Prediction System
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from train_model import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    build_preprocessor,
)

from validate_thresholds import load_validation_data


def run_experiment():

    print("\nVALIDATION WITHOUT VEHICLE COUNT")
    print("=" * 60)

    X_train, X_validation, y_train, y_validation = (
        load_validation_data()
    )

    # Remove potentially suspicious feature
    X_train = X_train.drop(
        columns=["vehicle_count_per_hr"]
    )

    X_validation = X_validation.drop(
        columns=["vehicle_count_per_hr"]
    )

    numerical_features = [
        feature for feature in NUMERICAL_FEATURES
        if feature != "vehicle_count_per_hr"
    ]

    categorical_features = CATEGORICAL_FEATURES.copy()

    print("\nREMAINING FEATURES:")
    print(X_train.columns.tolist())

    # Build preprocessing pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    print("\nTraining model without vehicle_count_per_hr...")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_validation)[:, 1]

    print("\nAVERAGE PRECISION:")
    print(f"{average_precision_score(y_validation, y_prob):.4f}")

    print("\nTHRESHOLD RESULTS")
    print("=" * 75)

    print(
        f"{'Threshold':<12}{'TP':<8}{'FP':<8}"
        f"{'FN':<8}{'Precision':<12}{'Recall':<10}{'F1':<10}"
    )

    for threshold in [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50
    ]:

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_validation, y_pred, zero_division=0
        )

        recall = recall_score(
            y_validation, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_validation, y_pred, zero_division=0
        )

        print(
            f"{threshold:<12.2f}{tp:<8}{fp:<8}{fn:<8}"
            f"{precision:<12.4f}{recall:<10.4f}{f1:<10.4f}"
        )

    print("\nEXPERIMENT COMPLETED.")


if __name__ == "__main__":
    run_experiment()