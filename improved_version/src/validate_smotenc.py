"""
SMOTENC Validation Experiment
Accident Prediction System

Training: July 1–4
Validation: July 5
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from imblearn.over_sampling import SMOTENC

from train_model import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

from validate_thresholds import (
    load_validation_data,
    evaluate_thresholds,
)


def prepare_smotenc_data(X_train, y_train):

    X_train = X_train.copy()

    X_train[CATEGORICAL_FEATURES] = (
        X_train[CATEGORICAL_FEATURES].astype(object)
    )

    numerical_imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    categorical_imputer = SimpleImputer(
        strategy="most_frequent",
        keep_empty_features=True
    )

    X_train[NUMERICAL_FEATURES] = (
        numerical_imputer.fit_transform(
            X_train[NUMERICAL_FEATURES]
        )
    )

    X_train[CATEGORICAL_FEATURES] = (
        categorical_imputer.fit_transform(
            X_train[CATEGORICAL_FEATURES]
        )
    )

    categorical_indices = [
        X_train.columns.get_loc(column)
        for column in CATEGORICAL_FEATURES
    ]

    sampler = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,
        random_state=42,
        k_neighbors=5
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    print("\nRESAMPLED TRAINING DATA")
    print(y_resampled.value_counts())

    return (
        X_resampled,
        y_resampled,
        numerical_imputer,
        categorical_imputer
    )


def train_smotenc_model(
    X_resampled,
    y_resampled,
    numerical_imputer,
    categorical_imputer,
    X_validation
):

    X_validation = X_validation.copy()

    X_validation[CATEGORICAL_FEATURES] = (
        X_validation[CATEGORICAL_FEATURES].astype(object)
    )

    # Encode categorical training features
    encoder = OneHotEncoder(handle_unknown="ignore")

    X_resampled_categorical = encoder.fit_transform(
        X_resampled[CATEGORICAL_FEATURES]
    )

    # Apply the same encoding to validation features
    X_validation_categorical = pd.DataFrame(
        categorical_imputer.transform(
            X_validation[CATEGORICAL_FEATURES]
        ),
        columns=CATEGORICAL_FEATURES,
        index=X_validation.index
    )

    X_validation_encoded = encoder.transform(
        X_validation_categorical
    )

    # Prepare numerical features
    X_resampled_numerical = (
        X_resampled[NUMERICAL_FEATURES].to_numpy()
    )

    X_validation_numerical = (
        numerical_imputer.transform(
            X_validation[NUMERICAL_FEATURES]
        )
    )

    # Combine numerical and encoded categorical features
    X_resampled_final = np.hstack([
        X_resampled_numerical,
        X_resampled_categorical.toarray()
    ])

    X_validation_final = np.hstack([
        X_validation_numerical,
        X_validation_encoded.toarray()
    ])

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining SMOTENC Random Forest...")

    model.fit(X_resampled_final, y_resampled)

    y_prob = model.predict_proba(
        X_validation_final
    )[:, 1]

    return y_prob

def run_smotenc_validation():

    print("\nSMOTENC VALIDATION EXPERIMENT")
    print("=" * 50)

    X_train, X_validation, y_train, y_validation = (
        load_validation_data()
    )

    (
        X_resampled,
        y_resampled,
        numerical_imputer,
        categorical_imputer
    ) = prepare_smotenc_data(X_train, y_train)

    y_prob = train_smotenc_model(
        X_resampled,
        y_resampled,
        numerical_imputer,
        categorical_imputer,
        X_validation
    )

    print(
        "\nSMOTENC VALIDATION AVERAGE PRECISION:",
        f"{average_precision_score(y_validation, y_prob):.4f}"
    )

    print("\nSMOTENC VALIDATION THRESHOLD RESULTS")
    print("=" * 75)

    print(
        f"{'Threshold':<12}{'TP':<8}{'FP':<8}"
        f"{'FN':<8}{'Precision':<12}{'Recall':<10}{'F1':<10}"
    )

    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score
    )

    thresholds = [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50
    ]

    for threshold in thresholds:

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

    print("\nSMOTENC VALIDATION COMPLETED.")


if __name__ == "__main__":
    run_smotenc_validation()