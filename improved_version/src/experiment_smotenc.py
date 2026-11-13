
"""
SMOTENC Experiment
Accident Prediction System

Tests whether SMOTENC improves accident detection
compared with the baseline model.
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

from imblearn.over_sampling import SMOTENC

from train_model import (
    load_and_split_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def prepare_training_data(X_train, y_train):
    """Impute training data and apply SMOTENC."""

    X_train = X_train.copy()
    y_train = y_train.copy()

    X_train[CATEGORICAL_FEATURES] = (
        X_train[CATEGORICAL_FEATURES].astype(object)
    )

    numerical_imputer = SimpleImputer(strategy="median")
    X_train[NUMERICAL_FEATURES] = numerical_imputer.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    X_train[CATEGORICAL_FEATURES] = categorical_imputer.fit_transform(
        X_train[CATEGORICAL_FEATURES]
    )

    categorical_indices = [
        X_train.columns.get_loc(column)
        for column in CATEGORICAL_FEATURES
    ]

    sampler = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=0.5,
        random_state=42,
        k_neighbors=5,
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train,
    )

    return (
        X_resampled,
        y_resampled,
        numerical_imputer,
        categorical_imputer,
    )


def train_and_evaluate_smotenc():

    print("\nLoading training and testing data...")

    X_train, X_test, y_train, y_test = load_and_split_data()

    print("\nOriginal training class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nPreparing data with SMOTENC...")

    (
        X_resampled,
        y_resampled,
        numerical_imputer,
        categorical_imputer,
    ) = prepare_training_data(X_train, y_train)

    print("\nResampled training class distribution:")
    print(y_resampled.value_counts().sort_index())

    # Fit encoder using resampled training data only
    encoder = OneHotEncoder(handle_unknown="ignore")

    X_resampled_encoded = encoder.fit_transform(
        X_resampled[CATEGORICAL_FEATURES]
    )

    # Transform test categorical features
    X_test_categorical = pd.DataFrame(
    categorical_imputer.transform(
        X_test[CATEGORICAL_FEATURES].astype(object)
    ),
    columns=CATEGORICAL_FEATURES,
    index=X_test.index,
    )

    X_test_encoded = encoder.transform(X_test_categorical)

    # Impute test numerical features using training imputer
    X_test_numerical = numerical_imputer.transform(
        X_test[NUMERICAL_FEATURES]
    )

    # Combine numerical and encoded categorical features
    X_resampled_final = np.hstack([
        X_resampled[NUMERICAL_FEATURES].to_numpy(),
        X_resampled_encoded.toarray(),
    ])

    X_test_final = np.hstack([
        X_test_numerical,
        X_test_encoded.toarray(),
    ])

    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_resampled_final, y_resampled)

    print("\nEvaluating on original test data...")

    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]

    print("\nSMOTENC MODEL RESULTS")
    print("---------------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall: "
        f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
    )
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(
        f"Average Precision: "
        f"{average_precision_score(y_test, y_prob):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nPrediction counts:")
    print(pd.Series(y_pred).value_counts().sort_index())


if __name__ == "__main__":
    train_and_evaluate_smotenc()
