import pandas as pd

from config import RAW_DATA_PATH, TARGET_COLUMN


def validate_dataset():
    """Validate the raw accident prediction dataset."""

    print("=" * 60)
    print("ACCIDENT PREDICTION - DATA VALIDATION")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Required columns
    required_columns = [
        "location_id",
        "timestamp",
        "vehicle_count_per_hr",
        "avg_speed_kmph",
        "accident_occurred",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    print("\n1. REQUIRED COLUMN CHECK")
    if missing_columns:
        print("FAILED - Missing columns:", missing_columns)
    else:
        print("PASSED - All required columns exist")

    # Missing values
    print("\n2. MISSING VALUE CHECK")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0].to_string())

    # Duplicate records
    print("\n3. DUPLICATE CHECK")
    print("Complete duplicate rows:", df.duplicated().sum())

    if {"location_id", "timestamp"}.issubset(df.columns):
        duplicate_keys = df.duplicated(
            subset=["location_id", "timestamp"]
        ).sum()
        print("Duplicate location-time records:", duplicate_keys)

    # Target validation
    print("\n4. TARGET VALIDATION")
    if TARGET_COLUMN in df.columns:
        print("Target values:")
        print(df[TARGET_COLUMN].value_counts(dropna=False).to_string())

        valid_targets = df[TARGET_COLUMN].isin([True, False])
        print("Invalid or missing target records:", (~valid_targets).sum())
    else:
        print("FAILED - Target column not found")

    # Timestamp validation
    print("\n5. TIMESTAMP VALIDATION")
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        print("Invalid timestamps:", timestamps.isna().sum())
        print("Earliest timestamp:", timestamps.min())
        print("Latest timestamp:", timestamps.max())

    print("\n" + "=" * 60)
    print("DATA VALIDATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    validate_dataset()
