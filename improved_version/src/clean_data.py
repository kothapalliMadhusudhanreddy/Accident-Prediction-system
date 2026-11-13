"""
Data cleaning pipeline for the Accident Prediction System.

The raw dataset is never modified.
All cleaning operations are performed on a separate DataFrame.
"""

import pandas as pd
import numpy as np

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    PROCESSED_DATA_PATH,
    CLEANING_REPORT_PATH,
    TARGET_COLUMN,
)
def load_raw_data():
    """
    Load the original dataset without modifying the source CSV.

    Returns:
        pd.DataFrame: A separate copy of the original dataset.
    """

    print("Loading the original dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    return df.copy()
def convert_timestamp(df):
    """
    Convert the timestamp column into pandas datetime format.
    Invalid timestamps will be identified as NaT.
    """

    print("\nConverting timestamp column...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_count = df["timestamp"].isna().sum()

    print(f"Invalid timestamps found: {invalid_count}")

    return df
def remove_invalid_timestamps(df):
    """
    Remove records where timestamp conversion failed.
    """

    print("\nChecking invalid timestamps...")

    before_rows = len(df)

    df = df.dropna(subset=["timestamp"]).copy()

    removed_rows = before_rows - len(df)

    print(f"Invalid timestamp rows removed: {removed_rows}")
    print(f"Remaining rows: {len(df)}")

    return df
def remove_duplicate_records(df):
    """
    Remove duplicate records based on location and timestamp.
    """

    print("\nChecking duplicate location-time records...")

    before_rows = len(df)

    df = df.drop_duplicates(
        subset=["location_id", "timestamp"],
        keep="first"
    ).copy()

    removed_rows = before_rows - len(df)

    print(f"Duplicate records removed: {removed_rows}")
    print(f"Remaining rows: {len(df)}")

    return df
def inspect_missing_values(df):
    """
    Inspect missing values in every dataset column.
    """

    print("\nInspecting missing values...")

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df)
    ) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    missing_report = missing_report[
        missing_report["missing_count"] > 0
    ].sort_values(
        by="missing_count",
        ascending=False
    )

    print("\nMissing value report:")
    print(missing_report)

    return missing_report
def inspect_target_distribution(df):
    """
    Inspect the distribution of accident and non-accident records.
    """

    print("\nInspecting accident target distribution...")

    target_counts = df["accident_occurred"].value_counts(
        dropna=False
    )

    target_percentages = df["accident_occurred"].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    target_report = pd.DataFrame({
        "record_count": target_counts,
        "percentage": target_percentages
    })

    print("\nTarget distribution:")
    print(target_report)

    print(f"\nTotal records: {len(df)}")

    return target_report
def inspect_traffic_quality(df):
    """
    Inspect traffic measurement quality flags.
    """

    print("\nInspecting traffic data quality flags...")

    if "traffic_data_quality_flag" not in df.columns:
        print("Traffic quality flag column not found.")
        return None

    quality_report = (
        df["traffic_data_quality_flag"]
        .value_counts(dropna=False)
    )

    print("\nTraffic quality flag distribution:")
    print(quality_report)

    return quality_report
def inspect_signal_quality(df):
    """
    Inspect traffic signal measurement quality flags.
    """

    print("\nInspecting signal data quality flags...")

    if "signal_data_quality_flag" not in df.columns:
        print("Signal quality flag column not found.")
        return None

    quality_report = (
        df["signal_data_quality_flag"]
        .value_counts(dropna=False)
    )

    print("\nSignal quality flag distribution:")
    print(quality_report)

    return quality_report
def verify_accident_records(df):
    """
    Verify that accident and non-accident records are preserved.
    """

    print("\nVerifying accident records...")

    total_records = len(df)

    accident_records = (
        df["accident_occurred"] == True
    ).sum()

    non_accident_records = (
        df["accident_occurred"] == False
    ).sum()

    print(f"Total records: {total_records}")
    print(f"Accident records: {accident_records}")
    print(f"Non-accident records: {non_accident_records}")

    if total_records == accident_records + non_accident_records:
        print("Verification passed: All target records are preserved.")
    else:
        print("WARNING: Target record counts do not match.")

    return {
        "total_records": total_records,
        "accident_records": accident_records,
        "non_accident_records": non_accident_records
    }
def inspect_vehicle_measurements(df):
    """
    Inspect vehicle count and average speed measurements.
    """

    print("\nInspecting vehicle measurements...")

    columns = [
        "vehicle_count_per_hr",
        "avg_speed_kmph"
    ]

    print("\nVehicle measurement summary:")
    print(df[columns].describe())

    print("\nMissing measurement counts:")
    print(df[columns].isnull().sum())

    print("\nNegative average speeds:")
    print(
        df[df["avg_speed_kmph"] < 0][
            ["location_id", "timestamp", "avg_speed_kmph"]
        ]
    )

    print("\nZero vehicle counts:")
    print(
        df[df["vehicle_count_per_hr"] == 0][
            ["location_id", "timestamp", "vehicle_count_per_hr"]
        ]
    )

    return df[columns].describe()
def clean_traffic_measurements(df):
    """
    Mark invalid traffic measurements as missing.
    The original raw dataset remains unchanged.
    """

    print("\nCleaning traffic measurements...")

    df = df.copy()

    # Identify sensor dropout records
    dropout_mask = (
        df["traffic_data_quality_flag"] == "veh_dropout"
    )

    # Mark dropout measurements as missing
    df.loc[
        dropout_mask,
        ["vehicle_count_per_hr", "avg_speed_kmph"]
    ] = np.nan

    # Identify invalid average speeds
    invalid_speed_mask = (
        (df["traffic_data_quality_flag"] == "speed_glitch") |
        (df["avg_speed_kmph"] < 0)
    )

    # Mark invalid speeds as missing
    df.loc[
        invalid_speed_mask,
        "avg_speed_kmph"
    ] = np.nan

    print(
        f"Dropout records identified: {dropout_mask.sum()}"
    )

    print(
        f"Invalid speed measurements identified: "
        f"{invalid_speed_mask.sum()}"
    )

    print("\nMissing measurements after cleaning:")

    print(
        df[
            ["vehicle_count_per_hr", "avg_speed_kmph"]
        ].isnull().sum()
    )

    return df

def clean_signal_measurements(df):
    """
    Mark suspicious traffic signal measurements as missing.
    Preserve all rows and accident labels.
    """

    print("\nCleaning signal measurements...")

    df = df.copy()

    # Identify signal quality problems
    glitch_mask = (
        df["signal_data_quality_flag"] == "sig_glitch"
    )

    # Identify suspicious zero green durations
    invalid_green_mask = (
        glitch_mask &
        (df["signal_status"] == "Working") &
        (df["green_duration_s"] == 0)
    )

    # Identify suspicious yellow durations
    invalid_yellow_mask = (
        glitch_mask &
        (df["yellow_duration_s"] == 65)
    )

    # Mark suspicious measurements as missing
    df.loc[
        invalid_green_mask,
        "green_duration_s"
    ] = np.nan

    df.loc[
        invalid_yellow_mask,
        "yellow_duration_s"
    ] = np.nan

    print(
        f"Invalid green measurements: "
        f"{invalid_green_mask.sum()}"
    )

    print(
        f"Invalid yellow measurements: "
        f"{invalid_yellow_mask.sum()}"
    )

    print("\nMissing signal measurements after cleaning:")

    columns = [
        "green_duration_s",
        "red_duration_s",
        "yellow_duration_s",
        "cycle_time_s",
        "violations_count"
    ]

    print(df[columns].isnull().sum())

    return df
def save_processed_data(df):
    """
    Save the cleaned dataset without modifying the raw dataset.
    """

    print("\nSaving processed dataset...")

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(f"Processed dataset saved at: {PROCESSED_DATA_PATH}")

    return PROCESSED_DATA_PATH
def generate_cleaning_report(df, original_shape):
    """
    Generate a text report describing the cleaning results.
    """

    print("\nGenerating cleaning report...")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    target_counts = df[TARGET_COLUMN].value_counts(
        dropna=False
    )

    missing_values = df.isnull().sum()

    report = []

    report.append("ACCIDENT PREDICTION SYSTEM")
    report.append("DATA CLEANING REPORT")
    report.append("=" * 50)

    report.append("\nDATASET OVERVIEW")
    report.append(f"Original shape: {original_shape}")
    report.append(f"Cleaned shape: {df.shape}")

    report.append("\nTARGET DISTRIBUTION")
    report.append(f"Non-accident records: {target_counts.get(False, 0)}")
    report.append(f"Accident records: {target_counts.get(True, 0)}")

    report.append("\nFINAL MISSING VALUES")

    for column, count in missing_values.items():
        if count > 0:
            report.append(f"{column}: {count}")

    report.append("\nDATA PRESERVATION")
    report.append("Original raw dataset was not modified.")
    report.append("Accident and non-accident records were preserved.")

    report.append("\nNOTE")
    report.append(
        "Missing values have not been imputed. "
        "Model-specific imputation will be performed later "
        "using training data only."
    )

    report_text = "\n".join(report)

    CLEANING_REPORT_PATH.write_text(
        report_text,
        encoding="utf-8"
    )

    print(f"Cleaning report saved at: {CLEANING_REPORT_PATH}")

    return CLEANING_REPORT_PATH
def run_cleaning_pipeline():
    """
    Execute the complete data cleaning pipeline.
    """

    print("\n" + "=" * 60)
    print("STARTING ACCIDENT DATA CLEANING PIPELINE")
    print("=" * 60)

    # Step 1: Load original dataset
    raw_df = load_raw_data()

    original_shape = raw_df.shape

    # Step 2: Create a separate working copy
    df = raw_df.copy()

    # Step 3: Convert timestamps
    df = convert_timestamp(df)

    # Step 4: Remove invalid timestamps
    df = remove_invalid_timestamps(df)

    # Step 5: Remove duplicate records
    df = remove_duplicate_records(df)

    # Step 6: Clean traffic measurements
    df = clean_traffic_measurements(df)

    # Step 7: Clean signal measurements
    df = clean_signal_measurements(df)

    # Step 8: Verify accident records
    verification = verify_accident_records(df)

    # Step 9: Save cleaned dataset
    save_processed_data(df)

    # Step 10: Generate cleaning report
    generate_cleaning_report(df, original_shape)

    print("\n" + "=" * 60)
    print("DATA CLEANING PIPELINE COMPLETED")
    print("=" * 60)

    return df
if __name__ == "__main__":
    run_cleaning_pipeline()