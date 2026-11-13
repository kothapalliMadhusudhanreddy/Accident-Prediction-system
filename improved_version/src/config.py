from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

# Dataset
RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"

# Target column
TARGET_COLUMN = "accident_occurred"


# Output files
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
CLEANING_REPORT_PATH = REPORT_DIR / "cleaning_report.txt"
