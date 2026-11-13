# Accident Prediction System

A machine learning project that explores accident occurrence classification using traffic, road, weather, lighting, and signal-related data.

The current experimental implementation is located in **`improved_version/`**.

## Project Overview

Road accidents can be influenced by several factors, including traffic volume, vehicle speed, road conditions, weather, lighting, and traffic signal conditions.

This project explores whether historical traffic and road-condition data can help classify observations as accident or non-accident cases.

**Important:** This is an experimental historical classification project. It has not been validated as a real-time accident early-warning system.

## Dataset

- Total records: 2,016
- Features: 33 columns in the original dataset
- Locations: 12
- Observation period: July 1–7, 2025
- Accident-labelled records: 93
- Non-accident records: 1,923

The dataset is imbalanced, with approximately 4.61% positive accident labels.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib

## Project Workflow

1. Dataset validation and quality checks
2. Exploratory data analysis and data auditing
3. Data cleaning and preprocessing
4. Feature selection
5. Chronological training and testing split
6. Random Forest model training
7. Model evaluation and threshold experiments
8. Feature importance analysis

## Machine Learning Approach

The current baseline uses `RandomForestClassifier` with:

- 300 decision trees
- Maximum depth of 8
- Balanced class weights
- Fixed random state for reproducibility

### Preprocessing

- Numerical features: median imputation
- Categorical features: most-frequent imputation
- Categorical encoding: One-Hot Encoding
- Preprocessing and classifier combined into a Scikit-learn Pipeline

Accident-specific fields such as severity, cause, and vehicles involved were excluded from the model features to reduce data leakage risk.

## Model Evaluation

The chronological test split contains records from July 6–7, 2025.

| Metric | Baseline result |
|---|---:|
| Accuracy | 90.62% |
| Precision | 2.94% |
| Recall | 4.55% |
| F1-score | 3.57% |
| Average Precision | 0.0692 |

**Interpretation:** Although accuracy is relatively high, the model detects very few accident-labelled observations at the default classification threshold. The results show that accuracy alone is not an adequate measure for this imbalanced classification problem.

The test period was examined in earlier experiments, so it should not be considered a completely untouched final holdout.

## Experiments

The project also includes experiments involving:

- SMOTENC class imbalance handling
- Classification threshold validation
- Feature importance analysis
- Model evaluation without vehicle count

These experiments are documented in the `reports/` directory.

## Project Structure

```text
improved_version/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── cleaned_dataset.csv
│
├── models/
│   └── accident_prediction_pipeline.joblib
│
├── reports/
│   ├── cleaning_log.txt
│   ├── cleaning_report.txt
│   ├── dataset_validation.txt
│   ├── model_evaluation_report.txt
│   ├── permutation_feature_importance.csv
│   └── random_forest_feature_importance.csv
│
├── src/
│   ├── clean_data.py
│   ├── config.py
│   ├── train_model.py
│   ├── validate_data.py
│   ├── validate_thresholds.py
│   ├── experiment_smotenc.py
│   ├── validate_smotenc.py
│   ├── analyze_feature_importance.py
│   └── validate_without_vehicle_count.py
│
└── requirements.txt
```

The repository also retains files from an earlier implementation at the root level. The `improved_version/` directory contains the current experimental workflow.

## Installation

Python 3.11 is recommended.

Clone the repository:

```bash
git clone https://github.com/kothapalliMadhusudhanreddy/Accident-Prediction-system.git
cd Accident-Prediction-system
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r .\improved_version\requirements.txt
```

## Run the Project

Run the data cleaning pipeline:

```powershell
python .\improved_version\src\clean_data.py
```

This reads the raw dataset, performs cleaning, and generates the cleaned dataset and cleaning report.

Train and evaluate the model:

```powershell
python .\improved_version\src\train_model.py
```

The trained pipeline is saved to:

```text
improved_version/models/accident_prediction_pipeline.joblib
```

## Limitations and Future Improvements

- The dataset covers only seven days and twelve locations.
- The positive accident class is significantly underrepresented.
- The current model has poor accident detection at its default threshold.
- Feature availability before an accident has not been established.
- Real-time traffic, weather, and signal data integration has not been implemented.
- A longer, independently collected dataset and an untouched chronological test period are needed for stronger validation.

Future work can explore improved feature engineering, better imbalance handling, threshold selection, independent validation, and real-time data integration.

## Author

Kothapalli Madhusudhan Reddy

GitHub: https://github.com/kothapalliMadhusudhanreddy
