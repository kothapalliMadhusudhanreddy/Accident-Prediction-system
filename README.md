# Accident-Prediction-system
# 🚦 Accident Prediction System

A machine learning-based **Accident Prediction System** that estimates the likelihood of a road accident using traffic conditions, road characteristics, vehicle speed, weather, lighting, and other related factors. The project uses a **Gradient Boosting Classifier** and a **Streamlit dashboard** to provide an interactive interface for accident risk assessment.

## 📌 Project Overview

Road accidents are influenced by multiple factors, including traffic density, road conditions, vehicle speed, weather, and lighting. This project uses machine learning to analyze these factors and predict whether an accident is likely to occur.

The system provides a user-friendly dashboard where users can enter traffic and road-related information and obtain a prediction based on a trained machine learning model.

## ✨ Features

- 🚗 Interactive dashboard built with Streamlit.
- 🛣️ Road and traffic information input.
- 🌦️ Weather and nighttime condition analysis.
- 🚦 Accident prediction using a Gradient Boosting Classifier.
- 📊 Feature scaling using StandardScaler.
- ⚖️ Class imbalance handling using SMOTE.
- 🧠 Feature engineering to derive additional traffic-related variables.
- 🔍 Expandable section to view the model input features.
- 📈 Probability-based classification using a configurable prediction threshold.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Frontend / Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Imbalanced Data Handling:** imbalanced-learn (SMOTE)
- **Model Serialization:** Joblib

## 📂 Project Structure

```text
Accident-Prediction-system/
│
├── app.py              # Streamlit dashboard and prediction interface
├── model.py            # Data preprocessing and model training
├── model.joblib        # Trained Gradient Boosting model
├── scaler.joblib       # Fitted StandardScaler
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies (recommended)
```

The trained model and scaler are saved as Joblib files and loaded by the dashboard for predictions.

## ⚙️ Machine Learning Workflow

The system follows these steps:

1. **Dataset Loading:** Loads the road accident dataset using Pandas.
2. **Data Preprocessing:** Handles missing values in numeric and categorical columns.
3. **Feature Engineering:** Creates features related to peak hours, nighttime conditions, weather, highways, heavy traffic, and weekends.
4. **Categorical Encoding:** Converts categorical features into numeric representations.
5. **Feature Scaling:** Applies StandardScaler to standardize the input features.
6. **Train-Test Split:** Divides the dataset into training and testing sets using an 80:20 split.
7. **Class Balancing:** Applies SMOTE to the training data to address class imbalance.
8. **Model Training:** Trains a Gradient Boosting Classifier.
9. **Model Evaluation:** Evaluates the model using precision, recall, F1-score, and accuracy.
10. **Model Saving:** Saves the trained model and scaler using Joblib.
11. **Prediction:** Loads the saved artifacts and predicts accident probability through the Streamlit dashboard.

## 📥 Dashboard Input Parameters

The dashboard accepts the following user inputs:

| Parameter | Description |
|---|---|
| Lane Count | Number of traffic lanes |
| Speed Limit | Posted speed limit in km/h |
| Blackspot Score | Road accident blackspot score |
| Vehicle Count per Hour | Number of vehicles passing per hour |
| Average Speed | Average vehicle speed in km/h |
| Hour of Day | Hour of the day from 0 to 23 |
| Peak Hour | Indicates whether the road is experiencing peak-hour traffic |
| Night | Indicates nighttime or dark lighting conditions |
| Weather Clear | Indicates clear weather conditions |
| Night & Rain | Indicates combined nighttime and rainy conditions |

Additional features required by the trained model are automatically populated with default values in the current implementation.

## 🚀 Installation and Setup

### Prerequisites

- Python 3.9 or a compatible Python version
- pip package manager
- The dataset used to train the model

### 1. Clone the Repository

```bash
git clone https://github.com/kothapalliMadhusudhanreddy/Accident-Prediction-system.git
```

### 2. Navigate to the Project Directory

```bash
cd Accident-Prediction-system
```

### 3. Install Dependencies

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn joblib
```

Alternatively, create a `requirements.txt` file with the following contents:

```text
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
joblib
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure the Model Files

Ensure that the following files are available:

- `model.joblib`
- `scaler.joblib`

**Important:** The current `app.py` uses a local Windows directory to locate these files. Update `folder_path` in `app.py` to match your environment, or modify the code to load the files relative to the project directory.

For example:

```python
from pathlib import Path

folder_path = Path(__file__).resolve().parent
model_path = folder_path / "model.joblib"
scaler_path = folder_path / "scaler.joblib"
```

The model and scaler must be compatible with the features and preprocessing used during training.

### 5. Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🖥️ How to Use the Application

1. Launch the Streamlit application.
2. Enter the road and traffic parameters in the sidebar.
3. Select the appropriate weather, nighttime, and peak-hour conditions.
4. Click **Predict Accident Risk**.
5. View the prediction displayed on the dashboard.
6. Expand **Show Input Details** to inspect the feature values supplied to the model.

## 📊 Model Details

The project uses the **Gradient Boosting Classifier**, an ensemble learning algorithm that combines multiple decision trees to make predictions.

The training script uses the following configuration for its final model:

```python
GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    random_state=42
)
```

SMOTE is applied to the training data to help address class imbalance. A prediction threshold is used to convert predicted probabilities into binary classifications.

The training script searches for a threshold that maximizes the F1-score on the test set, while the dashboard currently uses a threshold of `0.44`.

**Evaluation metrics:**

- Accuracy
- Precision
- Recall
- F1-score

Actual metric values should be added after running the training script on the project dataset.

## 🔮 Future Enhancements

- Integrate real-time traffic and weather data.
- Add interactive visualizations of accident risk.
- Display accident probability and risk categories.
- Improve the user interface with maps and location-based inputs.
- Compare multiple machine learning algorithms.
- Optimize the prediction threshold using validation data.
- Deploy the application to a cloud platform.
- Improve model interpretability using feature importance and explainability techniques.

## ⚠️ Limitations

- Prediction quality depends on the dataset and model training process.
- Some input features are assigned default values rather than collected from the user.
- Model performance may vary across different locations and traffic conditions.
- The system provides estimates and should not be treated as a guarantee that an accident will or will not occur.
- The current application requires compatible model and scaler files to run.

## 🎯 Project Objective

The primary objective of this project is to demonstrate how machine learning can be applied to road accident risk assessment using traffic and environmental data. The system provides an interactive way to explore accident predictions and can serve as a foundation for future intelligent transportation and road safety applications.

## 👨‍💻 Author

**Kothapalli Madhusudhan Reddy**

GitHub Repository: [Accident-Prediction-system](https://github.com/kothapalliMadhusudhanreddy/Accident-Prediction-system)
