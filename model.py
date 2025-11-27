# -----------------------------------------------
# Accident Prediction Pipeline - All in One
# -----------------------------------------------

# Step 0: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import os

# -----------------------------------------------
# Step 1: Load Dataset
# -----------------------------------------------
file_path = r"C:\Users\Admin\Downloads\Project_dataset (1) (1).csv"
df = pd.read_csv(file_path)

# -----------------------------------------------
# Step 2: Feature Engineering
# -----------------------------------------------
df['is_peak_hour'] = df['peak'].apply(lambda x: 1 if str(x).lower() in ['yes','y'] else 0)
df['is_night'] = df['lighting'].apply(lambda x: 1 if str(x).lower() in ['dark','night'] else 0)
df['weather_clear'] = df['weather'].apply(lambda x: 1 if str(x).lower() == 'clear' else 0)
df['road_highway'] = df['road_type'].apply(lambda x: 1 if 'highway' in str(x).lower() else 0)
df['weather_rainy'] = df['weather'].apply(lambda x: 1 if str(x).lower() in ['rain','rainy'] else 0)
df['weather_foggy'] = df['weather'].apply(lambda x: 1 if str(x).lower() in ['fog','foggy'] else 0)
df['peak_highway'] = df['is_peak_hour'] * df['road_highway']
df['night_rain'] = df['is_night'] * df['weather_rainy']
df['night_fog'] = df['is_night'] * df['weather_foggy']

if 'speed_limit' in df.columns:
    df['high_speed'] = df['speed_limit'].apply(lambda x: 1 if x >= 80 else 0)
if 'traffic_level' in df.columns:
    df['traffic_heavy'] = df['traffic_level'].apply(lambda x: 1 if str(x).lower() in ['heavy','high'] else 0)
if 'day_of_week' in df.columns:
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if str(x).lower() in ['saturday','sunday'] else 0)

print(" Feature engineering completed!")
# Step 5: Define Features & Target
target = 'accident_occurred'

# Drop 'severity' if it exists
if 'severity' in df.columns:
    df = df.drop(columns=['severity'])
if 'cause' in df.columns:
    df = df.drop(columns=['cause'])
if 'veh_count_at_accident' in df.columns:
    df = df.drop(columns=['veh_count_at_accident'])
   
X = df.drop(columns=[target])
y = df[target].astype(int)
df.dtypes
# -----------------------------------------------
# Step 3: Handle Missing Values
# -----------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -
# -----------------------------------------------
# Step X: Detect & Remove Outliers (IQR method)
# -----------------------------------------------

#----------------------------------------------
# Step 4: Encode Categorical Variables
# -----------------------------------------------
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------------------------
# Step 5: Define Features & Target
# -----------------------------------------------
target = 'accident_occurred'
X = df.drop(columns=[target])
y = df[target].astype(int)

# Ensure all columns are strings
X.columns = X.columns.astype(str)

# -----------------------------------------------
# Step 6: Scale Features
# -----------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# Step 7: Train-Test Split
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------
# Step 8: Handle Imbalance with SMOTE
# -----------------------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", np.bincount(y_train_res))

# -----------------------------------------------
# Step 9: Train Models
# -----------------------------------------------
# Logistic Regression

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train_res, y_train_res)
y_pred_gb = gb_clf.predict(X_test)

# -----------------------------------------------
# Step 10: Evaluate Models
# -----------------------------------------------
#  Train Gradient Boosting with better parameters
gb_clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    random_state=42
)
gb_clf.fit(X_train_res, y_train_res)



# Probabilities for positive class
y_prob_gb = gb_clf.predict_proba(X_test)[:, 1]

# Use a lower threshold to decrease precision
threshold = 0.28# try values <0.5
y_pred_thresh = (y_prob_gb >= threshold).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print("Precision:", round(precision_score(y_test, y_pred_thresh), 3))
print("Recall:", round(recall_score(y_test, y_pred_thresh), 3))
print("F1-score:", round(f1_score(y_test, y_pred_thresh), 3))
print("Accuracy:", round(accuracy_score(y_test, y_pred_thresh), 3))



import numpy as np
from sklearn.metrics import f1_score

# y_prob_gb = predicted probabilities from Gradient Boosting
thresholds = np.arange(0.0, 1.01, 0.01)  # test thresholds from 0 to 1
f1_scores = []

for t in thresholds:
    y_pred_thresh = (y_prob_gb >= t).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print("Best Threshold:", round(best_threshold, 3))
print("Best F1-score:", round(f1_scores[best_idx], 3))


import os
from joblib import dump

# Folder path
folder_path = r"C:\Users\madhu\Downloads\model23"

# Create folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Save model
dump(gb_clf, os.path.join(folder_path, "model.joblib"))

# Save scaler
dump(scaler, os.path.join(folder_path, "scaler.joblib"))

print("Model and scaler saved successfully!")
from joblib import load

model = load(r"C:\Users\madhu\Downloads\model23\model.joblib")
scaler = load(r"C:\Users\madhu\Downloads\model23\scaler.joblib")










