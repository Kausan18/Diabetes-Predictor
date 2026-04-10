import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset (assuming you still have df from Session 1)
df = pd.read_csv('diabetes.csv')  

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ⚠️ IMPORTANT: Split FIRST to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Class ratio (train): {y_train.value_counts(normalize=True).to_dict()}")

# Columns where 0 = missing
zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

preprocessor = Pipeline(steps=[
    # Replace 0s with median (sklearn handles missing_values=0 natively)
    ('imputer', SimpleImputer(missing_values=0, strategy='median')),
    # Scale all features to mean=0, std=1
    ('scaler', StandardScaler())
])

# ✅ FIT only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# ✅ TRANSFORM test data using training statistics
X_test_processed = preprocessor.transform(X_test)

# Convert back to DataFrame for easier debugging (optional but recommended)
X_train_processed = pd.DataFrame(X_train_processed, columns=X.columns, index=X_train.index)
X_test_processed  = pd.DataFrame(X_test_processed,  columns=X.columns, index=X_test.index)

print("✅ No missing values remaining?")
print("Train:", X_train_processed.isnull().sum().sum() == 0)
print("Test: ", X_test_processed.isnull().sum().sum() == 0)

print("\n✅ Scaled? (Mean ≈ 0, Std ≈ 1)")
print(X_train_processed.describe().round(2))

print("\n✅ Class balance preserved?")
print(y_train.value_counts(normalize=True).round(3))
print(y_test.value_counts(normalize=True).round(3))