import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# === CELL 2: Load & Replace Zeros with NaN ===
df = pd.read_csv('diabetes.csv')

# These columns cannot biologically be zero — treat as missing
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print("Zeros replaced with NaN")
print(df[zero_cols].isnull().sum())

# === CELL 3: Train/Test Split ===
# Split BEFORE any scaling — this is critical to prevent data leakage

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # preserves the 65/35 class ratio in both splits
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"\nClass balance in train:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\nClass balance in test:\n{y_test.value_counts(normalize=True).round(3)}")

# === CELL 4: Build the sklearn Pipeline ===
# The Pipeline chains steps — each step's output feeds the next
# Key rule: fit() only ever sees training data

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # fills NaN with median
    ('scaler',  StandardScaler())                   # z-score normalisation
])

# fit_transform on train: learns median & μ/σ from training data, then applies
X_train_processed = pipeline.fit_transform(X_train)

# transform on test: applies the SAME medians & μ/σ learned from train only
X_test_processed  = pipeline.transform(X_test)

print("Pipeline fitted and applied")
print(f"Processed train shape: {X_train_processed.shape}")
print(f"Processed test shape:  {X_test_processed.shape}")

# === CELL 5: Verify Scaling ===
# After StandardScaler, train data should have mean ≈ 0 and std ≈ 1
# Test data will be close but not exact — that's expected and correct

train_df = pd.DataFrame(X_train_processed, columns=X.columns)
test_df  = pd.DataFrame(X_test_processed,  columns=X.columns)

print("=== Train set stats (should be ~0 mean, ~1 std) ===")
print(train_df.describe().loc[['mean','std']].round(3))

print("\n=== Test set stats (close but not exact — this is correct) ===")
print(test_df.describe().loc[['mean','std']].round(3))

