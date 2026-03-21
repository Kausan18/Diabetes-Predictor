# === CELL 1: Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')
print("All imports successful")

# === CELL 2: Load Data ===
df = pd.read_csv('diabetes.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

# === CELL 3: Basic Stats ===
print("=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Class Balance ===")
print(df['Outcome'].value_counts())
print(f"\n{df['Outcome'].mean()*100:.1f}% of patients are diabetic")

print("\n=== Summary Stats ===")
df.describe().round(2)

# === CELL 4: Zero Value Check ===
# In medical data, a Glucose of 0 or BMI of 0 is biologically impossible
# These are actually missing values disguised as zeros

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("Zero counts in medical columns (these are actually missing values):")
for col in zero_cols:
    zeros = (df[col] == 0).sum()
    pct = zeros / len(df) * 100
    print(f"  {col:20s}: {zeros:3d} zeros ({pct:.1f}%)")


# === CELL 5: Feature Distributions ===
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

features = df.columns[:-1]  # all except Outcome

for i, col in enumerate(features):
    axes[i].hist(df[df['Outcome']==0][col], alpha=0.6,
                 label='No Diabetes', color='steelblue', bins=25)
    axes[i].hist(df[df['Outcome']==1][col], alpha=0.6,
                 label='Diabetes', color='coral', bins=25)
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

axes[-1].axis('off')
plt.suptitle('Feature Distributions by Outcome', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# === CELL 6: Correlation Heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# === CELL 7: Class Imbalance Bar ===
plt.figure(figsize=(5, 4))
counts = df['Outcome'].value_counts()
bars = plt.bar(['No Diabetes (0)', 'Diabetes (1)'],
               counts.values,
               color=['steelblue', 'coral'],
               edgecolor='white', width=0.5)

for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 5,
             f'{val}\n({val/len(df)*100:.1f}%)',
             ha='center', fontsize=11)

plt.title('Class Distribution')
plt.ylabel('Count')
plt.ylim(0, 600)
plt.tight_layout()
plt.show()