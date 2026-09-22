"""
Zepto Analytics Pipeline — Part A: Profiling, Cleaning & Data Story
File: analytics/01_eda.py
"""

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ensure output directories exist
os.makedirs("analytics/artifacts", exist_ok=True)

# -----------------------------------------------------------------------------
# TASK 1: Single Load & Offline Fallback Export
# -----------------------------------------------------------------------------
print("--- TASK 1: Loading Dataset & Profiling ---")
try:
    df_raw = sns.load_dataset('titanic')
    print("Successfully fetched 'titanic' via Seaborn loader.")
except Exception as e:
    print(f"Network load failed ({e}); checking local fallback 'analytics/titanic.csv'...")
    df_raw = pd.read_csv("analytics/titanic.csv")

# Save committed offline fallback immediately
df_raw.to_csv("analytics/titanic.csv", index=False)
print("Saved committed offline fallback to 'analytics/titanic.csv'.")

# Dataset Profiling
print("\n[df.info()]")
df_raw.info()

print("\n[df.describe()]")
print(df_raw.describe(include='all'))

print(f"\nDataset Shape: {df_raw.shape}")

# Compute & Report Missing Value Percentages
print("\n--- Missing Values Percentage Report ---")
missing_series = df_raw.isnull().mean() * 100
missing_cols = missing_series[missing_series > 0]
for col, pct in missing_cols.items():
    print(f"Column '{col}': {pct:.2f}% missing ({df_raw[col].isnull().sum()} rows)")

# -----------------------------------------------------------------------------
# TASK 2: Missing-Value Handling Strategy Application
# -----------------------------------------------------------------------------
print("\n--- TASK 2: Missing Value Strategy Application ---")
df = df_raw.copy()

# Strategy Application:
# 1. deck (77.22% missing > 30% threshold) -> Drop Column
df.drop(columns=['deck'], inplace=True)
print("Action: Dropped 'deck' column (>30% missing threshold).")

# 2. age (19.87% missing -> 5%-30% threshold) -> Impute with Median
age_median = df['age'].median()
df['age'].fillna(age_median, inplace=True)
print(f"Action: Imputed 'age' missing values with median ({age_median}).")

# 3. embarked & embark_town (0.22% missing < 5% threshold) -> Drop Rows
df.dropna(subset=['embarked', 'embark_town'], inplace=True)
print("Action: Dropped rows with missing 'embarked'/'embark_town' (<5% missing).")

print(f"Cleaned Dataset Shape: {df.shape}")

# -----------------------------------------------------------------------------
# TASK 3: Univariate Analysis, IQR Outliers & Skewness
# -----------------------------------------------------------------------------
print("\n--- TASK 3: Univariate Analysis & Outlier Detection ---")

def detect_iqr_outliers(series: pd.Series, name: str):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    print(f"{name} IQR Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Number of Outliers in {name}: {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")
    return outliers

detect_iqr_outliers(df['age'], "Age")
detect_iqr_outliers(df['fare'], "Fare")

fare_mean = df['fare'].mean()
fare_median = df['fare'].median()
fare_mode = df['fare'].mode()[0]

print(f"\nFare Central Tendencies:")
print(f"  Mean:   {fare_mean:.2f}")
print(f"  Median: {fare_median:.2f}")
print(f"  Mode:   {fare_mode:.2f}")

if fare_mean > fare_median > fare_mode:
    skew_str = "Strongly Right-Skewed (Positive Skew)"
elif fare_mean < fare_median < fare_mode:
    skew_str = "Strongly Left-Skewed (Negative Skew)"
else:
    skew_str = "Symmetric"
print(f"Fare Distribution Conclusion: {skew_str} because Mean ({fare_mean:.2f}) > Median ({fare_median:.2f}) > Mode ({fare_mode:.2f}).")

# Plot Distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['age'], kde=True, ax=axes[0, 0], color='skyblue').set_title("Age Distribution")
sns.boxplot(x=df['age'], ax=axes[0, 1], color='skyblue').set_title("Age Box Plot")
sns.histplot(df['fare'], kde=True, ax=axes[1, 0], color='salmon').set_title("Fare Distribution")
sns.boxplot(x=df['fare'], ax=axes[1, 1], color='salmon').set_title("Fare Box Plot")
plt.tight_layout()
plt.savefig("analytics/artifacts/eda_distributions.png")
plt.close()

# -----------------------------------------------------------------------------
# TASK 4: Bivariate Analysis & 6x6 Correlation Matrix
# -----------------------------------------------------------------------------
print("\n--- TASK 4: Bivariate Analysis & Correlation Matrix ---")

# (a) Survival by Sex
sex_surv = df.groupby('sex')['survived'].mean()
print("Survival Rate by Sex:\n", sex_surv)

# (b) Survival by Pclass
pclass_surv = df.groupby('pclass')['survived'].mean()
print("\nSurvival Rate by Pclass:\n", pclass_surv)

# (c) Survival by Sex and Pclass
sex_pclass_surv = df.groupby(['sex', 'pclass'])['survived'].mean()
print("\nSurvival Rate by Sex & Pclass:\n", sex_pclass_surv)

# Correlation Matrix (Restricted to exactly 6 numeric columns; excluding adult_male, alone)
target_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df[target_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
plt.title("6x6 Correlation Heatmap (Numeric Columns)")
plt.tight_layout()
plt.savefig("analytics/artifacts/eda_correlation_heatmap.png")
plt.close()

# Find top 2 off-diagonal absolute correlations
corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0].sort_values(ascending=False)
top_2 = corr_pairs.drop_duplicates().head(2)

print("\nTop 2 Strongest Off-Diagonal Correlations:")
for (c1, c2), val in top_2.items():
    actual_corr = corr_matrix.loc[c1, c2]
    print(f"  {c1} & {c2}: correlation = {actual_corr:.3f} (abs = {val:.3f})")

# -----------------------------------------------------------------------------
# TASK 5: Multivariate "Data Story" Visualizations
# -----------------------------------------------------------------------------
print("\n--- TASK 5: Generating Data Story Charts ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Survival by Sex
sns.barplot(data=df, x='sex', y='survived', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title("1. Survival Rate by Sex")
axes[0, 0].set_ylabel("Survival Rate")

# Chart 2: Survival by Pclass
sns.barplot(data=df, x='pclass', y='survived', hue='sex', ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title("2. Survival Rate by Class & Sex")
axes[0, 1].set_ylabel("Survival Rate")

# Chart 3: Age Distribution by Survival Status
sns.boxplot(data=df, x='survived', y='age', hue='pclass', ax=axes[1, 0], palette='Set3')
axes[1, 0].set_title("3. Age Distribution by Survival & Class")

# Chart 4: Fare vs Pclass by Survival
sns.boxplot(data=df, x='pclass', y='fare', hue='survived', ax=axes[1, 1], palette='coolwarm')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title("4. Log(Fare) by Class & Survival")

plt.tight_layout()
plt.savefig("analytics/artifacts/eda_data_story_grid.png")
plt.close()
print("Saved Data Story Grid to 'analytics/artifacts/eda_data_story_grid.png'.")

# -----------------------------------------------------------------------------
# TASK 6: Exploratory Standardization Check (z-score)
# -----------------------------------------------------------------------------
print("\n--- TASK 6: Exploratory Standardization Check ---")
df_z = df.copy()
df_z['age_z'] = (df_z['age'] - df_z['age'].mean()) / df_z['age'].std()
df_z['fare_z'] = (df_z['fare'] - df_z['fare'].mean()) / df_z['fare'].std()

print("Before Standardization:")
print(f"  Age  -> Mean: {df['age'].mean():.4f}, Std: {df['age'].std():.4f}")
print(f"  Fare -> Mean: {df['fare'].mean():.4f}, Std: {df['fare'].std():.4f}")

print("\nAfter Standardization (Z-score formula):")
print(f"  Age_Z  -> Mean: {df_z['age_z'].mean():.4f}, Std: {df_z['age_z'].std():.4f}")
print(f"  Fare_Z -> Mean: {df_z['fare_z'].mean():.4f}, Std: {df_z['fare_z'].std():.4f}")

print("\nPart A Processing Completed Successfully.")
