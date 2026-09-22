"""
Zepto Analytics Pipeline — Part B: Predictive Modeling & Regression
File: analytics/02_modeling.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

# Ensure artifacts directory exists
os.makedirs("analytics/artifacts", exist_ok=True)

# -----------------------------------------------------------------------------
# TASK 7 & 8: Data Loading, Stratified Split & Leakage-Free Pipeline
# -----------------------------------------------------------------------------
print("--- TASK 7 & 8: Stratified Split and Pipeline Preprocessing ---")

# Load from committed fallback
df = pd.read_csv("analytics/titanic.csv")

# Define target and features
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()
y = df['survived'].copy()

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Train class balance: {y_train.mean():.4f}, Test class balance: {y_test.mean():.4f}")

# Preprocessing Transformers
numeric_features = ['age', 'fare', 'sibsp', 'parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -----------------------------------------------------------------------------
# TASK 9 & 10: Model Training, Decision Tree Plot & Evaluation
# -----------------------------------------------------------------------------
print("\n--- TASK 9 & 10: Model Training and Multi-Metric Evaluation ---")

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
plt.figure(figsize=(8, 6))

for name, clf in classifiers.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1 Score': f1, 'ROC-AUC': auc, 'Confusion Matrix': cm
    }
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.title("ROC Curves Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("analytics/artifacts/model_roc_curves.png")
plt.close()

# Render Decision Tree
dt_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))])
dt_pipe.fit(X_train, y_train)

cat_encoder = dt_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + encoded_cat_names

plt.figure(figsize=(16, 8))
plot_tree(
    dt_pipe.named_steps['classifier'],
    feature_names=all_feature_names,
    class_names=['Not Survived', 'Survived'],
    filled=True, rounded=True, fontsize=10
)
plt.title("Decision Tree Visualization (Max Depth = 3)")
plt.tight_layout()
plt.savefig("analytics/artifacts/model_decision_tree.png")
plt.close()

# Print Comparison Results
print("\nClassification Evaluation Metrics:")
for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['Confusion Matrix']}")

# -----------------------------------------------------------------------------
# TASK 11: Imbalance Handling Comparison (Random Forest)
# -----------------------------------------------------------------------------
print("\n--- TASK 11: Class Imbalance Strategy Comparison ---")

# (a) Baseline
rf_base = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

# (b) Class Weight 'balanced'
rf_bal = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
rf_bal.fit(X_train, y_train)
y_pred_bal = rf_bal.predict(X_test)

# (c) SMOTE (Applied on training fold only)
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_res, y_train_res)
y_pred_smote = rf_smote.predict(X_test_prep)

print("Imbalance Strategy Performance (Random Forest):")
print(f"  1. Baseline       -> Precision: {precision_score(y_test, y_pred_base):.4f}, Recall: {recall_score(y_test, y_pred_base):.4f}, F1: {f1_score(y_test, y_pred_base):.4f}")
print(f"  2. Balanced Weight-> Precision: {precision_score(y_test, y_pred_bal):.4f}, Recall: {recall_score(y_test, y_pred_bal):.4f}, F1: {f1_score(y_test, y_pred_bal):.4f}")
print(f"  3. SMOTE (Train)  -> Precision: {precision_score(y_test, y_pred_smote):.4f}, Recall: {recall_score(y_test, y_pred_smote):.4f}, F1: {f1_score(y_test, y_pred_smote):.4f}")

# -----------------------------------------------------------------------------
# TASK 12: Hyperparameter Tuning & Out-of-Bag (OOB) Score
# -----------------------------------------------------------------------------
print("\n--- TASK 12: Hyperparameter Tuning & OOB Score ---")

rf_oob_base = RandomForestClassifier(oob_score=True, random_state=42)
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, 7],
    'classifier__max_features': ['sqrt', 'log2']
}

tune_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_oob_base)])
grid_search = GridSearchCV(tune_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
best_rf = best_pipeline.named_steps['classifier']

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Random Forest OOB Score: {best_rf.oob_score_:.4f}")

# -----------------------------------------------------------------------------
# TASK 13: Regression Side-Task (Predicting Fare)
# -----------------------------------------------------------------------------
print("\n--- TASK 13: Regression Side-Task (Predicting Fare) ---")

X_reg = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'survived', 'embarked']].copy()
y_reg = df['fare'].copy()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_num_features = ['age', 'sibsp', 'parch', 'survived']
reg_cat_features = ['pclass', 'sex', 'embarked']

reg_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), reg_num_features),
    ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), reg_cat_features)
])

reg_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', LinearRegression())])
reg_pipeline.fit(X_reg_train, y_reg_train)

y_reg_pred = reg_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

n = len(y_reg_test)
p = X_reg_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"Regression Metrics for 'fare':")
print(f"  MAE:          ${mae:.2f}")
print(f"  RMSE:         ${rmse:.2f}")
print(f"  R^2:          {r2:.4f}")
print(f"  Adjusted R^2: {adj_r2:.4f}")

# Residual Plot & Heteroscedasticity Analysis
residuals = y_reg_test - y_reg_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.6, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Regression Residuals vs Predicted Fare")
plt.xlabel("Predicted Fare ($)")
plt.ylabel("Residuals ($)")
plt.tight_layout()
plt.savefig("analytics/artifacts/model_residuals.png")
plt.close()

# -----------------------------------------------------------------------------
# TASK 14 & 15: Full Pipeline Export & Reload Verification
# -----------------------------------------------------------------------------
print("\n--- TASK 14 & 15: Exporting End-to-End Pipeline & Verification ---")

pipeline_export_path = "analytics/artifacts/zepto_titanic_pipeline.joblib"
joblib.dump(best_pipeline, pipeline_export_path)
print(f"Saved complete end-to-end fitted pipeline to '{pipeline_export_path}'.")

# Reload and verify on raw, unpreprocessed new data
reloaded_pipeline = joblib.load(pipeline_export_path)

# Mock raw customer order input
raw_new_data = pd.DataFrame([{
    'pclass': 1,
    'sex': 'female',
    'age': 28.0,
    'sibsp': 0,
    'parch': 0,
    'fare': 120.50,
    'embarked': 'S'
}])

prediction = reloaded_pipeline.predict(raw_new_data)[0]
probability = reloaded_pipeline.predict_proba(raw_new_data)[0][1]

print(f"\nVerification on Raw Input:")
print(f"  Input: 1st Class Female, 28 yrs, $120.50 Fare")
print(f"  Predicted Outcome: {'Survived (1)' if prediction == 1 else 'Not Survived (0)'}")
print(f"  Survival Probability: {probability:.4f}")

print("\nPart B Processing Completed Successfully.")
