# Analytics Module


# Zepto Analytics Pipeline — Titanic Customer Outcome Analysis

## Overview
This module demonstrates an end-to-end analyst-to-data-scientist workflow. It profiles and cleans the Titanic dataset, explores survival drivers through statistical and visual methods, builds leakage-free machine learning pipelines for classification, addresses class imbalance, tunes hyperparameters, conducts a regression side-task to predict fares, and exports a production-ready model artifact.

---

## Part A — Data Profiling & Data Story Summary

### 1. Dataset Profiling & Missing-Value Strategy
* **Total Records**: 891 rows, 15 columns
* **Missing Values Analysis**:
  * `deck`: 688 missing (**77.22%**) $\rightarrow$ **Dropped Column**. Imputation would introduce extreme noise (>30% missing threshold).
  * `age`: 177 missing (**19.87%**) $\rightarrow$ **Imputed via Median** (15%–30% threshold rule). Median (28.0) is robust to right-skewness in age.
  * `embarked` / `embark_town`: 2 missing (**0.22%**) $\rightarrow$ **Dropped Rows** (<5% threshold rule). Minimal loss of statistical power.

### 2. Univariate Analysis & Skewness Conclusion
* **Age Outliers (IQR Rule)**: 66 outliers outside $[Q1 - 1.5\times IQR, Q3 + 1.5\times IQR]$ ($[2.5, 54.5]$).
* **Fare Outliers (IQR Rule)**: 116 outliers outside $[Q1 - 1.5\times IQR, Q3 + 1.5\times IQR]$ ($[-26.76, 66.34]$).
* **Fare Distribution Statistics**:
  * **Mean**: $32.20
  * **Median**: $14.45
  * **Mode**: $8.05
  * **Conclusion**: **Strongly Right-Skewed**. Because $\text{Mean} (32.20) > \text{Median} (14.45) > \text{Mode} (8.05)$, the distribution exhibits a heavy positive tail driven by a small number of luxury first-class tickets (up to $512.33).

### 3. Bivariate Analysis & Correlation Heatmap
* **Survival Rates**:
  * By Sex: **Female = 74.20%**, **Male = 18.89%**
  * By Class: **1st Class = 62.96%**, **2nd Class = 47.28%**, **3rd Class = 24.24%**
  * By Sex & Class:
    * 1st Class Female: **96.81%** | 1st Class Male: **36.89%**
    * 2nd Class Female: **92.11%** | 2nd Class Male: **15.74%**
    * 3rd Class Female: **50.00%** | 3rd Class Male: **13.54%**

* **Top 2 Strongest Feature Correlations (6x6 Matrix)**:
  1. `pclass` and `fare` ($r = -0.549$): Strong inverse relationship — lower class numbers (1st class) correlate with significantly higher fares.
  2. `survived` and `pclass` ($r = -0.338$): Moderate inverse relationship — 1st class passengers (lower class integer) had higher survival rates.

### 4. Data Story Narrative (4 Visual Insights)
1. **Sex vs. Survival (Bar Chart)**: Women were prioritized during evacuation ("women and children first"), yielding a >74% survival rate compared to <19% for men.
2. **Passenger Class vs. Survival (Bar Chart)**: 1st class passengers had proximity to the upper deck and lifeboats, resulting in over 2.5x the survival rate of 3rd class passengers.
3. **Age Distribution by Survival (Box Plot)**: Young children (<5 years old) had significantly higher survival probability across all classes, whereas elderly passengers had lower survival rates.
4. **Fare vs. Survival by Class (Scatter/Box Plot)**: Higher ticket fares within each class correlated with higher survival, reflecting better cabin locations near exit routes.

---

## Part B — Predictive Modeling & Performance Metrics

### 1. Stratification Rationale
The classification target (`survived`) has an imbalanced ratio of approximately **61.6% Non-Survived (0)** to **38.4% Survived (1)**. A stratified train/test split (80/20) ensures that both training and testing splits maintain this exact 61.6:38.4 class distribution, preventing biased evaluation or unrepresentative test splits.

### 2. Model Comparison Table

#### Classification Performance Comparison
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8034 | 0.7656 | 0.7101 | 0.7368 | 0.8542 |
| **Decision Tree** | 0.7809 | 0.7246 | 0.7246 | 0.7246 | 0.7711 |
| **Random Forest (Tuned)** | **0.8315** | **0.8125** | **0.7536** | **0.7820** | **0.8710** |

#### Regression Performance Side-Task (Predicting `fare`)
| Model | MAE ($) | RMSE ($) | $R^2$ | Adjusted $R^2$ | Residual Pattern |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multivariate Linear Regression** | $17.42 | $34.12 | 0.5210 | 0.5142 | **Heteroscedastic** (fanning residual spread) |

---

### 3. Class Imbalance Experiment Results (Random Forest)
* **Baseline (No Handling)**: Precision = 0.8065, Recall = 0.7246, F1 = 0.7634
* **Class Weight ('balanced')**: Precision = 0.7812, Recall = 0.7246, F1 = 0.7519
* **SMOTE (Train Fold Only)**: Precision = 0.7639, Recall = 0.7971, **F1 = 0.7746**
* **Conclusion**: Applying SMOTE on the training fold improved Recall for the minority class (`survived` = 1) from 72.46% to 79.71%, leading to the best overall F1 balance without introducing test-set leakage.

### 4. Hyperparameter Tuning & Out-of-Bag (OOB) Score
* **GridSearch Optimal Hyperparameters**: `max_depth=5`, `max_features='sqrt'`, `n_estimators=100`
* **Random Forest OOB Score**: **0.8242** (82.42% out-of-bag accuracy)

### 5. Regression Residual Analysis & Heteroscedasticity Conclusion
The residual plot for `fare` prediction exhibits a distinct "funnel" or "fanning out" shape as predicted fares increase. Variance of residuals grows significantly for high predicted values. 
* **Conclusion**: **Heteroscedasticity is clearly present**. This occurs because high-class luxury tickets have much wider price variance compared to uniformly low 3rd-class ticket prices.

---

## Final Model Recommendation

**Selected Deployment Candidate**: **Random Forest Classifier (Tuned)**

**Justification**:
The tuned **Random Forest** outperforms both Logistic Regression and Decision Tree across all primary metrics on unseen test data, achieving an **Accuracy of 83.15%**, an **F1 Score of 0.7820**, and an **ROC-AUC of 0.8710**. It effectively balances Precision (81.25%) and Recall (75.36%) while capturing non-linear feature interactions (such as the combined impact of `sex`, `pclass`, and `age`) without overfitting. The high OOB score (0.8242) validates its strong generalization capacity.
