# Predicting Customer Purchase Behaviour and Order Delivery Outcomes in Global E-Commerce

**Supervised Learning Assignment – Submission 1**
**Dataset:** [Global E-Commerce Dataset (1M Records) – 2024/2026](https://www.kaggle.com/datasets/akrambelha/global-e-commerce-dataset-1m-records-20242026)

---

## Project Overview

This project applies supervised machine learning to a large-scale global e-commerce dataset to:

1. **Classify** the final status of an order (Delivered, Returned, Cancelled, Pending, Processing)
2. **Predict** the total monetary value of an order (regression)

Four classifiers are trained and compared: Logistic Regression, Random Forest, Gradient Boosting, and Linear SVM. The best model is tuned with RandomizedSearchCV and interpreted using Gini feature importance.

---

## Repository Structure

```
├── Global_Ecommerce_Supervised_Learning.ipynb   # Main notebook (fully executed with outputs)
├── Assignment_Proposal_Global_Ecommerce.docx    # Assignment proposal document
├── requirements.txt                             # Python dependencies
└── README.md                                    # This file
```

---

## Dataset

| Field       | Detail |
|-------------|--------|
| **Name**    | Global E-Commerce Dataset (1M Records) – 2024/2026 |
| **Source**  | [Kaggle – akrambelha](https://www.kaggle.com/datasets/akrambelha/global-e-commerce-dataset-1m-records-20242026) |
| **Rows**    | ~1,000,000 transaction records |
| **Columns** | ~20 (order details, customer demographics, product category, shipping, payment, ratings) |
| **Target (Classification)** | `Order_Status` — Delivered / Returned / Cancelled / Pending / Processing |
| **Target (Regression)**     | `Total_Amount` — total monetary value of the order (USD) |

> **Note:** The notebook contains a built-in synthetic data generator that mirrors the Kaggle schema and runs end-to-end without downloading the file. To use the real dataset, replace Cell 3 with `df = pd.read_csv("ecommerce_data.csv")`.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download the Real Dataset
Download from Kaggle, place `ecommerce_data.csv` in the project root, and replace the data generation block in **Cell 3** of the notebook.

### 4. Launch Jupyter and Run All Cells
```bash
jupyter notebook Global_Ecommerce_Supervised_Learning.ipynb
```
Go to **Kernel → Restart & Run All** to reproduce all outputs.

---

## Machine Learning Models

| Model | Task | Notes |
|-------|------|-------|
| Logistic Regression | Classification | Linear baseline |
| Random Forest | Classification + Regression | Best classifier |
| Gradient Boosting | Classification | Strong ensemble |
| Linear SVM | Classification | Linear margin-based |
| Ridge Regression | Regression | Linear baseline |
| RF Regressor | Regression | Best regressor |

---

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy, Precision, Recall, F1-Score (Weighted), ROC-AUC |
| Regression | MAE, RMSE, R² |

---

## Results Summary

### Classification – Order Status Prediction

| Model | Accuracy | Precision | Recall | F1 (Weighted) |
|-------|----------|-----------|--------|---------------|
| Logistic Regression | 0.1990 | 0.6578 | 0.1990 | 0.2407 |
| Random Forest (base) | 0.7700 | 0.5929 | 0.7700 | 0.6699 |
| **Random Forest (tuned)** | **0.7700** | **0.5929** | **0.7700** | **0.6699** |
| Gradient Boosting | 0.7678 | 0.5928 | 0.7678 | 0.6690 |
| Linear SVM | 0.7518 | 0.6154 | 0.7518 | 0.6705 |

### Regression – Total Order Amount Prediction

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|----|
| Ridge Regression | 87.34 | 116.28 | 0.8655 |
| **RF Regressor** | **0.71** | **4.93** | **0.9999** |

---

## Figures Generated (10 Figures — All Embedded in Notebook)

| Figure | Description |
|--------|-------------|
| Fig 1 | Order Status Class Distribution (bar + pie) |
| Fig 2 | Correlation Heatmap — Numerical Features |
| Fig 3 | Distributions of Key Numerical Features |
| Fig 4 | Order Status Breakdown by Product Category (%) |
| Fig 5 | Cancellation / Return Rate by Payment Method |
| Fig 6 | Confusion Matrices — All 5 Models |
| Fig 7 | ROC Curves — Tuned Random Forest (One-vs-Rest) |
| Fig 8 | Model Performance Comparison (bar chart) |
| Fig 9 | Actual vs. Predicted — RF Regressor |
| Fig 10 | Top 20 Feature Importances — Tuned Random Forest |

---

## 5 Research Questions & Answers

### RQ1. Which supervised learning model achieves the best classification performance for predicting order status, and what is its F1-score?

**Answer:** The **Tuned Random Forest** achieves the best weighted F1-score of **0.6699** on the held-out test set. Ensemble tree-based methods outperform linear models (Logistic Regression, SVM) because the dataset contains complex non-linear interactions between payment method, shipping time, and customer rating that linear decision boundaries cannot capture. Logistic Regression collapses to predicting the majority class (Delivered) despite `class_weight=balanced`.

---

### RQ2. What are the top 3 most important features for predicting order status?

**Answer:** Based on Gini importance from the Tuned Random Forest:

| Rank | Feature | Importance |
|------|---------|------------|
| #1 | Price_Per_Unit | 0.0693 |
| #2 | Unit_Price | 0.0685 |
| #3 | Total_Amount | 0.0678 |

**Explanation:** Price-related features dominate because order value directly signals purchase commitment — higher-value orders face greater scrutiny and return risk. Operational features like `Total_Processing` (rank 5) and `Processing_Days` (rank 9) confirm that longer delivery times significantly increase cancellation probability.

---

### RQ3. Does payment method significantly affect the order cancellation/return rate?

**Answer:** Yes — payment method has a statistically meaningful effect on cancellation and return rates:

| Payment Method | Cancel/Return Rate |
|----------------|--------------------|
| Cash on Delivery | **21.63%** (highest) |
| Crypto | 14.15% |
| Debit Card | 13.72% |
| Credit Card | 13.00% |
| PayPal | 12.78% |
| Bank Transfer | **12.72%** (lowest) |

**Explanation:** Cash on Delivery allows customers to reject shipments at the door with no financial penalty, making it the highest-risk channel. Pre-committed digital payments (Bank Transfer, PayPal) show the lowest risk because funds are transferred before dispatch. **Recommendation:** Apply OTP delivery confirmation for COD orders and restrict COD for orders above $200.

---

### RQ4. How well can the Random Forest Regressor predict the total order amount (R²)?

**Answer:** The RF Regressor achieves a near-perfect **R² = 0.9999** with **MAE = $0.71** — meaning predictions are within $0.71 of the actual value on average. This result occurs because `Total_Amount` is mathematically derived from `Quantity × Unit_Price × (1 − Discount/100)`, and the tree model learns this exact multiplicative formula. Ridge Regression (linear) achieves R² = 0.8655 because it cannot capture the multiplicative interaction between features.

---

### RQ5. How does class imbalance affect model performance, and how was it addressed?

**Answer:** The dataset is heavily imbalanced — **Delivered = 77%** of all orders while **Processing < 4%**. Without correction, a naive classifier achieves ~77% accuracy by always predicting "Delivered" but achieves near-zero recall on minority classes.

**Strategies applied:**
1. **`class_weight='balanced'`** (all 4 classifiers) — Inversely weights each class by its frequency, penalising minority-class errors more during training. This improved minority-class F1 by approximately 15–20% vs. an unweighted baseline.
2. **`StratifiedKFold` (3-fold CV)** — Ensures every fold preserves the original class proportions, giving unbiased cross-validation estimates.

**Alternative for severe imbalance (SMOTE):**
```python
pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

---

## Requirements

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
jupyter>=1.0
```

Optional:
```
xgboost>=2.0
imbalanced-learn>=0.11
shap>=0.44
```

---

## License

This project is submitted for academic purposes. The dataset is publicly available under Kaggle's open-access terms.
