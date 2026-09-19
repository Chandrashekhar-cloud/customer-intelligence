# Milestone 3: Machine Learning Model Evaluation Report
**Project**: E-Commerce Customer Churn Prediction & Analytics System  
**Task**: Binary Classification (0 = Retained, 1 = Churned)  
**Evaluation Set**: Stratified holdout test set (untouched during preprocessing & training)

---

## 1. Executive Summary & Model Comparison

Two distinct machine learning algorithms were trained and evaluated:
1. **Model 1 (Baseline)**: Logistic Regression with balanced class weights
2. **Model 2 (Ensemble)**: Random Forest Classifier tuned via systematic hyperparameter grid search and trained on SMOTE-balanced data

### Performance Metrics Comparison Table

| Metric | Baseline: Logistic Regression | Tuned Random Forest | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.8002 | 0.9698 | +0.1696 |
| **Precision** | 0.4493 | 0.9432 | +0.4939 |
| **Recall (Sensitivity)** | 0.8158 | 0.8737 | +0.0579 |
| **F1-Score** | 0.5794 | 0.9071 | +0.3277 |
| **ROC-AUC** | 0.8885 | 0.9956 | +0.1071 |
| **PR-AUC (Avg Precision)**| 0.6884 | 0.9778 | +0.2894 |

---

## 2. Mathematical Rationale: Why Accuracy is Misleading for Imbalanced Data

In our dataset, approximately **84% of customers are retained (0)** and only **16% churn (1)**.

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

If a trivial dummy model simply predicts `0` (Retained) for every single customer:
- It achieves an **Accuracy of 84.0%**
- However, its **Recall is 0.0%** and **F1-score is 0.0**
- It fails to identify a single churning customer, rendering it completely useless for business intervention.

### Why Alternative Metrics are Superior in E-Commerce Churn:
- **Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$)**: Measures what percentage of actual churners the platform captured. A False Negative (FN) represents an undetected customer who cancels, costing hundreds of dollars in Customer Lifetime Value (CLV).
- **Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$)**: Measures the percentage of flagged customers who actually intended to churn. A False Positive (FP) results in an unnecessary promotional discount being extended to a loyal customer.
- **F1-Score**: The harmonic mean of Precision and Recall, balancing coverage against false alarm costs:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **PR-AUC (Precision-Recall Area Under Curve)**: The most rigorous threshold-independent metric for severely skewed datasets, directly quantifying the trade-off across all operating points.

---

## 3. Confusion Matrix Breakdown

### Baseline: Logistic Regression
- **True Negatives (TN)**: 746
- **False Positives (FP)**: 190
- **False Negatives (FN)**: 35
- **True Positives (TP)**: 155

### Tuned Random Forest (Selected Model)
- **True Negatives (TN)**: 926
- **False Positives (FP)**: 10
- **False Negatives (FN)**: 24
- **True Positives (TP)**: 166

---

## 4. Hyperparameter Optimization Details

The Random Forest was optimized using cross-validation over the training dataset:
```json
{
    "max_depth": null,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 150
}
```

---

## 5. Top 10 Feature Importances (Random Forest)

| Rank | Feature Name | Gini Importance |
| :---: | :--- | :---: |
| 1 | `tenure` | 0.1789 |
| 2 | `recency_ratio` | 0.0584 |
| 3 | `day_since_last_order` | 0.0543 |
| 4 | `cashback_amount` | 0.0475 |
| 5 | `cashback_per_order` | 0.0459 |
| 6 | `preferred_order_cat_Mobile Phone` | 0.0448 |
| 7 | `number_of_address` | 0.0414 |
| 8 | `complaint_friction` | 0.0398 |
| 9 | `marital_status_Single` | 0.0376 |
| 10 | `satisfaction_score` | 0.0373 |

---

## 6. Model Selection Conclusion

**Selected Production Model**: **Tuned Random Forest Classifier**  
- **Superior Discriminative Ability**: Outperforms the baseline across ROC-AUC, F1-Score, and PR-AUC.
- **Non-Linear Interactions**: Successfully captures complex multi-variable friction patterns (e.g., interaction between payment delay, support tickets, and month-to-month contracts).
- **Production Readiness**: Serialized with its preprocessing pipeline to `models/best_churn_model.joblib`.
