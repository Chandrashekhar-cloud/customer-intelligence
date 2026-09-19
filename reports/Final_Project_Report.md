# E-Commerce Customer Churn Prediction & Analytics System
## Major Project Report (Assignment 2)

**Academic Level:** Final Year Undergraduate / Graduate Engineering  
**Department:** Computer Science & Engineering / Data Science  
**Domain:** Machine Learning, E-Commerce Analytics, Site Reliability & MLOps  
**Implementation Codebase:** [GitHub / Project Root](file:///c:/Users/chand/Downloads/aiml)  

---

## 1. Title
**Design and Implementation of an End-to-End Customer Churn Prediction Engine with Leak-Free Preprocessing, Resampling, and Production REST API Deployment**

---

## 2. Abstract
Customer retention represents a core business imperative for modern subscription and e-commerce platforms. Predicting customer churn before it manifests allows organizations to deliver targeted interventions, optimize customer lifetime value (CLV), and minimize revenue leakage. This project designs, evaluates, and deploys a complete, production-grade Machine Learning system capable of predicting customer churn in real time.

Addressing real-world data imperfections, the project generates a benchmark synthetic dataset of 5,000 customer records embedded with realistic class imbalance (~16.2% churn vs ~83.8% retained), missing telemetry values (up to 4.9% across selected indicators), and correlated behavioral anomalies. A strict leak-free feature engineering and preprocessing pipeline is implemented, utilizing stratified splitting, training-only median/mode imputation, IQR-based outlier winsorization, and domain interaction features. To counteract target skew, Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to training folds. A baseline Logistic Regression model is benchmarked against an ensemble Random Forest Classifier optimized via 5-fold cross-validated grid search. The final pipeline is exposed through an independent Flask REST API with strict schema validation and 8 automated Pytest unit tests (100% pass rate). An interactive web dashboard provides real-time inference, risk tiering, and operational recommendations.

---

## 3. Introduction
In competitive digital commerce, the cost of acquiring a new customer is estimated to be 5 to 7 times higher than retaining an existing customer. Predictive churn modeling transforms customer telemetry—such as login recency, ticket complaints, and invoice delays—into actionable probabilities that enable proactive customer success interventions.

Traditional educational machine learning projects often rely on pre-cleaned, balanced datasets (e.g., Kaggle Telco Churn), failing to prepare engineering students for production realities such as data leakage, missing telemetry, class skew, and API contract enforcement. This project replicates an authentic industry engineering lifecycle across five distinct milestones.

---

## 4. Problem Statement
Given multi-dimensional customer demographic, financial, and behavioral telemetry, build a binary classification system that predicts the target variable $\text{churn} \in \{0, 1\}$, where:
- `0`: Customer Retained (Active)
- `1`: Customer Churned (Cancelled / Inactive)

The system must handle severe class imbalance, avoid data leakage during scaling and imputation, provide threshold-independent metric rigor (PR-AUC, ROC-AUC, F1-Score), expose predictions via a microservice REST API, and support real-time user interaction through an accessible frontend interface.

---

## 5. Objectives
1. **Milestone 1 — Synthetic Data Engineering:** Synthesize 5,000 realistic e-commerce customer records incorporating class imbalance (~80/20), condition-based missingness, and correlated behavioral outliers.
2. **Milestone 2 — EDA & Feature Engineering:** Perform exploratory data analysis, implement leak-free imputation and outlier handling, and engineer domain behavioral indicators.
3. **Milestone 3 — Algorithmic Modeling:** Train a baseline Logistic Regression model and a tuned Random Forest Classifier; evaluate performance using non-accuracy metrics and persist mathematical artifacts using Joblib.
4. **Milestone 4 — REST API & Automated Testing:** Construct an independent Flask REST API with Pydantic-style schema validation, comprehensive error handling, and $\ge 6$ Pytest unit tests.
5. **Milestone 5 — Frontend Dashboard:** Deploy an interactive, clean browser dashboard featuring customer personas, real-time API communication, probability meters, and risk-tier alerts.

---

## 6. Literature & Background
Predicting customer churn has evolved from statistical survival analysis to modern supervised learning. In subscription models, churn exhibits strong correlation with customer effort scores (CES) and service friction.

Previous studies highlight several recurrent pitfalls in churn modeling:
- **Accuracy Paradox:** Naive models predicting the majority class achieve high raw accuracy while failing completely on minority recall.
- **Data Leakage in SMOTE:** Applying oversampling or scaling prior to train/test partitioning contaminates validation sets, producing overly optimistic test metrics that degrade catastrophically in production.
- **Serving Skew:** Discrepancies between training-time data transformations and runtime API processing pipeline.

---

## 7. Proposed System Architecture
The system architecture follows a modular, decoupled pipeline:

```
[Raw Telemetry Generator] 
       │
       ▼
[Stratified 80/20 Train-Test Partition]
       │
       ├──────────────────────────────────────────┐
       ▼ (Training Data ONLY)                     ▼ (Test Data)
[Fit Anomaly Caps & Medians]             [Apply Precomputed Caps & Medians]
       │                                          │
[Domain Feature Engineering]             [Domain Feature Engineering]
       │                                          │
[Fit ColumnTransformer] ───────────────> [Transform with Fitted ColumnTransformer]
       │                                          │
[Apply SMOTE Oversampling]                        │
       │                                          │
[Train Baseline & Tuned Random Forest]            │
       │                                          │
       └───────────────────┬──────────────────────┘
                           ▼
               [Holdout Evaluation & PR-AUC]
                           │
                           ▼
          [Joblib Serialization (Pipeline)]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [Flask REST API :5000]     [Pytest Unit Suite]
             │
             ▼
   [Frontend Dashboard UI]
```

---

## 8. Dataset Engineering (Milestone 1)

### Features Schema
The synthetic dataset encompasses 11 input features and 1 target:

| Feature Name | Type | Domain / Range | Description |
| :--- | :--- | :--- | :--- |
| `customer_age` | Integer | 18 – 75 | Customer chronological age |
| `tenure_months` | Integer | 1 – 72 | Active account duration in months |
| `monthly_spend` | Float | $15.00 – $910.38 | Current recurring monthly bill |
| `total_spend` | Float | $15.00 – $11,413.81 | Cumulative platform lifetime spend |
| `login_frequency` | Integer | 1 – 50 | Monthly portal / mobile app visits |
| `support_tickets` | Float | 0 – 14 | Customer support complaints logged |
| `payment_delay_days`| Integer | 0 – 84 | Invoice settlement delay in days |
| `subscription_type` | Categorical| Basic, Standard, Premium | Account service tier |
| `contract_type` | Categorical| Month-to-Month, One-Year, Two-Year | Agreement duration |
| `discount_used` | Binary | 0 or 1 | Promotional coupon redemption |
| `last_login_days` | Float | 1 – 90 | Recency of last active session |
| `churn` | Binary Target | 0 or 1 | 0 = Retained, 1 = Churned |

### Flaw Injection Rationale
- **Class Imbalance:** 4,189 Retained (83.78%) vs. 811 Churned (16.22%), mimicking real SaaS churn distributions.
- **Missing Telemetry:** Injected missing values into `monthly_spend` (194 missing, 3.88%), `support_tickets` (174 missing, 3.48%), and `last_login_days` (247 missing, 4.94%).
- **Correlated Anomalies:** Delinquency delays up to 84 days and extreme monthly spending outliers up to $910.38 were injected to evaluate model robustness against outliers.

---

## 9. Data Preprocessing & Leak-Free Pipeline (Milestone 2)

### Zero Data Leakage Enforcement
To prevent training statistics from contaminating the evaluation holdout:
1. The raw dataset (5,000 rows) is first split into **Training (4,000 rows, 80%)** and **Test (1,000 rows, 20%)** using stratified sampling on `churn`.
2. Imputation parameters (medians for continuous variables, modes for categoricals) are calculated strictly on the training set and stored in an imputer dictionary.
3. Outlier caps using the Interquartile Range ($3.0 \times \text{IQR}$) are computed solely from the training distribution.
4. The test set is transformed strictly using the training-derived parameters.

---

## 10. Feature Engineering (Milestone 2)
Five domain-specific features were engineered to capture customer behavioral friction:

1. **Average Historical Monthly Spend:**
   $$\text{avg\_historical\_spend} = \frac{\text{total\_spend}}{\text{tenure\_months} + 1}$$
2. **Spend Deviation:**
   $$\text{spend\_deviation} = \text{monthly\_spend} - \text{avg\_historical\_spend}$$
3. **Engagement Score:**
   $$\text{engagement\_score} = \frac{\text{login\_frequency}}{\text{last\_login\_days} + 1}$$
4. **Support Ticket Intensity:**
   $$\text{support\_ticket\_intensity} = \frac{\text{support\_tickets}}{\text{tenure\_months} + 1}$$
5. **Payment Risk Score:**
   $$\text{payment\_risk\_score} = \text{payment\_delay\_days} \times \text{support\_tickets}$$

---

## 11. Machine Learning Modeling & Optimization (Milestone 3)

### Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) was applied **strictly to the training feature matrix**:
- Pre-SMOTE Training Churn: 649 / 4,000 (16.2%)
- Post-SMOTE Resampled Training Records: 3,351 churn / 6,702 total (50.0% balanced)
- Untouched Test Set Churn: 162 / 1,000 (16.2% true natural prevalence)

### Models Evaluated
1. **Model 1 (Baseline):** Regularized Logistic Regression (`max_iter=1000`, `random_state=42`).
2. **Model 2 (Ensemble):** Random Forest Classifier optimized via `GridSearchCV` using 5-fold Stratified K-Fold cross-validation (`cv=5`, `scoring='roc_auc'`).
   - Hyperparameter grid searched: `n_estimators` [100, 150], `max_depth` [8, 12, None], `min_samples_split` [2, 5], `min_samples_leaf` [1, 2], `max_features` ['sqrt'].
   - Best Hyperparameters: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}`.
   - Best Cross-Validation ROC-AUC: **0.9715**.

---

## 12. Model Evaluation & Comparison

### Empirical Results on 1,000 Holdout Test Records

| Metric | Baseline: Logistic Regression | Tuned Random Forest | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7200 | **0.8270** | +0.1070 |
| **Precision** | 0.3397 | **0.4631** | +0.1234 |
| **Recall (Sensitivity)** | **0.7716** | 0.4259 | -0.3457 |
| **F1-Score** | **0.4717** | 0.4437 | -0.0280 |
| **ROC-AUC** | **0.8310** | 0.8016 | -0.0294 |
| **PR-AUC (Avg Precision)**| **0.5456** | 0.4620 | -0.0836 |

### Confusion Matrix Breakdown (Test Set, N = 1,000)
- **Baseline Logistic Regression:**
  - True Negatives (TN): 595
  - False Positives (FP): 243
  - False Negatives (FN): 37
  - True Positives (TP): 125 (High recall: captures 77.2% of all churners)
- **Tuned Random Forest:**
  - True Negatives (TN): 758
  - False Positives (FP): 80 (Low false alarms: 80 vs 243)
  - False Negatives (FN): 93
  - True Positives (TP): 69

### Analysis of Trade-offs
- The **Logistic Regression** model operates as a high-sensitivity screener (Recall 77.2%), capturing the vast majority of at-risk customers at the cost of higher false alarms.
- The **Random Forest** model delivers substantially higher overall accuracy (82.7% vs 72.0%) and precision (46.3% vs 34.0%), making it ideal when retention discounts carry significant cost.

---

## 13. REST API Architecture (Milestone 4)
Built using Flask and Flask-CORS, the backend microservice provides:
- `GET /health`: Returns JSON health status, microservice name, version, and model availability.
- `POST /predict`: Accepts customer JSON telemetry, validates against schema rules, feeds through the serialized pipeline, and returns:
  ```json
  {
    "prediction": 1,
    "prediction_label": "Likely to Churn",
    "probability": 0.7067,
    "risk_tier": "High Risk",
    "recommendation": "Immediate proactive outreach: Offer retention discount and schedule success manager check-in."
  }
  ```

---

## 14. Automated Testing Suite (Milestone 4)
The Pytest suite (`tests/test_api.py`) contains 8 automated unit tests:
1. `test_health_check_endpoint`: PASSED
2. `test_valid_prediction_request`: PASSED
3. `test_high_risk_customer_response`: PASSED
4. `test_missing_required_field`: PASSED
5. `test_invalid_numeric_data_type`: PASSED
6. `test_invalid_categorical_value`: PASSED
7. `test_negative_boundary_constraint`: PASSED
8. `test_prediction_response_schema_completeness`: PASSED

**Result:** `8 passed in 3.83s` (100% test success).

---

## 15. Frontend Dashboard (Milestone 5)
The frontend dashboard (`frontend/index.html`, `styles.css`, `app.js`) features:
- Live KPI cards: Total records, baseline churn rate, average spend, and model ROC-AUC.
- One-click customer persona presets:
  - 🟢 **Loyal VIP Customer:** 2-year contract, high spend, zero delay $\rightarrow$ **4.67% churn risk**.
  - 🟡 **At-Risk Account:** Month-to-month, moderate delay $\rightarrow$ **38.4% churn risk**.
  - 🔴 **Critical Hazard:** Month-to-month, 6 tickets, 18 days delay $\rightarrow$ **70.67% churn risk**.
- Real-time API connection indicator with automatic fallback.
- Dynamic probability meter with color-coded risk tiers and actionable SRE/retention guidance.

---

## 16. AI Tool Usage Documentation
In compliance with assignment guidelines, AI assistance was systematically leveraged across development:
1. **Synthetic Data Engineering:** AI prompted to inject correlated multivariate noise, MCAR/MAR missingness, and boundary anomalies into `data_generation.py`.
2. **Modular EDA Architecture:** AI generated publication-ready Matplotlib/Seaborn scripts avoiding deprecated APIs.
3. **Hyperparameter Optimization Routine:** AI structured leak-free GridSearchCV with stratified k-fold splits.
4. **FastAPI / Flask API Schema Design:** AI generated strict boundary checks and Pytest fixtures.
5. **Modern Frontend Aesthetics:** AI constructed dark-mode glassmorphic CSS tokens and responsive layout.

---

## 17. Conclusion & Viva Examination Guide
This project satisfies all requirements of Assignment 2 with verified, executable code.

### Viva / Interview Questions & Answers
1. **Q: Why shouldn't we use Accuracy as the sole metric for churn prediction?**  
   *A:* In our dataset, 83.8% of customers are retained. A trivial dummy classifier predicting 0 for all instances achieves 83.8% accuracy while missing 100% of churners (Recall = 0%). Recall and PR-AUC measure true minority detection effectiveness.

2. **Q: What is data leakage and how was it avoided in this project?**  
   *A:* Data leakage occurs when information from outside the training dataset is used to fit transformations. We performed stratified splitting *before* any imputation, outlier capping, scaling, or SMOTE resampling.

3. **Q: Why was SMOTE applied only to the training set?**  
   *A:* The test set must reflect true real-world class distribution. Applying SMOTE to the test set synthesizes artificial minority examples, invalidating empirical test evaluation.

4. **Q: How does the Flask API ensure consistency with training data?**  
   *A:* The API deserializes `best_churn_model.joblib`, which bundles the fitted imputer dictionary, IQR bounds, feature engineering transformer, and ColumnTransformer.
