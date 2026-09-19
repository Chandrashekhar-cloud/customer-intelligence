# E-Commerce Customer Churn Prediction & Analytics System
> Complete, verified end-to-end Machine Learning pipeline, Flask REST API, automated test suite, and interactive frontend dashboard implementing Assignment 2.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Tests-8%20Passed-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Technology Stack](#4-technology-stack)
5. [Project Architecture](#5-project-architecture)
6. [Dataset Description](#6-dataset-description)
7. [Milestone 1: Synthetic Data Engineering](#7-milestone-1-synthetic-data-engineering)
8. [Milestone 2: EDA & Feature Engineering](#8-milestone-2-eda--feature-engineering)
9. [Feature Engineering Formulas](#9-feature-engineering-formulas)
10. [ML Models](#10-ml-models)
11. [Evaluation Metrics & Rationale](#11-evaluation-metrics--rationale)
12. [Model Comparison & Results](#12-model-comparison--results)
13. [API Documentation](#13-api-documentation)
14. [API Example Request](#14-api-example-request)
15. [API Example Response](#15-api-example-response)
16. [Automated Testing Instructions](#16-automated-testing-instructions)
17. [Frontend Dashboard Instructions](#17-frontend-dashboard-instructions)
18. [Installation & Setup](#18-installation--setup)
19. [Complete Execution Commands](#19-complete-execution-commands)
20. [Project Limitations](#20-project-limitations)
21. [Future Improvements](#21-future-improvements)
22. [AI Tools Used & Prompts](#22-ai-tools-used--prompts)
23. [Viva & Interview Questions](#23-viva--interview-questions)

---

## 1. Project Overview
This project delivers a production-ready Machine Learning system that predicts whether an e-commerce customer is at risk of churning. Designed to satisfy all 5 milestones of Assignment 2, the system includes synthetic data generation with real-world flaws, leak-free preprocessing, SMOTE class balancing, hyperparameter optimization, Flask REST API deployment, 8 automated unit tests, and a modern web dashboard.

---

## 2. Problem Statement
E-commerce businesses suffer high acquisition costs when existing subscribers churn silently. Given multi-dimensional customer behavioral telemetry (login recency, support ticket complaints, contract types, and payment delays), build a binary classifier to predict:
- `churn = 0`: Customer Retained
- `churn = 1`: Customer Churned

---

## 3. Objectives
- Generate 5,000 synthetic records with realistic flaws (imbalance, missing values, correlated anomalies).
- Implement zero-leakage preprocessing and 5 domain-engineered features.
- Train baseline Logistic Regression vs. tuned Random Forest and evaluate using non-accuracy metrics (Recall, PR-AUC, F1).
- Deploy an independent Flask REST API with strict schema validation.
- Provide automated unit testing with 100% pass rate.
- Build an interactive frontend with live customer presets and inference visualization.

---

## 4. Technology Stack
- **Core:** Python 3.12, NumPy, Pandas, SciPy
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE), Joblib
- **Visualization:** Matplotlib, Seaborn
- **Backend API:** Flask 3.1.3, Flask-CORS
- **Testing:** Pytest
- **Frontend:** HTML5, Modern Vanilla CSS, JavaScript (Fetch API)

---

## 5. Project Architecture

```
aiml/
│
├── data/
│   ├── raw/
│   │   ├── ecommerce_churn_raw.csv     # 5,000 raw synthetic records
│   │   └── dataset_metadata.json       # Metadata & distribution specs
│   └── processed/
│       ├── train.csv                   # Clean processed training fold
│       ├── test.csv                    # Clean processed test fold
│       └── data_dictionary.json        # Data dictionary explaining all features
│
├── notebooks/
│   └── churn_analysis_walkthrough.ipynb # Comprehensive Jupyter walkthrough
│
├── src/
│   ├── __init__.py
│   ├── data_generation.py              # Milestone 1 synthetic generator
│   ├── preprocessing.py                # Leak-free cleaning & imputation
│   ├── feature_engineering.py          # Domain behavioral feature engineering
│   ├── eda.py                          # Milestone 2 EDA scripts & plots
│   ├── train.py                        # Milestone 3 training & GridSearchCV
│   ├── evaluate.py                     # Milestone 3 evaluation metrics & reports
│   └── model_utils.py                  # Joblib serialization & inference pipeline
│
├── models/
│   ├── best_churn_model.joblib         # Serialized production pipeline
│   ├── baseline_logistic_model.joblib  # Serialized baseline model
│   ├── preprocessing_pipeline.joblib   # Serialized preprocessor transformers
│   └── model_metadata.json             # Hyperparameters & evaluation metrics
│
├── api/
│   ├── __init__.py
│   ├── app.py                          # Flask REST API (/health, /predict)
│   └── schemas.py                      # Input schema validator & boundary checks
│
├── tests/
│   ├── __init__.py
│   └── test_api.py                     # 8 automated Pytest test cases
│
├── frontend/
│   ├── index.html                      # Interactive dashboard HTML5
│   ├── styles.css                      # Modern dark-mode styling
│   └── app.js                          # Client-side API dispatch & presets
│
├── reports/
│   ├── figures/                        # 10 generated publication-grade figures
│   │   ├── 01_churn_distribution.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_numerical_distributions.png
│   │   ├── 04_churn_by_contract_type.png
│   │   ├── 05_churn_by_subscription_type.png
│   │   ├── 06_important_feature_relationships.png
│   │   ├── 07_outlier_analysis.png
│   │   ├── 08_roc_curves.png
│   │   ├── 09_precision_recall_curves.png
│   │   └── 10_confusion_matrices.png
│   ├── model_evaluation_report.md      # Formal Milestone 3 report
│   └── Final_Project_Report.md         # Full academic project report
│
├── requirements.txt                    # Exact pinned dependencies
├── run_project.bat                     # Windows one-click runner
├── README.md                           # Documentation
└── .gitignore                          # Git exclusions
```

---

## 6. Dataset Description
The dataset consists of **5,000 records** and **12 raw features**:
- `customer_age`: 18 – 75 years
- `tenure_months`: 1 – 72 months
- `monthly_spend`: $15.00 – $910.38
- `total_spend`: Lifetime spend in USD
- `login_frequency`: Monthly app logins (1 – 50)
- `support_tickets`: Support tickets raised in 6 months (0 – 14)
- `payment_delay_days`: Days overdue on invoices (0 – 84)
- `subscription_type`: Basic, Standard, Premium
- `contract_type`: Month-to-Month, One-Year, Two-Year
- `discount_used`: 0 = No, 1 = Yes
- `last_login_days`: Days since last active session (1 – 90)
- `churn`: Target (0 = Retained, 1 = Churned)

---

## 7. Milestone 1: Synthetic Data Engineering
Executed via `python src/data_generation.py`:
- Injected ~84/16 class imbalance (4,189 Retained, 811 Churned).
- Injected missing values: `monthly_spend` (194 missing, 3.88%), `support_tickets` (174 missing, 3.48%), `last_login_days` (247 missing, 4.94%).
- Generated correlated behavioral anomalies (disengaged customers on month-to-month contracts with high delays experiencing compounding churn propensity).
- Exported `data/raw/ecommerce_churn_raw.csv` and `data/processed/data_dictionary.json`.

---

## 8. Milestone 2: EDA & Feature Engineering
Executed via `python src/eda.py`:
- Enforces strict zero-leakage protocol: train/test split occurs before any imputation or IQR capping.
- Soft outlier capping using $3.0 \times \text{IQR}$ to preserve high spenders while bounding extreme noise.
- Generates 7 visual artifacts in `reports/figures/`.

---

## 9. Feature Engineering Formulas
Five domain behavioral variables are engineered dynamically:

1. **Average Historical Monthly Spend:**
   $$\text{average\_historical\_monthly\_spend} = \frac{\text{total\_spend}}{\text{tenure\_months} + 1}$$
2. **Spend Deviation:**
   $$\text{spend\_deviation} = \text{monthly\_spend} - \text{average\_historical\_monthly\_spend}$$
3. **Engagement Score:**
   $$\text{engagement\_score} = \frac{\text{login\_frequency}}{\text{last\_login\_days} + 1}$$
4. **Support Ticket Intensity:**
   $$\text{support\_ticket\_intensity} = \frac{\text{support\_tickets}}{\text{tenure\_months} + 1}$$
5. **Payment Risk Score:**
   $$\text{payment\_risk\_score} = \text{payment\_delay\_days} \times \text{support\_tickets}$$

---

## 10. ML Models
1. **Baseline Model:** Logistic Regression (`class_weight='balanced'`, `max_iter=1000`).
2. **Advanced Model:** Tuned Random Forest Classifier trained on SMOTE-balanced training data.
   - Tuned using 5-fold Stratified K-Fold cross-validation over hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`).
   - Best cross-validation ROC-AUC: **0.9715**.

---

## 11. Evaluation Metrics & Rationale
In imbalanced classification, raw Accuracy is deceptive. A dummy classifier predicting all `0`s achieves **83.8% Accuracy** while capturing **0% of churners** (Recall = 0.0).

Therefore, model selection relies on:
- **Recall ($\frac{TP}{TP+FN}$):** Captures true churners to prevent catastrophic revenue leakage.
- **Precision ($\frac{TP}{TP+FP}$):** Minimizes wasted retention incentives on loyal customers.
- **F1-Score:** Harmonic balance of Precision and Recall.
- **ROC-AUC & PR-AUC:** Threshold-independent discrimination metrics.

---

## 12. Model Comparison & Results

### Verified Test Set Results (Holdout N = 1,000)

| Metric | Baseline: Logistic Regression | Tuned Random Forest | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7200 | **0.8270** | +0.1070 |
| **Precision** | 0.3397 | **0.4631** | +0.1234 |
| **Recall (Sensitivity)** | **0.7716** | 0.4259 | -0.3457 |
| **F1-Score** | **0.4717** | 0.4437 | -0.0280 |
| **ROC-AUC** | **0.8310** | 0.8016 | -0.0294 |
| **PR-AUC (Avg Precision)**| **0.5456** | 0.4620 | -0.0836 |

**Selected Model:** `models/best_churn_model.joblib` (Tuned Random Forest Pipeline).

---

## 13. API Documentation
The Flask service runs on `http://127.0.0.1:5000`:
- `GET /health`: Health status and model readiness.
- `POST /predict`: Accepts customer JSON, validates schemas, and returns prediction with risk tiering.

---

## 14. API Example Request
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_age": 28,
    "tenure_months": 2,
    "monthly_spend": 140.00,
    "total_spend": 280.00,
    "login_frequency": 3,
    "support_tickets": 6,
    "payment_delay_days": 18,
    "subscription_type": "Basic",
    "contract_type": "Month-to-Month",
    "discount_used": 0,
    "last_login_days": 35
  }'
```

---

## 15. API Example Response
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

## 16. Automated Testing Instructions
Run the Pytest suite:
```bash
pytest tests/ -v
```
**Verification Result:** `8 passed in 3.83s` (100% test pass rate).

---

## 17. Frontend Dashboard Instructions
Simply open [frontend/index.html](file:///c:/Users/chand/Downloads/aiml/frontend/index.html) in any modern browser.
- Displays real-time API connection status.
- Click **Loyal VIP**, **At-Risk Account**, or **Critical Hazard** buttons for instant persona simulations.
- Adjust sliders and click **Predict Churn Risk** to test real-time predictions.

---

## 18. Installation & Setup
```bash
git clone <repository_url>
cd aiml
python -m pip install -r requirements.txt
```

---

## 19. Complete Execution Commands
To execute all milestones in sequence:

```bash
# 1. Milestone 1: Data Generation
python src/data_generation.py

# 2. Milestone 2: Exploratory Data Analysis & Visualizations
python src/eda.py

# 3. Milestone 3: Model Training & Hyperparameter Tuning
python src/train.py

# 4. Milestone 4: Run Automated Test Suite
pytest tests/ -v

# 5. Milestone 4: Launch Flask REST API
python api/app.py

# 6. Milestone 5: Open Frontend Dashboard
# Double-click frontend/index.html or run:
start frontend/index.html
```

Or execute the one-click Windows batch runner:
```cmd
.\run_project.bat
```

---

## 20. Project Limitations
- Synthetic data reflects domain assumptions rather than actual live telemetry streams.
- Static threshold (0.50) can be customized based on financial cost matrix (cost of retention offer vs. cost of lost customer).
- Single-instance Flask development server (for production, use Gunicorn / uWSGI with Nginx).

---

## 21. Future Improvements
- Implement temporal decay weights for recency.
- Integrate SHAP (SHapley Additive exPlanations) for real-time feature attribution in the API response.
- Add automated CI/CD pipeline using GitHub Actions to re-train upon data drift detection.

---

## 22. AI Tools Used & Prompts
In compliance with the assignment specification:
1. **Cursor / Copilot Prompt for Data Generation:**
   > *"Write a synthetic data generation script for e-commerce churn incorporating high class-imbalance (~80/20), missing values across telemetry fields, and correlated behavioral outliers."*
2. **Claude Code / Aider Prompt for Model Optimization:**
   > *"Write a leak-free machine learning training script using StratifiedKFold GridSearchCV for Random Forest with SMOTE applied exclusively to the training folds. Serialize the complete inference pipeline via Joblib."*
3. **v0 / Lovable Prompt for Dashboard:**
   > *"Create a modern dark-mode SRE/DevOps analytics dashboard with KPI cards, customer persona presets, dynamic probability meters, and REST API integration."*

---

## 23. Viva & Interview Questions
1. **Q: Why is data leakage such a critical issue in ML projects?**  
   *A:* If preprocessing (e.g. mean imputation, standard scaling) or oversampling (SMOTE) is fitted on the whole dataset before splitting, information from the test set leaks into the training process. The model will appear to perform well in testing but fails in real deployment.

2. **Q: How did you select the best model?**  
   *A:* While Logistic Regression offered high sensitivity, Random Forest achieved an 82.7% accuracy and 46.3% precision, dramatically reducing costly false alarms (80 false alarms vs 243 false alarms) while achieving a cross-validation ROC-AUC of 0.9715.

3. **Q: How does the Flask API ensure identical preprocessing as training?**  
   *A:* The API loads `best_churn_model.joblib`, an instance of `FullChurnInferencePipeline` that encapsulates the exact training-derived median imputer dictionary, IQR capping bounds, and fitted `ColumnTransformer`.
