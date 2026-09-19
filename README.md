# Customer Intelligence

> A production-oriented B2B Customer Intelligence platform that uses Machine Learning to analyze customer behavior, estimate churn risk, explain risk factors, and help business teams prioritize customer retention actions.

[🚀 **Live API**](https://customer-intelligence-ct1h.onrender.com) 

---

## Overview

**Customer Intelligence** is an end-to-end Machine Learning application designed for businesses to understand customer behavior, identify accounts that may be at risk of churn, and support data-driven customer retention decisions.

The platform combines customer data processing, Machine Learning, explainable risk analysis, REST API services, automated testing, and cloud deployment into a single B2B application.
The system combines:

- Customer data processing
- Exploratory Data Analysis
- Feature engineering
- Machine Learning
- Explainable churn-risk prediction
- REST API
- Automated testing
- Interactive web application
- Production deployment

The platform is designed for internal business teams such as:

- Customer Success
- Customer Retention
- Business Analysts
- Operations
- Account Managers
- Business Managers

Instead of showing only a raw prediction such as `Churn = Yes/No`, the system converts the prediction into a practical customer-risk view containing:

- Churn probability
- Risk level
- Risk drivers
- Protective signals
- Customer information
- Recommended prioritization context

---

## Key Features

### Customer Intelligence

View important information about individual customer accounts and understand their current profile.

### Explainable Risk

The system provides the main factors contributing to a customer's predicted churn risk rather than presenting only a probability.

### Customer Prioritization

Customers can be organized according to their predicted risk so business teams can focus attention on higher-risk accounts.

### Batch Analysis

Customer datasets can be processed to analyze multiple customer accounts.

### Search and Filtering

Customer records can be searched and filtered to locate relevant accounts quickly.

### Machine Learning Prediction

The platform uses trained Machine Learning models to estimate customer churn probability.

### REST API

The trained model is exposed through Flask API endpoints for programmatic predictions.

### Automated Testing

The backend includes automated tests covering API behavior and prediction functionality.

### Production Deployment

The backend is deployed using Gunicorn on Render.

---

# System Architecture

```text
                    ┌─────────────────────┐
                    │     Web Browser     │
                    │  Customer Intelligence
                    │       Frontend      │
                    └──────────┬──────────┘
                               │
                               │ HTTP
                               ▼
                    ┌─────────────────────┐
                    │     Flask REST API  │
                    │                     │
                    │ /health             │
                    │ /predict            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ ML Inference Layer  │
                    │                     │
                    │ Preprocessing       │
                    │ Feature Engineering │
                    │ Model               │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Customer Risk       │
                    │ Prediction          │
                    │                     │
                    │ Probability         │
                    │ Risk Tier           │
                    │ Risk Drivers        │
                    │ Protective Signals  │
                    └─────────────────────┘
```

---

# Machine Learning Pipeline

The Machine Learning workflow follows an end-to-end pipeline:

```text
Customer Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Train / Test Split
       │
       ▼
Preprocessing
       │
       ▼
Class Imbalance Handling
       │
       ▼
Model Training
       │
       ├───────────────┐
       ▼               ▼
Logistic Regression   Random Forest
       │               │
       └───────┬───────┘
               ▼
        Model Evaluation
               │
               ▼
        Final Model
               │
               ▼
        API Integration
               │
               ▼
       Customer Risk Prediction
```

---

# Machine Learning Models

## 1. Logistic Regression

Logistic Regression is used as the baseline classification model.

It provides a simple and interpretable reference point for evaluating the performance of the more advanced model.

## 2. Random Forest

Random Forest is used as the advanced classification model.

It combines multiple decision trees and can capture nonlinear relationships between customer characteristics and churn behavior.

The project evaluates both models using multiple classification metrics instead of relying only on accuracy.

---

# Model Evaluation

The following evaluation metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

These metrics provide a broader understanding of model performance, especially when the target classes are imbalanced.

---

# Feature Engineering

The project derives useful customer-level features from the available customer information.

Examples include:

### Average Order Value

```text
Average Order Value =
Total Spend / Number of Orders
```

### Customer Lifetime Value

```text
Customer Lifetime Value =
Average Order Value × Purchase Frequency × Customer Lifetime
```

### Support Intensity

```text
Support Intensity =
Support Tickets / Customer Lifetime
```

### Recency

Customer activity recency is used to understand how recently the customer interacted with the business.

These engineered variables allow the Machine Learning model to capture behavioral patterns more effectively.

---

# Data Processing

The Machine Learning workflow follows controlled preprocessing steps.

The project includes:

- Missing-value handling
- Numerical feature scaling
- Categorical feature encoding
- Train/test separation
- Class imbalance handling
- Feature transformation
- Model serialization

Preprocessing is applied consistently between model training and API inference.

---

# Leakage Prevention

Data leakage was explicitly reviewed during the development process.

The workflow ensures:

- Train/test split occurs before model preprocessing.
- Imputation parameters are learned from training data.
- Scaling parameters are learned from training data.
- Encoding is learned from training data.
- The final test set remains untouched until evaluation.
- The target variable is not used as an input feature.
- API inference uses the same preprocessing logic as training.

Class imbalance handling is applied only to training data.

For cross-validation workflows, imbalance handling should be placed inside the cross-validation pipeline so synthetic samples cannot influence validation folds.

---

# Customer Intelligence Output

For every customer prediction, the application can provide information such as:

```text
Customer
   │
   ├── Churn Probability
   │
   ├── Risk Tier
   │
   ├── Risk Drivers
   │
   └── Protective Signals
```

Example response structure:

```json
{
  "risk_tier": "Critical",
  "churn_probability": 0.80,
  "risk_drivers": [
    "Low customer activity",
    "High support interaction"
  ],
  "protective_signals": [
    "Recent purchase activity"
  ]
}
```

The exact prediction values depend on the customer data supplied to the model.

---

# REST API

The backend is implemented using Flask.

## Health Check

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:5000/health
```

The endpoint is used to verify that the API and model service are running correctly.

---

## Customer Prediction

```http
POST /predict
```

The endpoint accepts customer information and returns the predicted churn risk.

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "customer_data": {
    "example_feature": "example_value"
  }
}'
```

The exact request fields depend on the trained model's feature schema.

---

# Backend Application Structure

The Flask application uses an application factory:

```python
def create_app():
    ...
```

For production WSGI deployment, the application exposes:

```python
app = create_app()
```

This allows Gunicorn to load the Flask application using:

```text
gunicorn api.app:app
```

The application also reads the deployment port from the environment:

```python
port = int(os.environ.get("PORT", 5000))
```

---

# Testing

Automated tests are implemented using Pytest.

The latest backend verification completed successfully with:

```text
18 of 18 tests passed
```

The tests verify important API and prediction functionality including:

- Application startup
- Health endpoint
- Prediction endpoint
- Input validation
- Model loading
- Response structure
- Error handling

Run the tests locally with:

```bash
pytest -q
```

---

# Project Structure

```text
customer-intelligence/
│
├── api/
│   ├── app.py
│   └── ...
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── frontend/
│   └── ...
│
├── tests/
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

The exact contents may evolve as the application is developed.

---

# Technology Stack

## Programming

- Python
- JavaScript
- HTML
- CSS

## Machine Learning

- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- Joblib

## Backend

- Flask
- Gunicorn

## Testing

- Pytest

## Development

- Git
- GitHub

## Deployment

- Render

## Frontend

- HTML
- CSS
- JavaScript

---

# Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Chandrashekhar-cloud/customer-intelligence.git
cd customer-intelligence
```

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Tests

```bash
pytest -q
```

## 5. Start the Flask API

```bash
python api/app.py
```

The API will run locally on:

```text
http://127.0.0.1:5000
```

---

# Production Deployment

The Flask backend is deployed using Render and Gunicorn.

## Build Command

```bash
pip install -r requirements.txt
```

## Start Command

```bash
gunicorn api.app:app
```

Gunicorn loads the Flask application using:

```text
api.app:app
```

where:

- `api` is the Python package
- `app` is the Python module
- `app` is the Flask application object

The production server binds to the port provided by Render.

---

# Deployment Status

The backend deployment was successfully verified with:

```text
Deploy succeeded · Live
```

Gunicorn successfully starts the Flask application and listens on the Render-provided port.

The production API should be tested using:

```text
GET /health
```

and:

```text
POST /predict
```

---

# Frontend

The frontend is designed as a premium B2B SaaS-style application rather than a traditional academic dashboard.

The interface focuses on:

- Clean typography
- Black and white visual language
- Neutral surfaces
- Minimal visual noise
- Clear customer information
- Risk-focused workflows
- Responsive layouts
- Professional enterprise styling

The main application areas are:

```text
Landing Page
     │
     ▼
Authentication / Workspace
     │
     ▼
Customer Intelligence
     │
     ├── Overview
     │
     ├── Customers
     │
     ├── Assess
     │
     ├── Insights
     │
     └── System
```

---

# Product Design Principles

The application follows these principles:

### 1. Business First

The product should help users understand customer risk rather than simply display Machine Learning outputs.

### 2. Explainability

Predictions should provide understandable supporting information.

### 3. Minimalism

Only useful functionality is included. Unnecessary dashboard widgets and decorative elements are avoided.

### 4. Professional SaaS Experience

The application is designed to resemble a modern B2B software product rather than a basic college project.

### 5. Responsive Design

The interface should remain usable across desktop and mobile screen sizes.

---

# Limitations

The current project has several limitations:

- Customer data may not represent every real-world business environment.
- Synthetic or benchmark datasets can contain assumptions that do not necessarily generalize to production customers.
- Churn predictions are statistical estimates rather than guaranteed outcomes.
- Model performance depends on the quality and distribution of the training data.
- Business users should validate predictions against real customer context before taking action.
- The current system focuses on churn-risk prediction rather than complete customer lifecycle management.

---

# Future Improvements

Potential improvements include:

- SHAP-based model explanations
- Better model monitoring
- Model version tracking
- Automated CI/CD
- Authentication and role-based access
- Database-backed customer storage
- Historical customer-risk tracking
- Advanced customer segmentation
- More robust production monitoring
- Scheduled batch predictions
- Model drift detection
- Additional Machine Learning models
- Cloud-native scaling

---

# End-to-End Workflow

```text
1. Customer data is collected
             │
             ▼
2. Data is cleaned and analyzed
             │
             ▼
3. Features are engineered
             │
             ▼
4. Dataset is split into training and testing data
             │
             ▼
5. Preprocessing is applied
             │
             ▼
6. Machine Learning models are trained
             │
             ▼
7. Models are evaluated
             │
             ▼
8. Final model is serialized
             │
             ▼
9. Flask API loads the model
             │
             ▼
10. Frontend sends customer information
             │
             ▼
11. API generates churn-risk prediction
             │
             ▼
12. Customer Intelligence interface displays
    probability, risk tier, drivers and signals
```

---

# Repository

GitHub:

https://github.com/Chandrashekhar-cloud/customer-intelligence

---

# Author

**Chandrashekhar H S**

Computer Science Engineering  
AI & ML

---

# Project Summary

**Customer Intelligence** demonstrates an end-to-end Machine Learning product workflow, starting from customer data processing and model development and continuing through API development, automated testing, frontend integration, and cloud deployment.

The project combines Machine Learning with software engineering practices to transform customer data into an actionable customer-risk intelligence platform.

---

**Customer Intelligence · © 2026**

**Designed & developed by Chandrashekhar H S**
