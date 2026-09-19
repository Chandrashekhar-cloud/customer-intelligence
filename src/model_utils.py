"""
Milestone 3: Model Utilities Module
------------------------------------
Provides utility functions for:
- Model serialization and loading via joblib
- Evaluation metrics calculation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix)
- Saving model metadata JSON
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for binary classification.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    return metrics


from src.preprocessing import (
    clean_anomalies,
    transform_missing_values,
    apply_outlier_caps
)
from src.feature_engineering import add_engineered_features


class FullChurnInferencePipeline:
    """
    Self-contained inference pipeline combining fitted imputation, IQR capping,
    feature engineering, ColumnTransformer, and the final estimator.
    """
    def __init__(self, imputer_dict, cap_bounds, transformer, estimator, feature_names):
        self.imputer_dict = imputer_dict
        self.cap_bounds = cap_bounds
        self.transformer = transformer
        self.estimator = estimator
        self.feature_names = feature_names

    def preprocess_input(self, df_input: pd.DataFrame) -> np.ndarray:
        cleaned = clean_anomalies(df_input)
        imputed = transform_missing_values(cleaned, self.imputer_dict)
        capped = apply_outlier_caps(imputed, self.cap_bounds)
        engineered = add_engineered_features(capped)
        transformed = self.transformer.transform(engineered)
        return transformed

    def predict(self, df_input: pd.DataFrame) -> np.ndarray:
        X = self.preprocess_input(df_input)
        return self.estimator.predict(X)

    def predict_proba(self, df_input: pd.DataFrame) -> np.ndarray:
        X = self.preprocess_input(df_input)
        return self.estimator.predict_proba(X)


def save_pipeline(pipeline: Any, filepath: str) -> None:
    """Save pipeline or model object using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"[+] Model artifact successfully serialized to: {filepath}")


def load_pipeline(filepath: str) -> Any:
    """Load serialized pipeline or model object using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")
    return joblib.load(filepath)


def save_model_metadata(metadata: Dict[str, Any], filepath: str) -> None:
    """Save training metadata, hyperparameters, and metrics to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Model metadata successfully saved to: {filepath}")
