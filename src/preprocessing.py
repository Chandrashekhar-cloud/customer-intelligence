"""
Milestone 2: Data Preprocessing Module (Kaggle E-Commerce Dataset)
------------------------------------------------------------------
Provides leakage-free functions for:
- Detecting and repairing domain glitches
- Imputing missing values using train-fitted medians and modes
- Calculating and applying IQR-based outlier capping without data leakage
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin


NUMERICAL_COLS = [
    "tenure",
    "city_tier",
    "warehouse_to_home",
    "hour_spend_on_app",
    "number_of_device_registered",
    "satisfaction_score",
    "number_of_address",
    "complain",
    "order_amount_hike_from_last_year",
    "coupon_used",
    "order_count",
    "day_since_last_order",
    "cashback_amount"
]

CATEGORICAL_COLS = [
    "preferred_login_device",
    "preferred_payment_mode",
    "gender",
    "preferred_order_cat",
    "marital_status"
]


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize domain representations and clamp invalid negative values.
    """
    cleaned = df.copy()

    # Normalize categorical aliases if present
    if "preferred_login_device" in cleaned.columns:
        cleaned["preferred_login_device"] = cleaned["preferred_login_device"].replace({"Phone": "Mobile Phone"})
    if "preferred_payment_mode" in cleaned.columns:
        cleaned["preferred_payment_mode"] = cleaned["preferred_payment_mode"].replace({
            "CC": "Credit Card",
            "COD": "Cash on Delivery"
        })
    if "preferred_order_cat" in cleaned.columns:
        cleaned["preferred_order_cat"] = cleaned["preferred_order_cat"].replace({"Mobile": "Mobile Phone"})

    # Non-negative constraints for counts & distances
    for col in ["tenure", "warehouse_to_home", "hour_spend_on_app", "coupon_used", "order_count", "day_since_last_order", "cashback_amount"]:
        if col in cleaned.columns:
            cleaned.loc[cleaned[col] < 0, col] = 0

    return cleaned


def fit_imputation_parameters(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute medians for numericals and modes for categoricals
    strictly from TRAINING data to prevent data leakage.
    """
    imputer_dict = {}

    for col in NUMERICAL_COLS:
        if col in train_df.columns:
            imputer_dict[col] = float(train_df[col].median(skipna=True))

    for col in CATEGORICAL_COLS:
        if col in train_df.columns:
            mode_val = train_df[col].mode(dropna=True)
            imputer_dict[col] = str(mode_val.iloc[0]) if not mode_val.empty else "Missing"

    return imputer_dict


def transform_missing_values(df: pd.DataFrame, imputer_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Impute missing values using training statistics.
    """
    imputed = df.copy()
    for col, fill_val in imputer_dict.items():
        if col in imputed.columns:
            imputed[col] = imputed[col].fillna(fill_val)
    return imputed


def fit_outlier_caps(
    train_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 3.0
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate IQR upper and lower bounds on training data.
    Uses 3.0 * IQR to bound extreme noise while preserving genuine behavioral extremes.
    """
    if columns is None:
        columns = ["warehouse_to_home", "cashback_amount", "day_since_last_order"]

    cap_bounds = {}
    for col in columns:
        if col in train_df.columns:
            q25 = float(train_df[col].quantile(0.25))
            q75 = float(train_df[col].quantile(0.75))
            iqr = q75 - q25
            lower_bound = max(0.0, q25 - (iqr_multiplier * iqr))
            upper_bound = q75 + (iqr_multiplier * iqr)
            cap_bounds[col] = (round(lower_bound, 4), round(upper_bound, 4))

    return cap_bounds


def apply_outlier_caps(df: pd.DataFrame, cap_bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Apply soft capping using the bounds fitted on training data.
    """
    capped = df.copy()
    for col, (lower_b, upper_b) in cap_bounds.items():
        if col in capped.columns:
            capped[col] = capped[col].clip(lower=lower_b, upper=upper_b)
    return capped


class AnomalyCleaner(BaseEstimator, TransformerMixin):
    """
    Stateless anomaly detection and domain standardization transformer.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return clean_anomalies(X)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Leak-free Scikit-Learn transformer for median/mode imputation.
    """
    def __init__(self):
        self.imputer_dict_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.imputer_dict_ = fit_imputation_parameters(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return transform_missing_values(X, self.imputer_dict_)


class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """
    Leak-free Scikit-Learn transformer for IQR outlier capping.
    """
    def __init__(self, columns: Optional[List[str]] = None, iqr_multiplier: float = 3.0):
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier
        self.cap_bounds_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.cap_bounds_ = fit_outlier_caps(X, self.columns, self.iqr_multiplier)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return apply_outlier_caps(X, self.cap_bounds_)
