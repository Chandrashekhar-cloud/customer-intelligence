"""
Milestone 2: Feature Engineering Module (Kaggle E-Commerce Dataset)
------------------------------------------------------------------
Implements domain-specific engineered features:
1. cashback_per_order: cashback_amount / (order_count + 1)
2. app_engagement: hour_spend_on_app * number_of_device_registered
3. complaint_friction: complain * (6 - satisfaction_score)
4. recency_ratio: day_since_last_order / (tenure + 1)
5. coupon_intensity: coupon_used / (order_count + 1)

Provides:
- Standalone feature calculation function
- Scikit-learn TransformerMixin class for pipeline serialization
- Unified ColumnTransformer for scaling and categorical encoding
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific engineered features on Kaggle customer data.
    """
    df_feat = df.copy()

    # 1. Cashback per Order (Promotional benefit yield)
    order_cnt = df_feat["order_count"].fillna(1.0)
    cashback = df_feat["cashback_amount"].fillna(0.0)
    df_feat["cashback_per_order"] = (cashback / (order_cnt + 1.0)).round(4)

    # 2. App Engagement (Time on app multiplied by devices)
    app_hours = df_feat["hour_spend_on_app"].fillna(2.0)
    devices = df_feat["number_of_device_registered"].fillna(2.0)
    df_feat["app_engagement"] = (app_hours * devices).round(4)

    # 3. Complaint Friction (Interaction of official complaints with low satisfaction)
    complain_val = df_feat["complain"].fillna(0.0)
    satisfaction = df_feat["satisfaction_score"].fillna(3.0)
    df_feat["complaint_friction"] = (complain_val * (6.0 - satisfaction)).round(4)

    # 4. Recency Ratio (Days since last purchase relative to tenure)
    days_since = df_feat["day_since_last_order"].fillna(3.0)
    tenure_val = df_feat["tenure"].fillna(1.0)
    df_feat["recency_ratio"] = (days_since / (tenure_val + 1.0)).round(4)

    # 5. Coupon Intensity (Coupon usage frequency per order)
    coupons = df_feat["coupon_used"].fillna(0.0)
    df_feat["coupon_intensity"] = (coupons / (order_cnt + 1.0)).round(4)

    return df_feat


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer dynamically adding engineered features.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return add_engineered_features(X)
        elif isinstance(X, dict):
            return add_engineered_features(pd.DataFrame([X]))
        else:
            return add_engineered_features(pd.DataFrame(X))


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Return exact lists of all numerical and categorical features.
    """
    numerical_features = [
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
        "cashback_amount",
        # Engineered:
        "cashback_per_order",
        "app_engagement",
        "complaint_friction",
        "recency_ratio",
        "coupon_intensity"
    ]

    categorical_features = [
        "preferred_login_device",
        "preferred_payment_mode",
        "gender",
        "preferred_order_cat",
        "marital_status"
    ]

    return numerical_features, categorical_features


def build_column_transformer() -> ColumnTransformer:
    """
    Construct ColumnTransformer standardizing numericals and one-hot encoding categoricals.
    """
    numerical_features, categorical_features = get_feature_lists()

    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features
            )
        ],
        remainder="drop"
    )
    return transformer


def create_preprocessor_transformer() -> ColumnTransformer:
    return build_column_transformer()
