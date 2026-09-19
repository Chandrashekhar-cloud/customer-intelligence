"""
Explainability Engine for E-Commerce Churn Intelligence
-------------------------------------------------------
Implements exact Tree-Path Feature Contribution Decomposition (Saabas additive tree attribution)
directly on the trained scikit-learn RandomForestClassifier.

Decomposes prediction:
    Predicted Probability = Bias (Base Rate) + Sum(Feature Contributions)

Categorizes contributions into:
- Risk Drivers: positive contributions pushing churn probability higher
- Protective Signals: negative contributions lowering churn probability / anchoring retention
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def get_feature_names(pipeline) -> List[str]:
    """Retrieve ordered feature names after ColumnTransformer preprocessing."""
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_cols = [
        "preferred_login_device",
        "preferred_payment_mode",
        "gender",
        "preferred_order_cat",
        "marital_status"
    ]
    num_cols = list(preprocessor.transformers_[0][2])
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    return num_cols + cat_names


def format_feature_signal(feature_name: str, raw_val: Any, contribution: float) -> str:
    """Format a feature and its value into an intuitive B2B customer intelligence statement."""
    is_risk = contribution > 0

    if feature_name == "complain":
        if raw_val == 1 or raw_val == 1.0:
            return "Active unresolved customer complaint logged"
        return "Clean customer support record (zero complaints)"

    if feature_name == "complaint_friction":
        return "Elevated complaint-to-satisfaction friction" if is_risk else "Low complaint friction index"

    if feature_name == "day_since_last_order":
        days = int(round(float(raw_val))) if raw_val is not None else 0
        if is_risk:
            return f"Prolonged order inactivity ({days} days dormant)"
        return f"Recent order activity ({days} days since last order)"

    if feature_name == "tenure":
        ten = int(round(float(raw_val))) if raw_val is not None else 0
        if is_risk:
            return f"Early lifecycle account tenure ({ten} months)"
        return f"Established loyal customer tenure ({ten} months)"

    if feature_name == "satisfaction_score":
        sat = int(round(float(raw_val))) if raw_val is not None else 3
        if is_risk:
            return f"Depressed CSAT satisfaction rating ({sat}/5)"
        return f"Favorable CSAT satisfaction rating ({sat}/5)"

    if feature_name == "cashback_amount":
        amt = float(raw_val) if raw_val is not None else 0.0
        if is_risk:
            return f"Below-average promotional cashback yield (${amt:.2f})"
        return f"Consistent loyalty cashback reward yield (${amt:.2f})"

    if feature_name == "cashback_per_order":
        amt = float(raw_val) if raw_val is not None else 0.0
        if is_risk:
            return f"Low cashback reward per purchase (${amt:.2f})"
        return f"High average cashback incentive per order (${amt:.2f})"

    if feature_name == "order_amount_hike_from_last_year":
        pct = float(raw_val) if raw_val is not None else 0.0
        if is_risk:
            return f"Stagnant purchase expansion (+{pct:.0f}% year-over-year)"
        return f"Strong purchase expansion (+{pct:.0f}% year-over-year)"

    if feature_name == "order_count":
        cnt = int(round(float(raw_val))) if raw_val is not None else 1
        if is_risk:
            return f"Low total order frequency ({cnt} lifetime orders)"
        return f"Established purchase order history ({cnt} orders)"

    if feature_name == "warehouse_to_home":
        dist = int(round(float(raw_val))) if raw_val is not None else 10
        if is_risk:
            return f"High fulfillment shipping distance ({dist} km)"
        return f"Proximity to fulfillment center ({dist} km)"

    if feature_name == "city_tier":
        tier = int(round(float(raw_val))) if raw_val is not None else 1
        return f"Tier {tier} metropolitan market profile"

    if feature_name == "hour_spend_on_app":
        hrs = float(raw_val) if raw_val is not None else 2.0
        if is_risk:
            return f"Low daily platform engagement ({hrs:.1f} hrs/day)"
        return f"Active daily platform screen time ({hrs:.1f} hrs/day)"

    if feature_name == "recency_ratio":
        return "High inactivity period relative to tenure" if is_risk else "Healthy order recency ratio"

    if feature_name == "app_engagement":
        return "Low device engagement ratio" if is_risk else "High multi-device engagement ratio"

    if feature_name == "coupon_used":
        coupons = int(round(float(raw_val))) if raw_val is not None else 0
        return f"Low coupon redemption frequency ({coupons} used)" if is_risk else f"Active coupon redemption ({coupons} used)"

    if feature_name.startswith("preferred_order_cat_"):
        cat = feature_name.replace("preferred_order_cat_", "")
        if is_risk:
            return f"Elevated churn propensity in {cat} category"
        return f"High retention affinity in {cat} category"

    if feature_name.startswith("preferred_payment_mode_"):
        mode = feature_name.replace("preferred_payment_mode_", "")
        if is_risk:
            return f"Payment method friction ({mode})"
        return f"Reliable payment method affinity ({mode})"

    if feature_name.startswith("preferred_login_device_"):
        dev = feature_name.replace("preferred_login_device_", "")
        return f"Primary access device: {dev}"

    if feature_name.startswith("marital_status_"):
        status = feature_name.replace("marital_status_", "")
        return f"Demographic segment: {status}"

    if feature_name.startswith("gender_"):
        g = feature_name.replace("gender_", "")
        return f"Customer segment ({g})"

    clean_name = feature_name.replace("_", " ").title()
    return f"{clean_name} influence"


def compute_tree_contributions(pipeline, df_sample: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute exact tree-path feature contributions for a single sample.

    Returns:
    --------
    (predicted_probability, risk_drivers, protective_signals)
    """
    X = df_sample.copy()
    for step_name, step in pipeline.steps:
        if step_name in ["cleaner", "imputer", "capper", "features"]:
            X = step.transform(X)
        elif step_name == "preprocessor":
            X_trans = step.transform(X)
            break

    rf = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)
    n_features = X_trans.shape[1]

    X_float = X_trans.astype(np.float32)
    contributions = np.zeros(n_features)

    for estimator in rf.estimators_:
        tree = estimator.tree_
        node_indicator = tree.decision_path(X_float)
        node_indices = node_indicator.indices
        values = tree.value[:, 0, 1] / np.sum(tree.value[:, 0], axis=1)

        for i in range(len(node_indices) - 1):
            curr_node = node_indices[i]
            next_node = node_indices[i + 1]
            feat = tree.feature[curr_node]
            contributions[feat] += (values[next_node] - values[curr_node])

    contributions /= len(rf.estimators_)
    prob = float(rf.predict_proba(X_trans)[0, 1])

    risk_drivers = []
    protective_signals = []

    raw_dict = df_sample.iloc[0].to_dict()

    for idx, (feat_name, cont) in enumerate(zip(feature_names, contributions)):
        if abs(cont) < 0.004:
            continue

        raw_val = raw_dict.get(feat_name, None)
        label = format_feature_signal(feat_name, raw_val, cont)

        item = {
            "feature": feat_name,
            "label": label,
            "contribution": float(round(cont, 4)),
            "impact_percent": float(round(cont * 100, 1))
        }

        if cont > 0:
            risk_drivers.append(item)
        else:
            protective_signals.append(item)

    risk_drivers = sorted(risk_drivers, key=lambda x: x["contribution"], reverse=True)
    protective_signals = sorted(protective_signals, key=lambda x: x["contribution"])

    return prob, risk_drivers[:4], protective_signals[:3]


def determine_risk_tier(prob: float) -> Tuple[str, str, str]:
    """
    Derive standardized 4-tier enterprise priority from real ML churn probability.

    Returns:
    --------
    (risk_tier, prediction_label, recommendation)
    """
    # Prediction label aligned with binary/multi-tier contract
    if prob >= 0.50:
        label = "Likely to Churn"
    elif prob >= 0.35:
        label = "Moderate Risk"
    else:
        label = "Customer Retained"

    if prob >= 0.65:
        tier = "Critical"
        rec = "Critical risk account: Immediate proactive outreach required. Dispatch dedicated CSM, review open complaints, and offer targeted retention incentives."
    elif prob >= 0.40:
        tier = "High Risk"
        rec = "High churn vulnerability: Initiate targeted re-engagement campaign, survey CSAT friction points, and deliver promotional loyalty perks within 48h."
    elif prob >= 0.18:
        tier = "Watch"
        rec = "Watchlist account: Monitor order recency and browsing telemetry. Trigger automated check-in and satisfaction follow-up."
    else:
        tier = "Healthy"
        rec = "Healthy retention baseline: Account stable. Eligible for VIP tier loyalty perks, cross-sell recommendations, and product expansion."

    return tier, label, rec

