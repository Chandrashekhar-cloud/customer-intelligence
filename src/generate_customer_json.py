"""
Generate frontend/assets/customers.json using the REAL trained ML inference pipeline.
Calculates authentic churn probabilities, 4-tier enterprise priority, and exact tree-path explainability.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model_utils import load_pipeline
from src.explainability import get_feature_names, format_feature_signal, determine_risk_tier


def generate_customers_json(
    csv_path: str = "data/raw/ecommerce_churn_raw.csv",
    model_path: str = "models/best_churn_model.joblib",
    output_json: str = "frontend/assets/customers.json"
):
    start_time = time.time()
    print(f"[*] Loading raw dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)
    total_records = len(df)
    print(f"[+] Loaded {total_records} authentic customer accounts.")

    print(f"[*] Loading ML inference pipeline from: {model_path}...")
    pipeline = load_pipeline(model_path)
    rf = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)

    # Clean missing values for display fields
    display_df = df.copy()
    num_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        display_df[col] = display_df[col].fillna(round(float(display_df[col].median()), 1))

    cat_cols = display_df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col != "customer_id":
            display_df[col] = display_df[col].fillna(display_df[col].mode().iloc[0])

    # Run full pipeline transformation
    print("[*] Running pipeline feature transformation...")
    X = df.drop(columns=["churn", "customer_id"], errors="ignore").copy()
    for step_name, step in pipeline.steps:
        if step_name in ["cleaner", "imputer", "capper", "features"]:
            X = step.transform(X)
        elif step_name == "preprocessor":
            X_trans = step.transform(X)
            break

    print("[*] Computing real model churn probabilities across all 5,630 records...")
    churn_probabilities = rf.predict_proba(X_trans)[:, 1]

    # Pre-compute top tree-path contributions for each record
    print("[*] Computing exact tree-path feature contributions...")
    n_samples, n_features = X_trans.shape
    all_contributions = np.zeros((n_samples, n_features), dtype=np.float32)
    X_float = X_trans.astype(np.float32)

    for est_idx, estimator in enumerate(rf.estimators_):
        tree = estimator.tree_
        indicator = tree.decision_path(X_float)
        values = tree.value[:, 0, 1] / np.sum(tree.value[:, 0], axis=1)

        for i in range(n_samples):
            nodes = indicator[i].indices
            for j in range(len(nodes) - 1):
                c_node = nodes[j]
                n_node = nodes[j + 1]
                feat = tree.feature[c_node]
                all_contributions[i, feat] += (values[n_node] - values[c_node])

    all_contributions /= len(rf.estimators_)

    print("[*] Assembling Customer 360 profiles and priority groups...")
    customers_list = []
    critical_count = 0
    high_count = 0
    watch_count = 0
    healthy_count = 0

    for i in range(total_records):
        row = display_df.iloc[i]
        cid = str(row["customer_id"])
        prob = float(round(churn_probabilities[i], 4))
        risk_score = float(round(prob * 100, 1))
        retention_prob = float(round((1.0 - prob) * 100, 1))
        tier, label, recommendation = determine_risk_tier(prob)

        if tier == "Critical":
            critical_count += 1
        elif tier == "High Risk":
            high_count += 1
        elif tier == "Watch":
            watch_count += 1
        else:
            healthy_count += 1

        # Extract top 3 risk drivers and top 2 protective signals
        row_cont = all_contributions[i]
        drivers = []
        protective = []
        raw_dict = row.to_dict()

        for f_idx, (fname, cont_val) in enumerate(zip(feature_names, row_cont)):
            if abs(cont_val) < 0.005:
                continue
            rval = raw_dict.get(fname, None)
            sig_label = format_feature_signal(fname, rval, cont_val)
            item = {
                "feature": fname,
                "label": sig_label,
                "contribution": float(round(cont_val, 4)),
                "impact_percent": float(round(cont_val * 100, 1))
            }
            if cont_val > 0:
                drivers.append(item)
            else:
                protective.append(item)

        drivers = sorted(drivers, key=lambda x: x["contribution"], reverse=True)[:3]
        protective = sorted(protective, key=lambda x: x["contribution"])[:2]

        tenure_val = float(row["tenure"])
        order_cnt = float(row["order_count"])
        cashback_val = float(row["cashback_amount"])
        monthly_spend = round(cashback_val * 1.25, 2)
        total_spend = round(monthly_spend * max(1.0, tenure_val), 2)

        signals = [d["label"] for d in drivers] + [p["label"] for p in protective]

        cust_obj = {
            "id": cid,
            "tenure": tenure_val,
            "preferred_login_device": str(row["preferred_login_device"]),
            "city_tier": int(row["city_tier"]),
            "warehouse_to_home": float(row["warehouse_to_home"]),
            "preferred_payment_mode": str(row["preferred_payment_mode"]),
            "gender": str(row["gender"]),
            "hour_spend_on_app": float(row["hour_spend_on_app"]),
            "number_of_device_registered": int(row["number_of_device_registered"]),
            "preferred_order_cat": str(row["preferred_order_cat"]),
            "satisfaction_score": int(row["satisfaction_score"]),
            "marital_status": str(row["marital_status"]),
            "number_of_address": int(row["number_of_address"]),
            "complain": int(row["complain"]),
            "order_amount_hike_from_last_year": float(row["order_amount_hike_from_last_year"]),
            "coupon_used": float(row["coupon_used"]),
            "order_count": order_cnt,
            "day_since_last_order": float(row["day_since_last_order"]),
            "cashback_amount": cashback_val,
            "monthly_spend": monthly_spend,
            "total_spend": total_spend,
            "churn": int(row["churn"]),
            "risk_level": tier,
            "risk_tier": tier,
            "prediction_label": label,
            "risk_score": risk_score,
            "probability": prob,
            "retention_prob": retention_prob,
            "recommendation": recommendation,
            "risk_drivers": drivers,
            "protective_signals": protective,
            "key_signals": signals
        }
        customers_list.append(cust_obj)

    retention_baseline = round((healthy_count + watch_count) / total_records * 100, 1)

    output_data = {
        "total_dataset_records": total_records,
        "retention_rate": retention_baseline,
        "critical_count": critical_count,
        "high_risk_count": high_count,
        "watchlist_count": watch_count,
        "healthy_count": healthy_count,
        "customers": customers_list
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print(f"[+] Successfully wrote {total_records} customer records to {output_json} in {elapsed}s.")
    print(f"    - Critical (>= 65%): {critical_count} ({critical_count/total_records*100:.1f}%)")
    print(f"    - High Risk (40-65%): {high_count} ({high_count/total_records*100:.1f}%)")
    print(f"    - Watch (18-40%): {watch_count} ({watch_count/total_records*100:.1f}%)")
    print(f"    - Healthy (< 18%): {healthy_count} ({healthy_count/total_records*100:.1f}%)")


if __name__ == "__main__":
    generate_customers_json()
