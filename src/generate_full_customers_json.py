"""
Utility script to generate frontend/assets/customers.json with all 5,000 dataset records.
Applies the trained ML inference pipeline to compute real risk scores, risk levels, and retention probabilities.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.model_utils import load_pipeline


def generate_full_customers():
    raw_csv_path = ROOT_DIR / "data" / "raw" / "ecommerce_churn_raw.csv"
    model_path = ROOT_DIR / "models" / "best_churn_model.joblib"
    output_json_path = ROOT_DIR / "frontend" / "assets" / "customers.json"

    print(f"[*] Reading raw dataset from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    total_records = len(df)
    print(f"[+] Total records read: {total_records}")

    print(f"[*] Loading ML model from: {model_path}")
    model = load_pipeline(str(model_path))

    # Drop target column for inference
    X = df.drop(columns=["churn"])

    print("[*] Running batch ML inference on all records...")
    churn_probabilities = model.predict_proba(X)[:, 1]

    # Preprocess missing values cleanly for UI presentation
    imputed_df = df.copy()
    for col in ["monthly_spend", "support_tickets", "discount_used", "last_login_days"]:
        if col in imputed_df.columns:
            median_val = imputed_df[col].median()
            imputed_df[col] = imputed_df[col].fillna(median_val)

    customers_list = []

    healthy_count = 0
    watchlist_count = 0
    high_risk_count = 0

    for i in range(total_records):
        row = imputed_df.iloc[i]
        prob = float(churn_probabilities[i])
        risk_score = round(prob * 100, 1)
        retention_prob = round((1.0 - prob) * 100, 1)

        if prob >= 0.65:
            risk_level = "High Risk"
            high_risk_count += 1
        elif prob >= 0.40:
            risk_level = "Watchlist"
            watchlist_count += 1
        else:
            risk_level = "Healthy"
            healthy_count += 1

        cust_id = f"CUST-{1001 + i}"
        age = int(row["customer_age"])
        tenure = int(row["tenure_months"])
        monthly_spend = round(float(row["monthly_spend"]), 2)
        total_spend = round(float(row["total_spend"]), 2)
        logins = int(row["login_frequency"])
        tickets = int(row["support_tickets"])
        delay = int(row["payment_delay_days"])
        sub_type = str(row["subscription_type"])
        contract_type = str(row["contract_type"])
        discount_used = int(row["discount_used"])
        last_login_days = int(row["last_login_days"])

        # Determine key behavioral signals based on actual metrics
        signals = []
        if contract_type == "Month-to-Month":
            signals.append("Month-to-Month contract increases volatility")
        elif contract_type == "Two-Year":
            signals.append("Two-Year committed agreement anchors retention")

        if delay >= 10:
            signals.append(f"Severe payment delinquency ({delay} days overdue)")
        elif delay >= 5:
            signals.append(f"Moderate payment delay ({delay} days)")
        elif delay == 0:
            signals.append("Punctual payment history")

        if tickets >= 4:
            signals.append(f"Elevated support ticket volume ({tickets} cases)")
        elif tickets == 0:
            signals.append("Zero support inquiries logged")

        if last_login_days >= 30:
            signals.append(f"Prolonged inactivity ({last_login_days} days dormant)")
        elif last_login_days <= 3:
            signals.append("Active daily platform engagement")

        if monthly_spend >= 120:
            signals.append(f"High-tier expenditure (${monthly_spend:.2f}/mo)")

        if tenure <= 3:
            signals.append("Early onboarding tenure (< 3 months)")
        elif tenure >= 24:
            signals.append(f"Established loyal tenure ({tenure} months)")

        customers_list.append({
            "id": cust_id,
            "customer_age": age,
            "tenure_months": tenure,
            "monthly_spend": monthly_spend,
            "total_spend": total_spend,
            "login_frequency": logins,
            "support_tickets": tickets,
            "payment_delay_days": delay,
            "subscription_type": sub_type,
            "contract_type": contract_type,
            "discount_used": discount_used,
            "last_login_days": last_login_days,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "retention_rate": f"{retention_prob}%",
            "retention_prob": retention_prob,
            "key_signals": signals
        })

    avg_monthly_spend = round(float(imputed_df["monthly_spend"].mean()), 2)
    overall_retention_rate = round((healthy_count + watchlist_count) / total_records * 100, 1)
    high_risk_pct = round((high_risk_count / total_records) * 100, 1)

    result_data = {
        "total_dataset_records": total_records,
        "retention_rate": overall_retention_rate,
        "avg_monthly_spend": avg_monthly_spend,
        "high_risk_percentage": high_risk_pct,
        "high_risk_count": high_risk_count,
        "healthy_count": healthy_count,
        "watchlist_count": watchlist_count,
        "customers": customers_list
    }

    os.makedirs(output_json_path.parent, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"[+] Successfully wrote {len(customers_list)} records to: {output_json_path}")
    print(f"    - Total: {total_records}")
    print(f"    - Healthy: {healthy_count}")
    print(f"    - Watchlist: {watchlist_count}")
    print(f"    - High Risk: {high_risk_count}")
    print(f"    - Avg Spend: ${avg_monthly_spend}")
    print(f"    - Retention Rate: {overall_retention_rate}%")


if __name__ == "__main__":
    generate_full_customers()
