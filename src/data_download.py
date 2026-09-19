"""
Milestone 1: Real Kaggle E-Commerce Customer Churn Data Ingestion Module
-----------------------------------------------------------------------
Downloads and ingests the authentic Kaggle 'Ecommerce Customer Churn Analysis
and Prediction' dataset (5,630 records, 20 attributes) with authentic real-world
missing values and true ~83/17 retention distribution.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/Leangonplu/Ecommerce_Customer_Churn_Analysis_and_Prediction/main/E%20Commerce%20Dataset.xlsx"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUTPUT_CSV = os.path.join(RAW_DIR, "ecommerce_churn_raw.csv")
METADATA_JSON = os.path.join(RAW_DIR, "dataset_metadata.json")


def download_and_standardize_kaggle_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[*] Downloading official Kaggle E-Commerce Churn dataset from:\n    {DATA_URL}...")
    
    df = pd.read_excel(DATA_URL, sheet_name="E Comm")
    print(f"[+] Downloaded successfully. Raw shape: {df.shape}")

    # Standardize column naming to clean snake_case
    rename_dict = {
        "CustomerID": "customer_id",
        "Churn": "churn",
        "Tenure": "tenure",
        "PreferredLoginDevice": "preferred_login_device",
        "CityTier": "city_tier",
        "WarehouseToHome": "warehouse_to_home",
        "PreferredPaymentMode": "preferred_payment_mode",
        "Gender": "gender",
        "HourSpendOnApp": "hour_spend_on_app",
        "NumberOfDeviceRegistered": "number_of_device_registered",
        "PreferedOrderCat": "preferred_order_cat",
        "SatisfactionScore": "satisfaction_score",
        "MaritalStatus": "marital_status",
        "NumberOfAddress": "number_of_address",
        "Complain": "complain",
        "OrderAmountHikeFromlastYear": "order_amount_hike_from_last_year",
        "CouponUsed": "coupon_used",
        "OrderCount": "order_count",
        "DaySinceLastOrder": "day_since_last_order",
        "CashbackAmount": "cashback_amount"
    }
    df = df.rename(columns=rename_dict)

    # Standardize category labels and aliases
    df["preferred_login_device"] = df["preferred_login_device"].replace({"Phone": "Mobile Phone"})
    df["preferred_payment_mode"] = df["preferred_payment_mode"].replace({
        "CC": "Credit Card",
        "COD": "Cash on Delivery"
    })
    df["preferred_order_cat"] = df["preferred_order_cat"].replace({"Mobile": "Mobile Phone"})

    # Ensure clean Customer ID format (e.g. CUST-50001)
    df["customer_id"] = "CUST-" + df["customer_id"].astype(str)

    # Reorder columns with customer_id first and churn last
    feature_cols = [c for c in df.columns if c not in ["customer_id", "churn"]]
    df = df[["customer_id"] + feature_cols + ["churn"]]

    # Export raw CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Exported cleaned Kaggle dataset to: {OUTPUT_CSV} ({len(df)} records)")

    # Compute metadata
    missing_summary = df.isnull().sum().to_dict()
    churn_counts = df["churn"].value_counts().to_dict()
    metadata = {
        "dataset_name": "Kaggle E-Commerce Customer Churn Analysis & Prediction",
        "source": "Kaggle (ankitverma2010)",
        "total_records": len(df),
        "total_features": len(feature_cols),
        "retained_count": int(churn_counts.get(0, 0)),
        "churned_count": int(churn_counts.get(1, 0)),
        "retention_rate": round(churn_counts.get(0, 0) / len(df) * 100, 2),
        "churn_rate": round(churn_counts.get(1, 0) / len(df) * 100, 2),
        "missing_values": {k: int(v) for k, v in missing_summary.items() if v > 0},
        "feature_list": feature_cols
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved dataset metadata to: {METADATA_JSON}")

    return df


if __name__ == "__main__":
    download_and_standardize_kaggle_data()
