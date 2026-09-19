"""
Milestone 1: E-Commerce Customer Churn Data Ingestion & Engineering Module
-------------------------------------------------------------------------
Loads and prepares the official Kaggle E-Commerce Customer Churn Dataset (5,630 records, 20 attributes).
"""

from src.data_download import download_and_standardize_kaggle_data


def generate_ecommerce_churn_dataset(n_samples: int = 5630, random_seed: int = 42):
    return download_and_standardize_kaggle_data()


if __name__ == "__main__":
    generate_ecommerce_churn_dataset()
