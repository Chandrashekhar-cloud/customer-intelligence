"""
Milestone 2: Exploratory Data Analysis (EDA) Module (Kaggle E-Commerce Dataset)
-------------------------------------------------------------------------------
Performs comprehensive exploratory data analysis and generates publication-grade
visualizations saved in reports/figures/:
1. 01_churn_distribution.png
2. 02_correlation_heatmap.png
3. 03_numerical_distributions.png
4. 04_churn_by_contract_type.png (Churn by Preferred Order Category)
5. 05_churn_by_subscription_type.png (Churn by Preferred Payment Mode)
6. 06_important_feature_relationships.png
7. 07_outlier_analysis.png
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import clean_anomalies, fit_imputation_parameters, transform_missing_values, fit_outlier_caps, apply_outlier_caps
from src.feature_engineering import add_engineered_features

# Configure Matplotlib styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_churn_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 1: Target class distribution highlighting class imbalance."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    churn_counts = df["churn"].value_counts()
    churn_pcts = df["churn"].value_counts(normalize=True) * 100

    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(["Retained (0)", "Churned (1)"], churn_counts.values, color=colors, width=0.5, edgecolor="#333333", linewidth=1)

    for bar, pct, count in zip(bars, churn_pcts.values, churn_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 60,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(f"Target Class Distribution: Kaggle E-Commerce Churn ({churn_pcts.iloc[0]:.1f}% vs {churn_pcts.iloc[1]:.1f}%)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.set_ylim(0, max(churn_counts.values) * 1.18)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_churn_distribution.png"))
    plt.close(fig)
    print("  [+] Generated: 01_churn_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 2: Correlation heatmap for all numerical and engineered features with churn."""
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    numerical_df = df.select_dtypes(include=[np.number])
    corr = numerical_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.7, vmin=-0.7, center=0,
                square=True, linewidths=0.6, annot=False,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}, ax=ax)

    ax.set_title("Feature Correlation Matrix (Real Kaggle Dataset)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"))
    plt.close(fig)
    print("  [+] Generated: 02_correlation_heatmap.png")


def plot_numerical_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 3: Numerical feature distributions stratified by churn status."""
    features_to_plot = [
        "tenure", "warehouse_to_home", "hour_spend_on_app",
        "satisfaction_score", "day_since_last_order", "cashback_amount"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        if feature in df.columns:
            sns.kdeplot(data=df, x=feature, hue="churn", common_norm=False, fill=True,
                        palette=["#2b5c8f", "#d9534f"], alpha=0.35, linewidth=1.8, ax=ax)
            ax.set_title(f"Distribution: {feature}", fontsize=11, fontweight="bold")
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)

    plt.suptitle("Density Distributions: Retained (0) vs. Churned (1) Customers", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_numerical_distributions.png"))
    plt.close(fig)
    print("  [+] Generated: 03_numerical_distributions.png")


def plot_churn_by_order_category(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 4: Churn breakdown across preferred order categories."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    col = "preferred_order_cat" if "preferred_order_cat" in df.columns else df.columns[1]
    summary = df.groupby(col)["churn"].agg(["count", "mean"]).reset_index()
    summary["churn_pct"] = summary["mean"] * 100

    bars = ax.bar(summary[col], summary["churn_pct"],
                  color=["#d9534f", "#f0ad4e", "#5cb85c", "#3498db", "#9b59b6"], width=0.45, edgecolor="#333", linewidth=1)

    for bar, pct, count in zip(bars, summary["churn_pct"], summary["count"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.8,
                f"{pct:.1f}%\n({count:,})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Customer Churn Rate by Preferred Order Category", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_ylim(0, max(summary["churn_pct"]) * 1.25)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_churn_by_contract_type.png"))
    plt.close(fig)
    print("  [+] Generated: 04_churn_by_contract_type.png")


def plot_churn_by_payment_mode(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 5: Churn breakdown across payment methods."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    col = "preferred_payment_mode" if "preferred_payment_mode" in df.columns else df.columns[2]
    summary = df.groupby(col)["churn"].agg(["count", "mean"]).reset_index()
    summary["churn_pct"] = summary["mean"] * 100

    bars = ax.bar(summary[col], summary["churn_pct"],
                  color=["#e67e22", "#3498db", "#9b59b6", "#2ecc71", "#e74c3c"], width=0.45, edgecolor="#333", linewidth=1)

    for bar, pct, count in zip(bars, summary["churn_pct"], summary["count"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.8,
                f"{pct:.1f}%\n({count:,})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Customer Churn Rate by Preferred Payment Mode", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_ylim(0, max(summary["churn_pct"]) * 1.25)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_churn_by_subscription_type.png"))
    plt.close(fig)
    print("  [+] Generated: 05_churn_by_subscription_type.png")


def plot_important_feature_relationships(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 6: Key relationships (Complain & Satisfaction vs Churn)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Complain vs Churn
    sns.barplot(data=df, x="complain", y="churn", palette=["#2b5c8f", "#d9534f"], ax=axes[0])
    axes[0].set_title("Customer Complaints vs. Churn Rate", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(["No Complaint (0)", "Lodged Complaint (1)"])
    axes[0].set_ylabel("Churn Rate", fontsize=10)

    # 2. Recency Ratio by Churn Class
    if "recency_ratio" in df.columns:
        sns.boxplot(data=df, x="churn", y="recency_ratio", hue="churn", palette=["#2b5c8f", "#d9534f"], legend=False, ax=axes[1], width=0.4)
        axes[1].set_title("Engineered Recency Ratio by Churn Class", fontsize=12, fontweight="bold")
        axes[1].set_xticklabels(["Retained (0)", "Churned (1)"])
        axes[1].set_ylabel("Recency Ratio (days / tenure)", fontsize=10)

    plt.suptitle("Key Behavioral Risk Patterns in Kaggle Dataset", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_important_feature_relationships.png"))
    plt.close(fig)
    print("  [+] Generated: 06_important_feature_relationships.png")


def plot_outlier_analysis(raw_df: pd.DataFrame, capped_df: pd.DataFrame, output_dir: str) -> None:
    """Plot 7: Outlier inspection before and after IQR capping."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    # Raw warehouse distance vs Capped
    sns.boxplot(x=raw_df["warehouse_to_home"].dropna(), ax=axes[0, 0], color="#f39c12")
    axes[0, 0].set_title("Raw Warehouse Distance (km) Outliers", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Warehouse To Home (km)")

    sns.boxplot(x=capped_df["warehouse_to_home"], ax=axes[0, 1], color="#27ae60")
    axes[0, 1].set_title("Warehouse Distance after IQR Capping", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Warehouse To Home (km)")

    # Raw cashback vs Capped
    sns.boxplot(x=raw_df["cashback_amount"].dropna(), ax=axes[1, 0], color="#e74c3c")
    axes[1, 0].set_title("Raw Cashback Amount ($) Outliers", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Cashback ($)")

    sns.boxplot(x=capped_df["cashback_amount"], ax=axes[1, 1], color="#2980b9")
    axes[1, 1].set_title("Cashback ($) after IQR Capping", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Cashback ($)")

    plt.suptitle("Outlier Analysis: Pre vs. Post IQR Boundary Winsorization", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_outlier_analysis.png"))
    plt.close(fig)
    print("  [+] Generated: 07_outlier_analysis.png")


def run_eda(raw_data_path: str = "data/raw/ecommerce_churn_raw.csv", output_dir: str = "reports/figures") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print(" MILESTONE 2: EXPLORATORY DATA ANALYSIS (EDA) - KAGGLE DATASET")
    print("=================================================================")
    print(f"[*] Reading dataset from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)

    cleaned_df = clean_anomalies(raw_df)
    imputer_dict = fit_imputation_parameters(cleaned_df)
    imputed_df = transform_missing_values(cleaned_df, imputer_dict)

    cap_bounds = fit_outlier_caps(imputed_df)
    capped_df = apply_outlier_caps(imputed_df, cap_bounds)

    engineered_df = add_engineered_features(capped_df)

    print(f"[*] Generating EDA visual artifacts to: {output_dir}")
    plot_churn_distribution(raw_df, output_dir)
    plot_correlation_heatmap(engineered_df, output_dir)
    plot_numerical_distributions(engineered_df, output_dir)
    plot_churn_by_order_category(raw_df, output_dir)
    plot_churn_by_payment_mode(raw_df, output_dir)
    plot_important_feature_relationships(engineered_df, output_dir)
    plot_outlier_analysis(raw_df, capped_df, output_dir)
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run Milestone 2 EDA and generate figures.")
    parser.add_argument("--data", type=str, default="data/raw/ecommerce_churn_raw.csv", help="Input CSV path")
    parser.add_argument("--outdir", type=str, default="reports/figures", help="Output directory for plots")
    args = parser.parse_args()

    run_eda(raw_data_path=args.data, output_dir=args.outdir)


if __name__ == "__main__":
    main()
