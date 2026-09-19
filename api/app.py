"""
Milestone 4: Flask REST API Module (Kaggle E-Commerce Churn Intelligence)
--------------------------------------------------------------------------
Provides a production-ready, lightweight REST API for real-time customer churn prediction,
explainable risk attribution, batch CSV scoring, customer directory queries, and portfolio insights.

Endpoints:
- GET  /health          : Verifies service and ML model operational status
- POST /predict         : Single customer churn risk prediction + tree-based explainability
- POST /predict/batch   : Batch CSV/JSON evaluation with schema validation & risk attribution
- GET  /customers       : Filterable & searchable directory of all 5,630 Kaggle customer accounts
- GET  /customers/<id>  : Individual customer profile & telemetry lookup
- GET  /insights        : Dynamic portfolio aggregations computed across all 5,630 authentic records
- GET  /model/info      : Transparent model architecture, hyperparameters, and holdout evaluation metrics
- GET  /analytics       : Aggregated portfolio retention metrics (legacy compatible)
"""

import os
import sys
import io
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.model_utils import load_pipeline
from src.explainability import compute_tree_contributions, determine_risk_tier
from api.schemas import (
    validate_prediction_payload,
    validate_batch_payload,
    parse_and_validate_csv_data,
    REQUIRED_FIELDS
)


def create_app(model_path: str = "models/best_churn_model.joblib") -> Flask:
    """
    Application factory for the Churn Prediction Flask API.
    """
    frontend_dir = os.path.join(ROOT_DIR, "frontend")
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    CORS(app, resources={r"/*": {"origins": "*"}})

    resolved_path = os.path.abspath(model_path)
    model_pipeline = None

    if os.path.exists(resolved_path):
        try:
            model_pipeline = load_pipeline(resolved_path)
            print(f"[*] ML Inference Pipeline successfully loaded from: {resolved_path}")
        except Exception as e:
            print(f"[!] Warning: Failed to load model artifact: {e}")
    else:
        print(f"[!] Warning: Model file not found at: {resolved_path}. It will be loaded on demand.")

    def _get_model():
        nonlocal model_pipeline
        if model_pipeline is None and os.path.exists(resolved_path):
            model_pipeline = load_pipeline(resolved_path)
        return model_pipeline

    # Load customers cache
    customers_data_cache = None
    customers_json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "assets", "customers.json"
    )

    def _ensure_customers_cache():
        nonlocal customers_data_cache
        if customers_data_cache is None:
            if os.path.exists(customers_json_path):
                try:
                    with open(customers_json_path, "r", encoding="utf-8") as f:
                        customers_data_cache = json.load(f)
                    print(f"[*] Preloaded {len(customers_data_cache.get('customers', []))} Kaggle customers into memory.")
                except Exception as ex:
                    print(f"[!] Error loading customers.json: {ex}")
        return customers_data_cache

    _ensure_customers_cache()

    # ============================================================
    # 1. HEALTH & TRANSPARENCY ENDPOINTS
    # ============================================================

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint returning system status and model readiness."""
        is_ready = _get_model() is not None
        return jsonify({
            "status": "healthy",
            "service": "E-Commerce Customer Churn Prediction API",
            "version": "2.0.0",
            "dataset": "Kaggle E-Commerce Customer Churn (5,630 records)",
            "model_loaded": is_ready,
            "explainability_engine": "Tree-Path Feature Contribution Decomposition",
            "active_endpoints": [
                "GET /health",
                "POST /predict",
                "POST /predict/batch",
                "GET /customers",
                "GET /customers/<id>",
                "GET /insights",
                "GET /model/info"
            ]
        }), 200

    @app.route("/model/info", methods=["GET"])
    def model_info():
        """Retrieve model metadata, architecture details, and holdout evaluation metrics."""
        metadata_path = ROOT_DIR / "models" / "model_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return jsonify({
                    "status": "success",
                    "metadata": meta,
                    "pipeline_stages": [
                        "1. AnomalyCleaner (removes negative anomalies)",
                        "2. DataFrameImputer (median for numeric, mode for categorical)",
                        "3. IQROutlierCapper (robust IQR 1.5x thresholding)",
                        "4. DomainFeatureEngineer (cashback_per_order, friction, recency_ratio)",
                        "5. ColumnTransformer (OneHotEncoder + Passthrough)",
                        "6. Zero-Leakage SMOTE (applied strictly inside CV folds)",
                        "7. RandomForestClassifier (150 estimators, sqrt max_features)"
                    ]
                }), 200
            except Exception as ex:
                return jsonify({"error": "ReadError", "message": str(ex)}), 500

        return jsonify({
            "model_name": "Tuned Random Forest Classifier",
            "version": "2.0.0",
            "dataset_size": 5630,
            "feature_count": 35
        }), 200

    # ============================================================
    # 2. PREDICTION & EXPLAINABILITY ENDPOINTS
    # ============================================================

    @app.route("/predict", methods=["POST"])
    def predict_churn():
        """
        Predict customer churn and compute exact tree-based feature contributions.
        """
        if not request.is_json:
            return jsonify({
                "error": "InvalidContentType",
                "message": "Content-Type must be 'application/json'."
            }), 400

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({
                "error": "MalformedJSON",
                "message": "Failed to parse JSON body."
            }), 400

        is_valid, error_message, cleaned_data = validate_prediction_payload(payload)
        if not is_valid:
            return jsonify({
                "error": "ValidationError",
                "message": error_message
            }), 400

        pipeline = _get_model()
        if pipeline is None:
            return jsonify({
                "error": "ModelUnavailable",
                "message": "Trained model artifact is currently unavailable."
            }), 503

        try:
            input_df = pd.DataFrame([cleaned_data])
            prediction = int(pipeline.predict(input_df)[0])
            probabilities = pipeline.predict_proba(input_df)[0]
            churn_probability = float(round(probabilities[1], 4))

            # Compute exact tree path feature contributions
            prob, risk_drivers, protective_signals = compute_tree_contributions(pipeline, input_df)
            tier, label, recommendation = determine_risk_tier(churn_probability)

            response = {
                "prediction": prediction,
                "prediction_label": label,
                "probability": churn_probability,
                "risk_tier": tier,
                "recommendation": recommendation,
                "risk_drivers": risk_drivers,
                "protective_signals": protective_signals
            }

            return jsonify(response), 200

        except Exception as ex:
            print(f"[!] Internal prediction error: {ex}")
            return jsonify({
                "error": "InternalInferenceError",
                "message": f"An error occurred while evaluating the customer prediction: {str(ex)}"
            }), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """
        Batch prediction endpoint supporting CSV uploads and JSON arrays.
        Validates all records, executes the ML pipeline, and generates explainability drivers.
        """
        pipeline = _get_model()
        if pipeline is None:
            return jsonify({
                "error": "ModelUnavailable",
                "message": "Trained model artifact is currently unavailable."
            }), 503

        cleaned_records = None
        row_errors = []

        # Handle Multipart Form Upload (CSV File)
        if "file" in request.files:
            uploaded_file = request.files["file"]
            filename = uploaded_file.filename or ""
            if not filename.lower().endswith(".csv"):
                return jsonify({
                    "error": "ValidationError",
                    "message": "Unsupported file format. Please upload a valid CSV file (.csv)."
                }), 400

            try:
                csv_bytes = uploaded_file.read()
                csv_text = csv_bytes.decode("utf-8", errors="replace")
                is_valid, err_msg, cleaned_records, row_errors = parse_and_validate_csv_data(csv_text)
                if not is_valid:
                    return jsonify({
                        "error": "ValidationError",
                        "message": err_msg,
                        "row_errors": row_errors
                    }), 400
            except Exception as ex:
                return jsonify({
                    "error": "ValidationError",
                    "message": f"Failed to read CSV upload: {str(ex)}"
                }), 400

        # Handle JSON Payload
        elif request.is_json:
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify({"error": "MalformedJSON", "message": "Failed to parse JSON body."}), 400

            records = payload.get("records") if isinstance(payload, dict) else payload
            is_valid, err_msg, cleaned_records, row_errors = validate_batch_payload(records)
            if not is_valid:
                return jsonify({
                    "error": "ValidationError",
                    "message": err_msg,
                    "row_errors": row_errors
                }), 400

        else:
            return jsonify({
                "error": "InvalidRequest",
                "message": "Expected multipart/form-data with 'file' or application/json with records array."
            }), 400

        if not cleaned_records:
            return jsonify({
                "error": "ValidationError",
                "message": "No valid customer records found for evaluation."
            }), 400

        try:
            batch_df = pd.DataFrame(cleaned_records)
            eval_df = batch_df.drop(columns=["customer_id"], errors="ignore")

            predictions = pipeline.predict(eval_df)
            probabilities = pipeline.predict_proba(eval_df)[:, 1]

            results = []
            critical_count = 0
            high_count = 0
            watch_count = 0
            healthy_count = 0

            for i in range(len(cleaned_records)):
                rec = cleaned_records[i]
                cust_id = rec.get("customer_id", f"REC-{i + 1}")
                prob = float(round(probabilities[i], 4))
                pred = int(predictions[i])
                tier, label, recommendation = determine_risk_tier(prob)

                if tier == "Critical":
                    critical_count += 1
                elif tier == "High Risk":
                    high_count += 1
                elif tier == "Watch":
                    watch_count += 1
                else:
                    healthy_count += 1

                # Fast heuristic or sample explainability for batch display
                single_row = eval_df.iloc[[i]]
                _, drivers, protective = compute_tree_contributions(pipeline, single_row)

                results.append({
                    "customer_id": cust_id,
                    "prediction": pred,
                    "prediction_label": label,
                    "probability": prob,
                    "risk_score": float(round(prob * 100, 1)),
                    "risk_tier": tier,
                    "recommendation": recommendation,
                    "risk_drivers": drivers,
                    "protective_signals": protective,
                    "tenure": rec.get("tenure", 0),
                    "preferred_order_cat": rec.get("preferred_order_cat", ""),
                    "preferred_payment_mode": rec.get("preferred_payment_mode", ""),
                    "cashback_amount": rec.get("cashback_amount", 0.0),
                    "complain": int(rec.get("complain", 0)),
                    "satisfaction_score": int(rec.get("satisfaction_score", 3)),
                    "day_since_last_order": rec.get("day_since_last_order", 0)
                })

            summary = {
                "total_records": len(results),
                "critical_count": critical_count,
                "high_count": high_count,
                "watch_count": watch_count,
                "healthy_count": healthy_count,
                "average_probability": float(round(float(np.mean(probabilities)), 4))
            }

            return jsonify({
                "total_evaluated": len(results),
                "summary": summary,
                "results": results
            }), 200

        except Exception as ex:
            print(f"[!] Batch prediction error: {ex}")
            return jsonify({
                "error": "BatchInferenceError",
                "message": f"Failed during batch ML evaluation: {str(ex)}"
            }), 500

    # ============================================================
    # 3. CUSTOMER DIRECTORY & FILTERING (5,630 RECORDS)
    # ============================================================

    @app.route("/customers", methods=["GET"])
    def get_customers():
        """
        Paginated customer directory endpoint serving the 5,630 Kaggle records.
        Supports advanced filtering by risk priority, category, payment mode, tenure,
        spend range, complaints, satisfaction, and sorting.
        """
        cache = _ensure_customers_cache()
        if not cache:
            return jsonify({"error": "DataUnavailable", "message": "Customer dataset unavailable."}), 500

        all_customers = cache.get("customers", [])

        page = max(1, request.args.get("page", default=1, type=int))
        limit = max(1, min(10000, request.args.get("limit", default=25, type=int)))
        search_query = request.args.get("search", default="", type=str).strip().lower()

        risk_filter = request.args.get("risk", default="all", type=str).strip().lower()
        order_cat_filter = request.args.get("order_cat", default="all", type=str).strip().lower()
        payment_filter = request.args.get("payment", default="all", type=str).strip().lower()
        tenure_range = request.args.get("tenure_range", default="all", type=str).strip().lower()
        spend_range = request.args.get("spend_range", default="all", type=str).strip().lower()
        complain_filter = request.args.get("complain", default="all", type=str).strip().lower()
        sat_filter = request.args.get("sat", default="all", type=str).strip().lower()

        sort_by = request.args.get("sort_by", default="id", type=str).strip()
        sort_order = request.args.get("sort_order", default="asc", type=str).strip().lower()

        filtered = all_customers

        # Filter by search term
        if search_query:
            filtered = [
                c for c in filtered
                if search_query in c["id"].lower()
                or search_query in c.get("preferred_order_cat", "").lower()
                or search_query in c.get("preferred_payment_mode", "").lower()
            ]

        # Filter by risk tier / level (handling both Critical/High/Watch/Healthy and legacy High Risk/Watchlist)
        if risk_filter != "all":
            def matches_risk(c, r_filter):
                lvl = c.get("risk_level", "").lower()
                if r_filter == "critical":
                    return lvl == "critical"
                if r_filter == "high risk":
                    return lvl == "high risk"
                if r_filter == "high":
                    return lvl in ["high", "high risk"]
                if r_filter in ["watch", "watchlist"]:
                    return lvl in ["watch", "watchlist"]
                if r_filter == "healthy":
                    return lvl == "healthy"
                return lvl == r_filter

            filtered = [c for c in filtered if matches_risk(c, risk_filter)]

        # Filter by order category
        if order_cat_filter != "all":
            filtered = [c for c in filtered if c.get("preferred_order_cat", "").lower() == order_cat_filter]

        # Filter by payment method
        if payment_filter != "all":
            filtered = [c for c in filtered if c.get("preferred_payment_mode", "").lower() == payment_filter]

        # Filter by tenure range
        if tenure_range != "all":
            if tenure_range == "0-6":
                filtered = [c for c in filtered if c.get("tenure", 0) <= 6]
            elif tenure_range == "6-12":
                filtered = [c for c in filtered if 6 < c.get("tenure", 0) <= 12]
            elif tenure_range == "12-24":
                filtered = [c for c in filtered if 12 < c.get("tenure", 0) <= 24]
            elif tenure_range == "24+":
                filtered = [c for c in filtered if c.get("tenure", 0) > 24]

        # Filter by spend/cashback range
        if spend_range != "all":
            if spend_range == "<120":
                filtered = [c for c in filtered if c.get("cashback_amount", 0) < 120]
            elif spend_range == "120-160":
                filtered = [c for c in filtered if 120 <= c.get("cashback_amount", 0) < 160]
            elif spend_range == "160-200":
                filtered = [c for c in filtered if 160 <= c.get("cashback_amount", 0) < 200]
            elif spend_range == "200+":
                filtered = [c for c in filtered if c.get("cashback_amount", 0) >= 200]

        # Filter by complaint
        if complain_filter in ["1", "true", "yes"]:
            filtered = [c for c in filtered if c.get("complain", 0) == 1]
        elif complain_filter in ["0", "false", "no"]:
            filtered = [c for c in filtered if c.get("complain", 0) == 0]

        # Filter by satisfaction
        if sat_filter == "low":
            filtered = [c for c in filtered if c.get("satisfaction_score", 3) <= 2]
        elif sat_filter == "high":
            filtered = [c for c in filtered if c.get("satisfaction_score", 3) >= 4]

        # Sorting
        reverse = (sort_order == "desc")
        numeric_sort_keys = {
            "risk_score", "probability", "tenure", "cashback_amount",
            "order_count", "satisfaction_score", "day_since_last_order",
            "hour_spend_on_app", "complain"
        }
        if sort_by in numeric_sort_keys:
            filtered = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=reverse)
        else:
            filtered = sorted(filtered, key=lambda x: str(x.get("id", "")), reverse=reverse)

        total_filtered = len(filtered)
        total_pages = max(1, (total_filtered + limit - 1) // limit)
        page = min(page, total_pages)

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paged_records = filtered[start_idx:end_idx]

        return jsonify({
            "total": len(all_customers),
            "filtered_total": total_filtered,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "customers": paged_records,
            "stats": {
                "total_customers": len(all_customers),
                "critical_count": cache.get("critical_count", 0),
                "high_risk_count": cache.get("high_risk_count", 0),
                "watchlist_count": cache.get("watchlist_count", 0),
                "healthy_count": cache.get("healthy_count", 0),
                "retention_rate": cache.get("retention_rate", 83.2)
            }
        }), 200

    @app.route("/customers/<cust_id>", methods=["GET"])
    def get_customer(cust_id):
        """Retrieve full Customer 360 profile for an individual customer."""
        cache = _ensure_customers_cache()
        if not cache:
            return jsonify({"error": "DataUnavailable"}), 500

        cust_target = cust_id.strip().upper()
        for c in cache.get("customers", []):
            if c["id"].upper() == cust_target:
                return jsonify(c), 200

        return jsonify({"error": "NotFound", "message": f"Customer '{cust_id}' not found."}), 404

    # ============================================================
    # 4. INSIGHTS & ANALYTICS ENDPOINTS (REAL DATASET DERIVED)
    # ============================================================

    @app.route("/insights", methods=["GET"])
    def get_insights():
        """
        Dynamically calculate actionable retention patterns, category churn rates,
        tenure risk curves, and complaint impacts across all 5,630 authentic records.
        """
        cache = _ensure_customers_cache()
        if not cache:
            return jsonify({"error": "DataUnavailable"}), 500

        all_c = cache.get("customers", [])
        total = len(all_c) or 1

        # 1. Health Distribution
        critical = [c for c in all_c if c.get("risk_level") == "Critical" or c.get("risk_score", 0) >= 65.0]
        high = [c for c in all_c if (c.get("risk_level") in ["High", "High Risk"]) or (40.0 <= c.get("risk_score", 0) < 65.0)]
        watch = [c for c in all_c if (c.get("risk_level") in ["Watch", "Watchlist"]) or (18.0 <= c.get("risk_score", 0) < 40.0)]
        healthy = [c for c in all_c if c.get("risk_level") == "Healthy" or c.get("risk_score", 0) < 18.0]

        # 2. Category Churn Distribution
        categories = ["Mobile Phone", "Fashion", "Laptop & Accessory", "Grocery", "Others"]
        cat_stats = []
        for cat in categories:
            cat_customers = [c for c in all_c if c.get("preferred_order_cat") == cat]
            if cat_customers:
                at_risk = len([c for c in cat_customers if c.get("risk_score", 0) >= 40.0])
                churn_pct = round(at_risk / len(cat_customers) * 100, 1)
                cat_stats.append({
                    "category": cat,
                    "total": len(cat_customers),
                    "at_risk": at_risk,
                    "churn_rate": churn_pct
                })

        # 3. Complaints vs Churn Impact
        with_comp = [c for c in all_c if c.get("complain") == 1]
        no_comp = [c for c in all_c if c.get("complain") == 0]

        comp_churn = round(len([c for c in with_comp if c.get("risk_score", 0) >= 40.0]) / (len(with_comp) or 1) * 100, 1)
        no_comp_churn = round(len([c for c in no_comp if c.get("risk_score", 0) >= 40.0]) / (len(no_comp) or 1) * 100, 1)

        # 4. Tenure Lifecycle Risk Curve
        tenure_buckets = [
            ("0–3 mo", lambda t: t <= 3),
            ("4–6 mo", lambda t: 3 < t <= 6),
            ("7–12 mo", lambda t: 6 < t <= 12),
            ("13–24 mo", lambda t: 12 < t <= 24),
            ("> 24 mo", lambda t: t > 24)
        ]
        tenure_stats = []
        for label, fn in tenure_buckets:
            bucket_cust = [c for c in all_c if fn(c.get("tenure", 0))]
            if bucket_cust:
                bucket_risk = len([c for c in bucket_cust if c.get("risk_score", 0) >= 40.0])
                rate = round(bucket_risk / len(bucket_cust) * 100, 1)
                tenure_stats.append({
                    "range": label,
                    "total": len(bucket_cust),
                    "churn_rate": rate
                })

        # 5. Order Inactivity vs Churn
        inactivity_buckets = [
            ("0–3 Days", lambda d: d <= 3),
            ("4–7 Days", lambda d: 3 < d <= 7),
            ("8–14 Days", lambda d: 7 < d <= 14),
            ("> 14 Days", lambda d: d > 14)
        ]
        inactivity_stats = []
        for label, fn in inactivity_buckets:
            bucket = [c for c in all_c if fn(c.get("day_since_last_order", 0))]
            if bucket:
                at_risk = len([c for c in bucket if c.get("risk_score", 0) >= 40.0])
                rate = round(at_risk / len(bucket) * 100, 1)
                inactivity_stats.append({
                    "bucket": label,
                    "total": len(bucket),
                    "churn_rate": rate
                })

        # 6. High-Value Accounts at Risk (Top 10 Spenders at Critical/High Risk)
        urgent_accounts = sorted(
            [c for c in all_c if c.get("risk_score", 0) >= 40.0],
            key=lambda x: x.get("cashback_amount", 0),
            reverse=True
        )[:10]

        return jsonify({
            "total_records": total,
            "retention_baseline": round((len(healthy) + len(watch)) / total * 100, 1),
            "health_distribution": {
                "critical": len(critical),
                "critical_pct": round(len(critical) / total * 100, 1),
                "high": len(high),
                "high_pct": round(len(high) / total * 100, 1),
                "watch": len(watch),
                "watch_pct": round(len(watch) / total * 100, 1),
                "healthy": len(healthy),
                "healthy_pct": round(len(healthy) / total * 100, 1)
            },
            "category_churn_rates": cat_stats,
            "complaint_impact": {
                "with_complaint_churn_rate": comp_churn,
                "without_complaint_churn_rate": no_comp_churn,
                "risk_multiplier": round(comp_churn / (no_comp_churn or 1), 1)
            },
            "tenure_risk_curve": tenure_stats,
            "inactivity_stats": inactivity_stats,
            "high_value_at_risk": urgent_accounts
        }), 200

    @app.route("/analytics", methods=["GET"])
    def get_analytics():
        """Retrieve aggregated portfolio metrics (legacy endpoint)."""
        cache = _ensure_customers_cache()
        if not cache:
            return jsonify({"error": "DataUnavailable"}), 500

        all_c = cache.get("customers", [])
        total = len(all_c) or 1

        healthy = [c for c in all_c if c.get("risk_level") == "Healthy" or c.get("risk_score", 0) < 18.0]
        watchlist = [c for c in all_c if c.get("risk_level") in ["Watch", "Watchlist"] or (18.0 <= c.get("risk_score", 0) < 40.0)]
        high_risk = [c for c in all_c if c.get("risk_level") in ["Critical", "High", "High Risk"] or c.get("risk_score", 0) >= 40.0]

        return jsonify({
            "total_records": total,
            "portfolio": {
                "healthy_count": len(healthy),
                "healthy_rate": round(len(healthy) / total * 100, 1),
                "watchlist_count": len(watchlist),
                "watchlist_rate": round(len(watchlist) / total * 100, 1),
                "high_risk_count": len(high_risk),
                "high_risk_rate": round(len(high_risk) / total * 100, 1),
                "retention_rate": cache.get("retention_rate", 83.2)
            }
        }), 200

    @app.route("/", methods=["GET"])
    def serve_frontend_index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:path>", methods=["GET"])
    def serve_frontend_static(path):
        target = os.path.join(frontend_dir, path)
        if os.path.isfile(target):
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, "index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    print("[*] Starting Flask REST API on http://127.0.0.1:5000...")
    app.run(host="127.0.0.1", port=5000, debug=False)
