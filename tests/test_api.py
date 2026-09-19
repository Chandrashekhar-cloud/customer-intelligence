"""
Milestone 4: Automated Testing Suite (Kaggle E-Commerce Dataset)
---------------------------------------------------------------
Pytest unit test suite verifying the Flask REST API:
1. Health check endpoint functionality
2. Valid customer prediction payload
3. High-risk churn prediction sensitivity
4. Missing required fields rejection (HTTP 400)
5. Invalid data types rejection (HTTP 400)
6. Invalid categorical values rejection (HTTP 400)
7. Domain boundary violations rejection (e.g. negative tenure, invalid satisfaction)
8. Prediction response schema completeness
9. Customers pagination endpoint (5,630 records)
10. Customer search and risk filter
11. Individual customer profile by ID
12. Analytics portfolio aggregation endpoint
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from api.app import create_app


@pytest.fixture
def client():
    """Create test client fixture for the Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_customer_payload():
    """Standard valid customer payload (loyal profile)."""
    return {
        "tenure": 24.0,
        "preferred_login_device": "Computer",
        "city_tier": 1,
        "warehouse_to_home": 10.0,
        "preferred_payment_mode": "Credit Card",
        "gender": "Female",
        "hour_spend_on_app": 3.0,
        "number_of_device_registered": 3,
        "preferred_order_cat": "Laptop & Accessory",
        "satisfaction_score": 4,
        "marital_status": "Married",
        "number_of_address": 3,
        "complain": 0,
        "order_amount_hike_from_last_year": 14.0,
        "coupon_used": 2.0,
        "order_count": 4.0,
        "day_since_last_order": 2.0,
        "cashback_amount": 220.0
    }


@pytest.fixture
def high_risk_customer_payload():
    """High-risk churn customer profile."""
    return {
        "tenure": 1.0,
        "preferred_login_device": "Mobile Phone",
        "city_tier": 3,
        "warehouse_to_home": 35.0,
        "preferred_payment_mode": "Cash on Delivery",
        "gender": "Male",
        "hour_spend_on_app": 1.0,
        "number_of_device_registered": 4,
        "preferred_order_cat": "Mobile Phone",
        "satisfaction_score": 1,
        "marital_status": "Single",
        "number_of_address": 8,
        "complain": 1,
        "order_amount_hike_from_last_year": 11.0,
        "coupon_used": 0.0,
        "order_count": 1.0,
        "day_since_last_order": 14.0,
        "cashback_amount": 110.0
    }


def test_health_check_endpoint(client):
    """Test 1: Verify GET /health returns HTTP 200 and operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "service" in data


def test_valid_prediction_request(client, valid_customer_payload):
    """Test 2: Verify POST /predict accepts valid input and returns predictions."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "prediction_label" in data
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_high_risk_customer_response(client, high_risk_customer_payload):
    """Test 3: Verify model flags high-risk behavioral profile."""
    response = client.post("/predict", json=high_risk_customer_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] == 1
    assert data["probability"] > 0.50
    assert data["prediction_label"] in ["Likely to Churn", "Moderate Risk"]


def test_missing_required_field(client, valid_customer_payload):
    """Test 4: Verify rejection when a required field is omitted."""
    payload = valid_customer_payload.copy()
    del payload["preferred_payment_mode"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "preferred_payment_mode" in data["message"]


def test_invalid_numeric_data_type(client, valid_customer_payload):
    """Test 5: Verify rejection when a numeric field receives a string value."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = "twenty-four"

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "tenure" in data["message"]


def test_invalid_categorical_value(client, valid_customer_payload):
    """Test 6: Verify rejection when an unauthorized categorical value is passed."""
    payload = valid_customer_payload.copy()
    payload["preferred_order_cat"] = "SpacecraftParts"

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "preferred_order_cat" in data["message"]


def test_negative_boundary_constraint(client, valid_customer_payload):
    """Test 7: Verify rejection when numerical domain constraints are violated."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = -10.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "tenure" in data["message"]


def test_prediction_response_schema_completeness(client, valid_customer_payload):
    """Test 8: Verify response contains all required fields for frontend integration."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.get_json()
    expected_fields = ["prediction", "prediction_label", "probability", "risk_tier", "recommendation"]
    for field in expected_fields:
        assert field in data, f"Missing expected field '{field}' in prediction response."


def test_customers_pagination(client):
    """Test 9: Verify GET /customers returns 5,630 records with pagination."""
    response = client.get("/customers?page=1&limit=25")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 5630
    assert data["filtered_total"] == 5630
    assert data["page"] == 1
    assert data["limit"] == 25
    assert len(data["customers"]) == 25
    assert data["stats"]["total_customers"] == 5630
    assert "CUST-" in data["customers"][0]["id"]


def test_customers_search_and_risk_filter(client):
    """Test 10: Verify search by ID and filter by risk level on /customers."""
    res_search = client.get("/customers?search=CUST-50001")
    assert res_search.status_code == 200
    data_search = res_search.get_json()
    assert data_search["filtered_total"] == 1
    assert data_search["customers"][0]["id"] == "CUST-50001"

    res_risk = client.get("/customers?risk=high%20risk&limit=50")
    assert res_risk.status_code == 200
    data_risk = res_risk.get_json()
    assert data_risk["filtered_total"] > 0
    for c in data_risk["customers"]:
        assert c["risk_level"] == "High Risk"


def test_get_customer_by_id(client):
    """Test 11: Verify GET /customers/<id> returns individual customer profile."""
    response = client.get("/customers/CUST-50001")
    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == "CUST-50001"
    assert "risk_score" in data
    assert "retention_prob" in data
    assert "cashback_amount" in data


def test_analytics_endpoint(client):
    """Test 12: Verify GET /analytics returns 5,630 record aggregations."""
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_records"] == 5630
    assert "portfolio" in data


def test_prediction_explainability(client, high_risk_customer_payload):
    """Test 13: Verify tree-based explainability (risk drivers & protective signals) in /predict."""
    response = client.post("/predict", json=high_risk_customer_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "risk_drivers" in data
    assert "protective_signals" in data
    assert isinstance(data["risk_drivers"], list)
    assert len(data["risk_drivers"]) > 0
    first_driver = data["risk_drivers"][0]
    assert "feature" in first_driver
    assert "label" in first_driver
    assert "contribution" in first_driver
    assert first_driver["contribution"] > 0


def test_batch_prediction_json_success(client, valid_customer_payload, high_risk_customer_payload):
    """Test 14: Verify POST /predict/batch with JSON records evaluates successfully."""
    records = [
        dict(valid_customer_payload, customer_id="TEST-001"),
        dict(high_risk_customer_payload, customer_id="TEST-002")
    ]
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_evaluated"] == 2
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["customer_id"] == "TEST-001"
    assert data["results"][1]["customer_id"] == "TEST-002"
    assert "risk_drivers" in data["results"][1]


def test_batch_prediction_validation_failure(client, valid_customer_payload):
    """Test 15: Verify POST /predict/batch returns 400 when records violate schema."""
    invalid_row = valid_customer_payload.copy()
    del invalid_row["satisfaction_score"]

    response = client.post("/predict/batch", json={"records": [invalid_row]})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "row_errors" in data
    assert len(data["row_errors"]) == 1
    assert "satisfaction_score" in data["row_errors"][0]["error"]


def test_batch_prediction_csv_success(client, valid_customer_payload):
    """Test 16: Verify POST /predict/batch with uploaded multipart CSV."""
    import io
    import pandas as pd

    df = pd.DataFrame([valid_customer_payload, valid_customer_payload])
    df["customer_id"] = ["CSV-001", "CSV-002"]
    csv_bytes = io.BytesIO(df.to_csv(index=False).encode("utf-8"))

    response = client.post(
        "/predict/batch",
        data={"file": (csv_bytes, "batch_test.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_evaluated"] == 2
    assert data["results"][0]["customer_id"] == "CSV-001"


def test_insights_endpoint(client):
    """Test 17: Verify GET /insights computes real portfolio aggregations."""
    response = client.get("/insights")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_records"] == 5630
    assert "health_distribution" in data
    assert "category_churn_rates" in data
    assert "complaint_impact" in data
    assert "tenure_risk_curve" in data
    assert "high_value_at_risk" in data


def test_model_info_endpoint(client):
    """Test 18: Verify GET /model/info returns transparency metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "metadata" in data
    assert "pipeline_stages" in data
    assert "RandomForestClassifier" in str(data["pipeline_stages"])

