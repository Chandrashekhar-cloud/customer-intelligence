"""
Milestone 4: API Request Schemas & Validation Module (Kaggle E-Commerce Dataset)
--------------------------------------------------------------------------------
Provides strict schema validation for the E-Commerce Churn Prediction REST API.
Verifies:
- Presence of required fields
- Valid data types (numeric vs categorical)
- Value ranges and boundary constraints (e.g., non-negative tenure, valid satisfaction score)
- Membership in allowed categorical sets
"""

from typing import Dict, Any, Tuple, List, Optional


VALID_LOGIN_DEVICES = {"Mobile Phone", "Computer", "Phone"}
VALID_PAYMENT_MODES = {"Debit Card", "Credit Card", "UPI", "Cash on Delivery", "E wallet", "CC", "COD"}
VALID_GENDERS = {"Female", "Male"}
VALID_ORDER_CATS = {"Laptop & Accessory", "Mobile Phone", "Mobile", "Fashion", "Grocery", "Others"}
VALID_MARITAL_STATUS = {"Single", "Married", "Divorced"}

NUMERICAL_FIELDS = [
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

CATEGORICAL_FIELDS = [
    "preferred_login_device",
    "preferred_payment_mode",
    "gender",
    "preferred_order_cat",
    "marital_status"
]

REQUIRED_FIELDS = NUMERICAL_FIELDS + CATEGORICAL_FIELDS


def validate_prediction_payload(payload: Any) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate incoming customer payload against schema boundaries.

    Returns:
    --------
    (is_valid: bool, error_message: Optional[str], cleaned_data: Optional[Dict[str, Any]])
    """
    if not isinstance(payload, dict):
        return False, "Request payload must be a JSON object.", None

    # 1. Check for missing required fields
    missing_fields = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}", None

    cleaned = {}

    # 2. Check numerical fields and bounds
    for col in NUMERICAL_FIELDS:
        val = payload.get(col)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return False, f"Invalid type for '{col}': expected numeric, got {type(val).__name__}.", None
        
        # Non-negative check
        if val < 0:
            return False, f"Invalid value for '{col}': must be non-negative. Got {val}.", None
        
        cleaned[col] = float(val)

    # Specific boundaries
    if cleaned["satisfaction_score"] < 1 or cleaned["satisfaction_score"] > 5:
        return False, f"Invalid value for 'satisfaction_score': must be between 1 and 5. Got {cleaned['satisfaction_score']}.", None

    if cleaned["city_tier"] not in [1.0, 2.0, 3.0]:
        return False, f"Invalid value for 'city_tier': must be 1, 2, or 3. Got {cleaned['city_tier']}.", None

    if cleaned["complain"] not in [0.0, 1.0]:
        return False, f"Invalid value for 'complain': must be 0 or 1. Got {cleaned['complain']}.", None

    # 3. Check categorical fields
    dev = payload.get("preferred_login_device")
    if dev not in VALID_LOGIN_DEVICES:
        return False, f"Invalid value for 'preferred_login_device'. Allowed: {sorted(list(VALID_LOGIN_DEVICES))}.", None
    cleaned["preferred_login_device"] = "Mobile Phone" if dev == "Phone" else dev

    pay = payload.get("preferred_payment_mode")
    if pay not in VALID_PAYMENT_MODES:
        return False, f"Invalid value for 'preferred_payment_mode'. Allowed: {sorted(list(VALID_PAYMENT_MODES))}.", None
    cleaned["preferred_payment_mode"] = "Credit Card" if pay == "CC" else ("Cash on Delivery" if pay == "COD" else pay)

    gen = payload.get("gender")
    if gen not in VALID_GENDERS:
        return False, f"Invalid value for 'gender'. Allowed: {sorted(list(VALID_GENDERS))}.", None
    cleaned["gender"] = gen

    cat = payload.get("preferred_order_cat")
    if cat not in VALID_ORDER_CATS:
        return False, f"Invalid value for 'preferred_order_cat'. Allowed: {sorted(list(VALID_ORDER_CATS))}.", None
    cleaned["preferred_order_cat"] = "Mobile Phone" if cat == "Mobile" else cat

    mar = payload.get("marital_status")
    if mar not in VALID_MARITAL_STATUS:
        return False, f"Invalid value for 'marital_status'. Allowed: {sorted(list(VALID_MARITAL_STATUS))}.", None
    cleaned["marital_status"] = mar

    return True, None, cleaned


def validate_batch_payload(records: Any) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Validate a list of customer prediction payloads.

    Returns:
    --------
    (is_valid: bool, error_message: Optional[str], cleaned_records: Optional[List[Dict[str, Any]]], row_errors: List[Dict[str, Any]])
    """
    if not isinstance(records, list):
        return False, "Batch payload must be a JSON array of customer records.", None, []

    if len(records) == 0:
        return False, "Batch records array cannot be empty.", None, []

    if len(records) > 1000:
        return False, "Batch size exceeds maximum limit of 1,000 records per evaluation.", None, []

    cleaned_list = []
    row_errors = []

    for i, item in enumerate(records):
        cust_id = item.get("customer_id") or item.get("id") or f"ROW-{i + 1}"
        is_val, err_msg, cleaned_item = validate_prediction_payload(item)
        if not is_val:
            row_errors.append({
                "row_index": i + 1,
                "customer_id": str(cust_id),
                "error": err_msg
            })
        else:
            cleaned_item["customer_id"] = str(cust_id)
            cleaned_list.append(cleaned_item)

    if row_errors:
        return False, f"Validation failed on {len(row_errors)} of {len(records)} records.", cleaned_list, row_errors

    return True, None, cleaned_list, []


def parse_and_validate_csv_data(csv_text: str) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Parse CSV string and validate against required Kaggle schema boundaries.

    Returns:
    --------
    (is_valid, error_message, cleaned_records, row_errors)
    """
    import io
    import pandas as pd

    if not csv_text or not csv_text.strip():
        return False, "Uploaded CSV file is empty.", None, []

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as ex:
        return False, f"Failed to parse CSV file: {str(ex)}", None, []

    if df.empty:
        return False, "Uploaded CSV contains no customer rows.", None, []

    # Check for missing required columns
    cols = set(df.columns)
    missing_required = [f for f in REQUIRED_FIELDS if f not in cols]
    if missing_required:
        return False, f"CSV missing required columns: {', '.join(missing_required)}", None, []

    # Convert to records
    records = df.to_dict(orient="records")
    return validate_batch_payload(records)

