"""Contract and validation tests for the Task 10 API."""

from fastapi.testclient import TestClient

from task10_mlops_pipeline.app.main import app


VALID_READING = {
    "product_type": "M",
    "air_temperature_k": 300.1,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 40.0,
    "tool_wear_min": 100.0,
}


def test_health_reports_loaded_model():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "Random Forest",
    }


def test_valid_prediction_contract():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_READING)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["failure_probability"] <= 1
    assert body["alarm_threshold"] == 0.81
    assert isinstance(body["machine_failure_alarm"], bool)
    assert body["decision"] in {"maintenance_required", "normal_operation"}


def test_text_in_numeric_field_returns_400():
    invalid = {**VALID_READING, "torque_nm": "forty"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_invalid_product_type_returns_400():
    invalid = {**VALID_READING, "product_type": "X"}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_out_of_range_sensor_returns_400():
    invalid = {**VALID_READING, "rotational_speed_rpm": -10.0}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_unknown_field_returns_400():
    invalid = {**VALID_READING, "unknown_sensor": 123.0}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_malformed_json_returns_400():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=b"{not-json}",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400


def test_batch_rejects_empty_payload():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json=[])
    assert response.status_code == 400
