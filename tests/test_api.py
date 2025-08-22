import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_prediction_payload():
    """Provides a sample payload for testing the /predict endpoint."""
    return {
        "tenure": 120,
        "total_songs": 500,
        "total_listen_time": 120000.0,
        "num_artists": 150,
        "num_thumbs_up": 20,
        "num_thumbs_down": 2,
        "num_sessions": 30,
        "num_friends_added": 5,
        "num_downgrades": 0,
        "avg_songs_per_session": 16.67,
        "gender_Male": True,
        "last_level_paid": True,
        "os_Windows": True,
    }


def test_read_root():
    """Test the root endpoint for a successful response."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Welcome to the Churn Prediction API",
    }


def test_predict_endpoint_success(sample_prediction_payload):
    """Test the /predict endpoint for a successful prediction."""
    model_exists = os.path.exists("ml_artifacts/random_forest_churn_model.pkl")
    features_exist = os.path.exists("ml_artifacts/feature_list.joblib")

    if not model_exists or not features_exist:
        pytest.skip(
            "Model artifacts not found, skipping prediction test. "
            "Run `make train` first."
        )

    response = client.post("/predict", json=sample_prediction_payload)

    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert data["error"] is None
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_endpoint_invalid_payload():
    """Test the /predict endpoint with a missing required field."""
    invalid_payload = {
        "tenure": 120,
        # "total_songs" is missing
    }
    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"] == "Field required"
    assert data["detail"]["loc"] == ["body", "total_songs"]
