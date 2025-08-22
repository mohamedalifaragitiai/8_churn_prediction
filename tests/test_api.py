import pytest
from fastapi.testclient import TestClient
from api.main import app
import os

# Create a test client for the FastAPI app
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
    assert response.json() == {"status": "ok", "message": "Welcome to the Churn Prediction API"}

def test_predict_endpoint_success(sample_prediction_payload):
    """Test the /predict endpoint for a successful prediction."""
    # Ensure the model and feature artifacts exist before testing
    if not os.path.exists("ml_artifacts/lgbm_churn_model.pkl") or not os.path.exists("ml_artifacts/feature_list.joblib"):
        pytest.skip("Model artifacts not found, skipping prediction test. Run `make train` first.")
    
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
    
    # FastAPI's Pydantic validation should catch this
    assert response.status_code == 422 # Unprocessable Entity
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"] == "Field required"
    assert data["detail"][0]["loc"] == ["body", "total_songs"]

