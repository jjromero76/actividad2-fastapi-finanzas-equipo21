from fastapi.testclient import TestClient
import pytest
from financial_api.api import app
from financial_api.train import train_model

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_artifacts():
    """Garantiza la presencia del modelo antes de ejecutar las pruebas."""
    train_model()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_market_data_endpoint():
    response = client.get("/market-data/AAPL")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"
    assert "close" in response.json()

def test_predict_endpoint():
    payload = {"symbol": "AAPL", "prediction_horizon": 1, "use_cached_data": True}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["up", "down"]
    assert 0.0 <= response.json()["probability_up"] <= 1.0

def test_metadata_endpoint():
    response = client.get("/model/metadata")
    assert response.status_code == 200
    assert "accuracy_metric" in response.json()