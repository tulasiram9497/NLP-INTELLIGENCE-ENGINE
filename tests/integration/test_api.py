from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sentiment_analysis_endpoint():
    response = client.post(
        "/api/v1/sentiment",
        json={"text": "I love this product and it works perfectly."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= payload["confidence"] <= 1.0
