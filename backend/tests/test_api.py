import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_unsupported_type():
    # Sending a plain text file should fail validation
    files = {'file': ('test.txt', b'fake data', 'text/plain')}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "unsupported file type" in response.json()['detail']
