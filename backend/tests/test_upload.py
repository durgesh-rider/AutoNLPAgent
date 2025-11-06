import pytest
from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_csv_file():
    """Test uploading a CSV file"""
    # Create a sample CSV file
    csv_content = "text,label\nHello world,positive\nThis is bad,negative\n"
    file = io.BytesIO(csv_content.encode('utf-8'))
    file.name = "test.csv"

    response = client.post(
        "/upload/",
        files={"file": ("test.csv", file, "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "dataset_id" in data
    assert data["detected_task"] in ["classification", "sentiment_analysis", "unknown"]


def test_upload_invalid_file_type():
    """Test uploading an invalid file type"""
    file = io.BytesIO(b"invalid content")
    file.name = "test.exe"

    response = client.post(
        "/upload/",
        files={"file": ("test.exe", file, "application/octet-stream")}
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_get_datasets():
    """Test getting all datasets"""
    response = client.get("/upload/datasets")
    assert response.status_code == 200
    assert "datasets" in response.json()


def test_task_detection():
    """Test task detection endpoint"""
    # First upload a dataset
    csv_content = "review,sentiment\nGreat product!,positive\nBad quality,negative\n"
    file = io.BytesIO(csv_content.encode('utf-8'))
    file.name = "sentiment.csv"

    upload_response = client.post(
        "/upload/",
        files={"file": ("sentiment.csv", file, "text/csv")}
    )

    dataset_id = upload_response.json()["dataset_id"]

    # Test task detection
    response = client.post(f"/tasks/detect/{dataset_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detected_task" in data


def test_preprocessing():
    """Test data preprocessing endpoint"""
    # First upload a dataset
    csv_content = "text,label\nHello world!,positive\nThis is bad.,negative\n"
    file = io.BytesIO(csv_content.encode('utf-8'))
    file.name = "preprocess.csv"

    upload_response = client.post(
        "/upload/",
        files={"file": ("preprocess.csv", file, "text/csv")}
    )

    dataset_id = upload_response.json()["dataset_id"]

    # Test preprocessing
    response = client.post(f"/tasks/preprocess/{dataset_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "preprocessing_steps" in data


def test_supported_tasks():
    """Test getting supported tasks"""
    response = client.get("/tasks/supported")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) > 0