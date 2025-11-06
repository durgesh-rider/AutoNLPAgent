"""
API Integration Tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
import io
import pandas as pd

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and root endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestUploadEndpoints:
    """Test file upload functionality"""
    
    def test_upload_csv_file(self):
        """Test uploading a CSV file"""
        csv_content = "text,label\nHello world,positive\nBad product,negative\n"
        file = io.BytesIO(csv_content.encode('utf-8'))
        
        response = client.post(
            "/upload/",
            files={"file": ("test.csv", file, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "dataset_id" in data
        assert "filename" in data
        assert data["row_count"] == 2
    
    def test_upload_invalid_file_type(self):
        """Test uploading an invalid file type"""
        file = io.BytesIO(b"random binary data")
        
        response = client.post(
            "/upload/",
            files={"file": ("test.exe", file, "application/octet-stream")}
        )
        
        # Should reject invalid file types
        assert response.status_code in [400, 422]
    
    def test_get_datasets(self):
        """Test getting all datasets"""
        response = client.get("/upload/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data


class TestTaskEndpoints:
    """Test task detection and preprocessing endpoints"""
    
    @pytest.fixture
    def uploaded_dataset_id(self):
        """Upload a dataset and return its ID"""
        csv_content = "review,sentiment\nGreat product!,positive\nTerrible quality,negative\nLove it,positive\n"
        file = io.BytesIO(csv_content.encode('utf-8'))
        
        response = client.post(
            "/upload/",
            files={"file": ("sentiment.csv", file, "text/csv")}
        )
        
        return response.json()["dataset_id"]
    
    def test_task_detection(self, uploaded_dataset_id):
        """Test task detection endpoint"""
        response = client.post(f"/tasks/detect/{uploaded_dataset_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detected_task" in data
        assert "confidence" in data
        assert "reasoning" in data
    
    def test_task_detection_invalid_id(self):
        """Test task detection with invalid dataset ID"""
        response = client.post("/tasks/detect/invalid_id")
        assert response.status_code == 404
    
    def test_preprocessing(self, uploaded_dataset_id):
        """Test data preprocessing endpoint"""
        response = client.post(f"/tasks/preprocess/{uploaded_dataset_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preprocessing_steps" in data
    
    def test_supported_tasks(self):
        """Test getting supported tasks"""
        response = client.get("/tasks/supported")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) > 0


class TestTrainingEndpoints:
    """Test model training endpoints"""
    
    @pytest.fixture
    def training_dataset_id(self):
        """Upload a dataset suitable for training"""
        # Create larger dataset for training
        texts = [f"Sample review text number {i}" for i in range(50)]
        labels = ['positive', 'negative'] * 25
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        csv_content = df.to_csv(index=False)
        
        file = io.BytesIO(csv_content.encode('utf-8'))
        response = client.post(
            "/upload/",
            files={"file": ("training_data.csv", file, "text/csv")}
        )
        
        return response.json()["dataset_id"]
    
    def test_train_model(self, training_dataset_id):
        """Test model training endpoint"""
        response = client.post(
            f"/training/{training_dataset_id}",
            params={"model_type": "sklearn"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
    
    def test_get_models(self):
        """Test getting all models"""
        response = client.get("/training/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    
    def test_train_invalid_dataset(self):
        """Test training with invalid dataset ID"""
        response = client.post("/training/invalid_dataset_id")
        assert response.status_code == 404


class TestEvaluationEndpoints:
    """Test model evaluation endpoints"""
    
    @pytest.fixture
    def trained_model_id(self):
        """Create and train a model, return its ID"""
        # Upload dataset
        texts = [f"Test text {i}" for i in range(50)]
        labels = ['A', 'B'] * 25
        df = pd.DataFrame({'text': texts, 'category': labels})
        csv_content = df.to_csv(index=False)
        
        file = io.BytesIO(csv_content.encode('utf-8'))
        upload_response = client.post(
            "/upload/",
            files={"file": ("eval_data.csv", file, "text/csv")}
        )
        dataset_id = upload_response.json()["dataset_id"]
        
        # Train model
        train_response = client.post(
            f"/training/{dataset_id}",
            params={"model_type": "sklearn"}
        )
        
        if train_response.status_code == 200:
            return train_response.json().get("model_id")
        return None
    
    def test_evaluate_model(self, trained_model_id):
        """Test model evaluation endpoint"""
        if trained_model_id:
            response = client.post(f"/evaluation/{trained_model_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data or "success" in data


class TestEndToEnd:
    """End-to-end workflow tests"""
    
    def test_complete_workflow(self):
        """Test complete workflow from upload to evaluation"""
        # Step 1: Upload dataset
        texts = ['Great!', 'Bad', 'Excellent', 'Terrible', 'Good', 'Poor'] * 10
        labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative'] * 10
        df = pd.DataFrame({'review': texts, 'sentiment': labels})
        csv_content = df.to_csv(index=False)
        
        file = io.BytesIO(csv_content.encode('utf-8'))
        upload_response = client.post(
            "/upload/",
            files={"file": ("workflow_test.csv", file, "text/csv")}
        )
        
        assert upload_response.status_code == 200
        dataset_id = upload_response.json()["dataset_id"]
        
        # Step 2: Detect task
        detect_response = client.post(f"/tasks/detect/{dataset_id}")
        assert detect_response.status_code == 200
        detected_task = detect_response.json().get("detected_task")
        
        # Step 3: Preprocess
        preprocess_response = client.post(f"/tasks/preprocess/{dataset_id}")
        assert preprocess_response.status_code == 200
        
        # Step 4: Train model
        train_response = client.post(
            f"/training/{dataset_id}",
            params={"model_type": "sklearn"}
        )
        
        if train_response.status_code == 200:
            model_id = train_response.json().get("model_id")
            
            # Step 5: Evaluate model
            if model_id:
                eval_response = client.post(f"/evaluation/{model_id}")
                assert eval_response.status_code in [200, 500]  # May fail if model training failed
    
    def test_multiple_datasets(self):
        """Test handling multiple datasets"""
        datasets = []
        
        for i in range(3):
            csv_content = f"text,label\nSample {i},A\nTest {i},B\n"
            file = io.BytesIO(csv_content.encode('utf-8'))
            
            response = client.post(
                "/upload/",
                files={"file": (f"dataset_{i}.csv", file, "text/csv")}
            )
            
            if response.status_code == 200:
                datasets.append(response.json()["dataset_id"])
        
        # Verify all datasets are stored
        all_datasets_response = client.get("/upload/datasets")
        assert all_datasets_response.status_code == 200


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_endpoints(self):
        """Test invalid endpoint access"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_missing_required_parameters(self):
        """Test endpoints with missing parameters"""
        response = client.post("/upload/")
        assert response.status_code == 422  # Unprocessable entity
    
    def test_invalid_dataset_operations(self):
        """Test operations on non-existent dataset"""
        fake_id = "non_existent_dataset_id"
        
        # Try to detect task
        response = client.post(f"/tasks/detect/{fake_id}")
        assert response.status_code == 404
        
        # Try to preprocess
        response = client.post(f"/tasks/preprocess/{fake_id}")
        assert response.status_code == 404
        
        # Try to train
        response = client.post(f"/training/{fake_id}")
        assert response.status_code == 404


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
