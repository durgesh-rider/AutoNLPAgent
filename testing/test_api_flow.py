"""
Test script to verify all AutoNLP-Agent endpoints are working
Tests the complete flow: Upload → Training → Evaluation → Prediction
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_FILE = Path(__file__).parent.parent / "backend" / "uploads" / "002be6dc-dbf7-49fe-a62c-34525059090f.csv"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_health():
    """Test health check endpoint"""
    print_header("Testing Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_upload(file_path):
    """Test file upload endpoint"""
    print_header("Testing File Upload")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/upload/", files=files)
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Dataset ID: {data.get('dataset_id')}")
        print(f"Task Type: {data.get('task_type')}")
        print(f"Rows: {data.get('row_count')}")
        
        return data.get('dataset_id') if response.status_code == 200 else None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_get_datasets():
    """Test get datasets endpoint"""
    print_header("Testing Get Datasets")
    try:
        response = requests.get(f"{BASE_URL}/upload/datasets")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Number of datasets: {len(data.get('datasets', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_dataset_info(dataset_id):
    """Test get specific dataset info"""
    print_header(f"Testing Get Dataset Info: {dataset_id}")
    try:
        response = requests.get(f"{BASE_URL}/upload/datasets/{dataset_id}")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Task Type: {data.get('task_type')}")
        print(f"Columns: {data.get('columns', [])}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_training(dataset_id):
    """Test model training endpoint"""
    print_header(f"Testing Training: {dataset_id}")
    try:
        config = {
            "model_type": "transformer",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.00002,
            "max_length": 128,
            "test_size": 0.2,
            "use_validation": True,
            "early_stopping": False
        }
        
        print("Sending training request...")
        print(f"Config: {json.dumps(config, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/training/{dataset_id}",
            json=config,
            timeout=300  # 5 minute timeout
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Model ID: {data.get('model_id')}")
        print(f"Success: {data.get('success')}")
        
        return data.get('model_id') if response.status_code == 200 else None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_get_models():
    """Test get models endpoint"""
    print_header("Testing Get Models")
    try:
        response = requests.get(f"{BASE_URL}/training/models")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Number of models: {len(data.get('models', {}))}")
        for model_id, model_info in list(data.get('models', {}).items())[:3]:
            print(f"  - {model_id}: {model_info.get('task_type')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_model_info(model_id):
    """Test get specific model info"""
    print_header(f"Testing Get Model Info: {model_id}")
    try:
        response = requests.get(f"{BASE_URL}/training/models/{model_id}")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Task Type: {data.get('task_type')}")
        print(f"Model Type: {data.get('model_info', {}).get('model_type')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_metrics(model_id):
    """Test get model metrics endpoint"""
    print_header(f"Testing Get Metrics: {model_id}")
    try:
        response = requests.get(f"{BASE_URL}/evaluation/metrics/{model_id}")
        print(f"Status: {response.status_code}")
        data = response.json()
        metrics = data.get('metrics', {})
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction(model_id):
    """Test prediction endpoint"""
    print_header(f"Testing Prediction: {model_id}")
    try:
        payload = {
            "texts": ["This is a test text for prediction"],
            "return_probabilities": True
        }
        
        response = requests.post(
            f"{BASE_URL}/training/predict/{model_id}",
            json=payload
        )
        
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Predictions: {data.get('predictions')}")
        if data.get('probabilities'):
            print(f"Probabilities: {data.get('probabilities')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🚀 AutoNLP-Agent API Testing Suite" + "\n")
    
    results = {}
    
    # Test 1: Health Check
    results['health'] = test_health()
    time.sleep(1)
    
    # Test 2: Upload (use existing file if available)
    if TEST_FILE.exists():
        dataset_id = test_upload(TEST_FILE)
        results['upload'] = dataset_id is not None
    else:
        print("⚠️  Test file not found, skipping upload test")
        # Try to get existing dataset
        response = requests.get(f"{BASE_URL}/upload/datasets")
        datasets = response.json().get('datasets', [])
        dataset_id = datasets[0]['dataset_id'] if datasets else None
        results['upload'] = False
    
    time.sleep(1)
    
    # Test 3: Get Datasets
    results['get_datasets'] = test_get_datasets()
    time.sleep(1)
    
    # Test 4: Get Dataset Info
    if dataset_id:
        results['get_dataset_info'] = test_get_dataset_info(dataset_id)
        time.sleep(1)
    
    # Test 5: Training (optional - takes time)
    train_model = input("\n🤔 Do you want to test model training? (y/n): ").lower() == 'y'
    if train_model and dataset_id:
        model_id = test_training(dataset_id)
        results['training'] = model_id is not None
        time.sleep(2)
    else:
        # Try to get existing model
        response = requests.get(f"{BASE_URL}/training/models")
        models = response.json().get('models', {})
        model_id = list(models.keys())[0] if models else None
        results['training'] = False
    
    # Test 6: Get Models
    results['get_models'] = test_get_models()
    time.sleep(1)
    
    # Test 7: Get Model Info
    if model_id:
        results['get_model_info'] = test_get_model_info(model_id)
        time.sleep(1)
        
        # Test 8: Get Metrics
        results['get_metrics'] = test_get_metrics(model_id)
        time.sleep(1)
        
        # Test 9: Prediction
        results['prediction'] = test_prediction(model_id)
    
    # Print Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total*100):.1f}%\n")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print("\n" + "="*60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
