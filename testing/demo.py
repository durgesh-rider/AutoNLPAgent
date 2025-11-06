"""
AutoNLP-Agent Demo Script
Demonstrates the complete ML pipeline
"""
import pandas as pd
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def check_server():
    """Check if server is running"""
    print_section("1. Checking Server Health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy!")
            print(f"   Response: {response.json()}")
            return True
    except Exception as e:
        print(f"❌ Server is not running!")
        print(f"   Error: {str(e)}")
        print("\n   Please start the server first:")
        print("   cd r:\\WEB\\autonlp\\autonlp\\backend")
        print("   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return False

def create_sample_dataset():
    """Create a sample dataset for testing"""
    print_section("2. Creating Sample Dataset")
    
    # Create sample sentiment analysis data
    data = {
        'review': [
            'This product is absolutely amazing! Best purchase ever!',
            'Terrible quality. Would not recommend.',
            'Pretty good, meets expectations.',
            'Worst experience. Very disappointed.',
            'Excellent service and fast delivery!',
            'Not worth the money. Poor quality.',
            'Love it! Exactly what I needed.',
            'Bad product, broke after one use.',
            'Great value for money!',
            'Awful experience, never buying again.'
        ] * 10,  # 100 rows
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ] * 10
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = "sample_sentiment_data.csv"
    df.to_csv(filename, index=False)
    
    print(f"✅ Created sample dataset: {filename}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Sample data:")
    print(df.head(3).to_string(index=False))
    
    return filename

def upload_dataset(filename):
    """Upload dataset to server"""
    print_section("3. Uploading Dataset")
    
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/upload/", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"   Dataset ID: {data['dataset_id']}")
            print(f"   Filename: {data['filename']}")
            print(f"   Rows: {data['row_count']}")
            print(f"   Columns: {data['columns']}")
            print(f"   Detected Task: {data.get('detected_task', 'N/A')}")
            return data['dataset_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def detect_task(dataset_id):
    """Detect NLP task type"""
    print_section("4. Detecting Task Type")
    
    try:
        response = requests.post(f"{BASE_URL}/tasks/detect/{dataset_id}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Task detection successful!")
            print(f"   Detected Task: {data['detected_task']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            print(f"   Reasoning: {data['reasoning']}")
            return data['detected_task']
        else:
            print(f"❌ Detection failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def preprocess_data(dataset_id):
    """Preprocess the dataset"""
    print_section("5. Preprocessing Data")
    
    try:
        response = requests.post(f"{BASE_URL}/tasks/preprocess/{dataset_id}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Preprocessing successful!")
            print(f"   Steps performed:")
            for step in data.get('preprocessing_steps', []):
                print(f"      - {step}")
            return True
        else:
            print(f"❌ Preprocessing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def train_model(dataset_id):
    """Train a model"""
    print_section("6. Training Model")
    
    print("🔄 Training in progress (this may take a moment)...")
    
    try:
        params = {'model_type': 'sklearn'}
        response = requests.post(f"{BASE_URL}/training/{dataset_id}", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Training successful!")
            print(f"   Model ID: {data.get('model_id')}")
            print(f"   Model Type: {data.get('model_type', 'N/A')}")
            print(f"   Training Time: {data.get('training_time', 0):.2f}s")
            print(f"   Training Samples: {data.get('training_samples', 'N/A')}")
            print(f"   Test Samples: {data.get('test_samples', 'N/A')}")
            return data.get('model_id')
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def evaluate_model(model_id):
    """Evaluate the trained model"""
    print_section("7. Evaluating Model")
    
    try:
        response = requests.post(f"{BASE_URL}/evaluation/{model_id}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Evaluation successful!")
            
            metrics = data.get('metrics', {})
            print(f"\n   📊 Performance Metrics:")
            print(f"      Accuracy:  {metrics.get('accuracy', 0):.2%}")
            print(f"      Precision: {metrics.get('precision_weighted', 0):.2%}")
            print(f"      Recall:    {metrics.get('recall_weighted', 0):.2%}")
            print(f"      F1-Score:  {metrics.get('f1_weighted', 0):.2%}")
            
            return True
        else:
            print(f"❌ Evaluation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run the complete demo"""
    print("\n" + "="*60)
    print("  🤖 AutoNLP-Agent - Complete Pipeline Demo")
    print("="*60)
    print("\nThis demo will:")
    print("  1. Check server health")
    print("  2. Create a sample dataset")
    print("  3. Upload the dataset")
    print("  4. Detect the task type")
    print("  5. Preprocess the data")
    print("  6. Train a model")
    print("  7. Evaluate the model")
    
    input("\nPress Enter to start...")
    
    # Step 1: Check server
    if not check_server():
        return
    
    time.sleep(1)
    
    # Step 2: Create sample dataset
    filename = create_sample_dataset()
    time.sleep(1)
    
    # Step 3: Upload dataset
    dataset_id = upload_dataset(filename)
    if not dataset_id:
        return
    time.sleep(1)
    
    # Step 4: Detect task
    task_type = detect_task(dataset_id)
    if not task_type:
        return
    time.sleep(1)
    
    # Step 5: Preprocess data
    if not preprocess_data(dataset_id):
        return
    time.sleep(1)
    
    # Step 6: Train model
    model_id = train_model(dataset_id)
    if not model_id:
        return
    time.sleep(1)
    
    # Step 7: Evaluate model
    evaluate_model(model_id)
    
    # Final summary
    print_section("✅ Demo Complete!")
    print("The complete ML pipeline has been executed successfully!")
    print("\n📝 Summary:")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Task Type:  {task_type}")
    print(f"   Model ID:   {model_id}")
    print("\n🌐 Explore more:")
    print(f"   API Docs: {BASE_URL}/docs")
    print(f"   Health:   {BASE_URL}/health")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
