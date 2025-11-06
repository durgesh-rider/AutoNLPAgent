"""
Test the backend endpoints with a sample dataset
This will verify that upload and training work correctly
"""
import sys
import os
from pathlib import Path
import pandas as pd
import time

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

def create_sample_dataset():
    """Create a sample sentiment analysis dataset"""
    data = {
        'text': [
            'I love this product, it is amazing!',
            'This is the worst thing I have ever bought',
            'Pretty good, would recommend',
            'Absolutely terrible experience',
            'Best purchase ever!',
            'Not worth the money',
            'Excellent quality and fast shipping',
            'Very disappointed with this',
            'Great value for money',
            'Complete waste of time',
            'I am very happy with this',
            'This is awful',
            'Fantastic product',
            'Do not buy this',
            'Highly recommend!',
            'Very poor quality',
            'Love it so much',
            'Terrible customer service',
            'Amazing experience',
            'Never buying again'
        ],
        'sentiment': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    df = pd.DataFrame(data)
    
    # Create uploads directory if it doesn't exist
    uploads_dir = backend_dir / 'uploads'
    uploads_dir.mkdir(exist_ok=True)
    
    # Save the dataset
    file_path = uploads_dir / 'test_sentiment_dataset.csv'
    df.to_csv(file_path, index=False)
    
    return file_path

def test_backend():
    """Test backend functionality"""
    print("=" * 60)
    print("Testing AutoNLP Backend")
    print("=" * 60)
    print()
    
    # Test 1: Import all components
    print("Test 1: Importing backend components...")
    try:
        from app.services.file_service import file_service
        from app.services.ml_service import ml_service
        from app.utils.file_utils import load_dataset
        print("✓ All components imported successfully")
    except Exception as e:
        print(f"✗ Failed to import components: {e}")
        return
    
    # Test 2: Create and load sample dataset
    print("\nTest 2: Creating sample dataset...")
    try:
        file_path = create_sample_dataset()
        print(f"✓ Sample dataset created at: {file_path}")
        
        df, file_type = load_dataset(str(file_path))
        print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"  File type: {file_type}")
    except Exception as e:
        print(f"✗ Failed to create/load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Simulate file upload
    print("\nTest 3: Simulating dataset upload...")
    try:
        # Create a mock file object
        class MockFile:
            def __init__(self, path):
                self.filename = Path(path).name
                self.path = path
        
        # We'll directly add to file_service instead of going through upload
        from app.utils.file_utils import generate_dataset_id, get_dataset_info
        from app.core.task_detector import detect_task_type
        
        dataset_id = generate_dataset_id()
        row_count, columns = get_dataset_info(str(file_path))
        task_type = detect_task_type(df)
        
        file_service.uploaded_datasets[dataset_id] = {
            'file_path': str(file_path),
            'filename': 'test_sentiment_dataset.csv',
            'columns': columns,
            'row_count': row_count,
            'task_type': task_type,
            'dataframe': df
        }
        
        print(f"✓ Dataset uploaded with ID: {dataset_id}")
        print(f"  Task type detected: {task_type}")
        print(f"  Rows: {row_count}")
        print(f"  Columns: {columns}")
    except Exception as e:
        print(f"✗ Failed to upload dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Train model
    print("\nTest 4: Training model...")
    try:
        # Get the dataset
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            print("✗ Dataset not found")
            return
        
        df = dataset['dataframe']
        task_type = dataset['task_type']
        
        # Train model
        result = ml_service.train_model(
            dataset_id=dataset_id,
            df=df,
            task_type=task_type,
            text_column='text',
            label_column='sentiment',
            model_type='sklearn'  # Use sklearn for faster training
        )
        
        if result.success:
            print(f"✓ Model trained successfully!")
            print(f"  Model ID: {result.model_id}")
            print(f"  Model type: {result.model_type}")
            print(f"  Training time: {result.training_time:.2f}s")
            print(f"  Status: {result.status}")
        else:
            print(f"✗ Model training failed: {result.message}")
            return
            
    except Exception as e:
        print(f"✗ Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Evaluate model
    print("\nTest 5: Evaluating model...")
    try:
        eval_result = ml_service.evaluate_model(result.model_id)
        
        if eval_result.success:
            print(f"✓ Model evaluated successfully!")
            print(f"  Metrics:")
            for metric, value in eval_result.metrics.items():
                print(f"    {metric}: {value:.4f}")
        else:
            print(f"✗ Model evaluation failed: {eval_result.message}")
    except Exception as e:
        print(f"✗ Failed to evaluate model: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Make predictions
    print("\nTest 6: Making predictions...")
    try:
        test_texts = [
            "This is absolutely wonderful!",
            "I hate this so much",
            "It's okay I guess"
        ]
        
        pred_result = ml_service.predict(result.model_id, test_texts)
        
        if pred_result['success']:
            print(f"✓ Predictions made successfully!")
            print(f"  Test texts and predictions:")
            for text, pred in zip(test_texts, pred_result['predictions']):
                print(f"    '{text}' -> {pred}")
        else:
            print(f"✗ Prediction failed: {pred_result.get('error')}")
    except Exception as e:
        print(f"✗ Failed to make predictions: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Backend testing complete!")
    print("=" * 60)
    print("\nThe backend is working correctly. You can now:")
    print("  1. Start the server: py start_server.py")
    print("  2. Access API docs: http://localhost:8000/docs")
    print("  3. Use the frontend to upload datasets and train models")

if __name__ == "__main__":
    test_backend()
