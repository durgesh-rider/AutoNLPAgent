"""
Test script to verify predictions return actual labels instead of integers
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import pandas as pd
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import model_trainer


def test_sentiment_prediction_decoding():
    """Test that sentiment predictions return actual labels like 'positive'/'negative'"""
    
    # Initialize preprocessor
    text_preprocessor = TextPreprocessor()
    
    # Create a simple sentiment dataset with more samples
    data = {
        'text': [
            'This is great!',
            'I love this product',
            'This is terrible',
            'I hate this',
            'Amazing quality',
            'Worst purchase ever',
            'Absolutely wonderful',
            'Very disappointing',
            'Excellent service',
            'Poor quality',
            'Highly recommend',
            'Do not buy this',
            'Perfect product',
            'Waste of money',
            'Outstanding performance',
            'Terrible experience',
            'Love it so much',
            'Very bad',
            'Superb quality',
            'Not worth it'
        ],
        'sentiment': [
            'positive', 'positive', 'negative', 'negative',
            'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative'
        ]
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("Testing Prediction Label Decoding")
    print("=" * 60)
    print("\nOriginal Dataset:")
    print(df)
    
    # Preprocess
    print("\n1. Preprocessing dataset...")
    processed_df, metadata = text_preprocessor.preprocess_dataset(df, 'sentiment_analysis')
    
    print(f"   Label encoder classes: {metadata['label_encoder'].classes_}")
    print(f"   Encoded labels: {processed_df['sentiment'].unique()}")
    
    # Train model
    print("\n2. Training model...")
    result = model_trainer.train_model(
        df=processed_df,
        task_type='sentiment_analysis',
        text_column='text',
        label_column='sentiment',
        model_type='sklearn',
        training_config={'epochs': 3, 'batch_size': 8, 'use_validation': True},
        label_encoder=metadata['label_encoder']  # Pass label encoder
    )
    
    if result.get('success'):
        print(f"   ✓ Model trained successfully")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Accuracy: {result['metrics'].get('accuracy', 'N/A'):.3f}")
    else:
        print(f"   ✗ Training failed: {result.get('error')}")
        return
    
    # Make predictions
    print("\n3. Making predictions...")
    test_texts = [
        "This is absolutely fantastic!",
        "I really dislike this",
        "Best thing ever"
    ]
    
    predictions = model_trainer.predict(result['model_id'], test_texts)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    for text, pred in zip(test_texts, predictions):
        print(f"\nText: '{text}'")
        print(f"Prediction: {pred} (type: {type(pred).__name__})")
        
        # Check if prediction is a string (decoded) or int (not decoded)
        if isinstance(pred, str):
            print("✓ SUCCESS: Prediction is decoded to actual label!")
        else:
            print("✗ ISSUE: Prediction is still an integer, not decoded")
    
    # Verify predictions are strings
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    all_strings = all(isinstance(p, str) for p in predictions)
    all_valid_labels = all(p in ['positive', 'negative'] for p in predictions)
    
    if all_strings:
        print("✓ All predictions are strings (decoded)")
    else:
        print("✗ Some predictions are not strings")
    
    if all_valid_labels:
        print("✓ All predictions are valid sentiment labels")
    else:
        print("✗ Some predictions are not valid sentiment labels")
    
    if all_strings and all_valid_labels:
        print("\n🎉 TEST PASSED: Predictions are properly decoded!")
        return True
    else:
        print("\n❌ TEST FAILED: Predictions are not properly decoded")
        return False


if __name__ == "__main__":
    try:
        success = test_sentiment_prediction_decoding()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
