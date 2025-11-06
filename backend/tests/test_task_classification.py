"""
Comprehensive tests for Text Classification task
"""
import pytest
import pandas as pd
import numpy as np
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import model_trainer
from app.core.task_detector import detect_task_type


class TestTextClassification:
    """Test suite for text classification tasks"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.preprocessor = TextPreprocessor()
    
    def test_binary_classification(self):
        """Test binary classification (2 classes)"""
        # Create binary classification dataset
        data = {
            'text': [
                'This is a spam message',
                'This is not spam',
                'Buy now limited offer',
                'Hello how are you',
                'Win a free prize',
                'See you tomorrow',
                'Click here to win',
                'Thanks for your help',
                'Congratulations you won',
                'Meeting at 3pm'
            ] * 3,  # Repeat for more samples
            'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 3
        }
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'classification')
        
        # Train model
        result = model_trainer.train_model(
            df=processed_df,
            task_type='classification',
            text_column='text',
            label_column='label',
            model_type='sklearn',
            training_config={'epochs': 3, 'batch_size': 8, 'use_validation': True},
            label_encoder=metadata.get('label_encoder')
        )

        assert result['success'] == True
        assert 'model_id' in result
        assert result['task_type'] == 'classification'
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        
        # Test prediction
        test_texts = ['Buy this product now', 'How was your day']
        predictions = model_trainer.predict(result['model_id'], test_texts)
        
        assert len(predictions) == 2
        assert all(isinstance(p, str) for p in predictions)
        assert all(p in ['spam', 'ham'] for p in predictions)
    
    def test_multiclass_classification(self):
        """Test multiclass classification (3+ classes)"""
        data = {
            'text': [
                'I love this product',
                'This is okay',
                'Terrible experience',
                'Amazing quality',
                'It works fine',
                'Waste of money',
                'Highly recommend',
                'Average product',
                'Very disappointed',
                'Excellent service',
                'Meets expectations',
                'Not worth it'
            ] * 3,
            'category': [
                'positive', 'neutral', 'negative',
                'positive', 'neutral', 'negative',
                'positive', 'neutral', 'negative',
                'positive', 'neutral', 'negative'
            ] * 3
        }
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'classification')
        
        # Train
        result = model_trainer.train_model(
            df=processed_df,
            task_type='classification',
            text_column='text',
            label_column='category',
            model_type='sklearn',
            training_config={'epochs': 5, 'use_validation': True},
            label_encoder=metadata.get('label_encoder')
        )
        
        assert result['success'] == True
        assert result['metrics']['accuracy'] >= 0.0
        
        # Predict
        predictions = model_trainer.predict(result['model_id'], ['Great product', 'Could be better'])
        assert len(predictions) == 2
        assert all(p in ['positive', 'neutral', 'negative'] for p in predictions)
    
    def test_classification_with_various_configs(self):
        """Test classification with different training configurations"""
        data = {
            'text': ['Text ' + str(i) for i in range(100)],
            'label': ['class_' + str(i % 4) for i in range(100)]
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'classification')
        
        configs = [
            {'epochs': 1, 'batch_size': 16, 'use_validation': False},
            {'epochs': 3, 'batch_size': 32, 'use_validation': True},
            {'epochs': 5, 'batch_size': 8, 'use_validation': True, 'test_size': 0.3}
        ]
        
        for config in configs:
            result = model_trainer.train_model(
                df=processed_df,
                task_type='classification',
                text_column='text',
                label_column='label',
                model_type='sklearn',
                training_config=config,
                label_encoder=metadata.get('label_encoder')
            )
            
            assert result['success'] == True
            assert 'training_time' in result
            assert result['training_time'] > 0
    
    def test_classification_task_detection(self):
        """Test that classification is properly detected"""
        data = {
            'content': ['Sample text ' + str(i) for i in range(20)],
            'category': ['cat_' + str(i % 3) for i in range(20)]
        }
        df = pd.DataFrame(data)
        
        detected_task = detect_task_type(df)
        assert detected_task == 'classification'
    
    def test_classification_empty_prediction(self):
        """Test classification handles empty text gracefully"""
        data = {
            'text': ['Text ' + str(i) for i in range(30)],
            'label': ['A' if i % 2 == 0 else 'B' for i in range(30)]
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'classification')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='classification',
            text_column='text',
            label_column='label',
            model_type='sklearn',
            training_config={'epochs': 2},
            label_encoder=metadata.get('label_encoder')
        )
        
        # Test with empty and whitespace strings
        predictions = model_trainer.predict(result['model_id'], ['', '  ', 'Normal text'])
        assert len(predictions) == 3
        assert all(isinstance(p, str) for p in predictions)
