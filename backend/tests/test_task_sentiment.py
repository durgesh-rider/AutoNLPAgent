"""
Comprehensive tests for Sentiment Analysis task
"""
import pytest
import pandas as pd
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import model_trainer
from app.core.task_detector import detect_task_type


class TestSentimentAnalysis:
    """Test suite for sentiment analysis tasks"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.preprocessor = TextPreprocessor()
    
    def test_binary_sentiment(self):
        """Test binary sentiment (positive/negative)"""
        data = {
            'review': [
                'I love this product amazing',
                'Terrible waste of money',
                'Excellent quality highly recommend',
                'Very bad experience',
                'Outstanding performance',
                'Horrible product',
                'Best purchase ever',
                'Disappointed not good',
                'Fantastic service',
                'Poor quality'
            ] * 4,
            'sentiment': ['positive', 'negative'] * 20
        }
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'sentiment_analysis')
        
        # Train
        result = model_trainer.train_model(
            df=processed_df,
            task_type='sentiment_analysis',
            text_column='review',
            label_column='sentiment',
            model_type='sklearn',
            training_config={'epochs': 5, 'use_validation': True},
            label_encoder=metadata.get('label_encoder')
        )
        
        assert result['success'] == True
        assert result['task_type'] == 'classification'  # Sentiment is treated as classification
        assert 'accuracy' in result['metrics']
        
        # Predict
        predictions = model_trainer.predict(result['model_id'], [
            'This is absolutely fantastic',
            'Terrible experience hate it'
        ])
        
        assert len(predictions) == 2
        assert all(p in ['positive', 'negative'] for p in predictions)
    
    def test_multiclass_sentiment(self):
        """Test multiclass sentiment (positive/neutral/negative)"""
        data = {
            'text': [
                'I love this',
                'It is okay',
                'I hate this',
                'Amazing product',
                'Works fine',
                'Terrible quality',
                'Highly recommend',
                'Average performance',
                'Very disappointed',
                'Excellent service',
                'Neutral opinion',
                'Worst purchase'
            ] * 4,
            'sentiment': ['positive', 'neutral', 'negative'] * 16
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'sentiment_analysis')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='sentiment_analysis',
            text_column='text',
            label_column='sentiment',
            model_type='sklearn',
            training_config={'epochs': 5, 'batch_size': 16},
            label_encoder=metadata.get('label_encoder')
        )
        
        assert result['success'] == True
        
        predictions = model_trainer.predict(result['model_id'], [
            'Great product',
            'It works',
            'Bad quality'
        ])
        
        assert len(predictions) == 3
        assert all(p in ['positive', 'neutral', 'negative'] for p in predictions)
    
    def test_sentiment_task_detection(self):
        """Test sentiment analysis is properly detected"""
        data = {
            'review_text': ['Review ' + str(i) for i in range(20)],
            'sentiment': ['positive' if i % 2 == 0 else 'negative' for i in range(20)]
        }
        df = pd.DataFrame(data)
        
        detected_task = detect_task_type(df)
        assert detected_task == 'sentiment_analysis'
    
    def test_sentiment_with_ratings(self):
        """Test sentiment with star ratings (1-5)"""
        data = {
            'review': ['Review text ' + str(i) for i in range(50)],
            'rating': [i % 5 + 1 for i in range(50)]  # 1 to 5 stars
        }
        df = pd.DataFrame(data)
        
        # Convert ratings to sentiment
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else ('neutral' if x == 3 else 'negative'))
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'sentiment_analysis')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='sentiment_analysis',
            text_column='review',
            label_column='sentiment',
            model_type='sklearn',
            training_config={'epochs': 3},
            label_encoder=metadata.get('label_encoder')
        )
        
        assert result['success'] == True
        assert result['metrics']['accuracy'] >= 0.0
    
    def test_sentiment_prediction_confidence(self):
        """Test sentiment prediction returns valid labels"""
        data = {
            'text': [
                'Absolutely love it',
                'Terrible product',
                'Okay I guess',
                'Best thing ever',
                'Worst purchase',
                'Fine for the price'
            ] * 6,
            'sentiment': ['positive', 'negative', 'neutral'] * 12
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'sentiment_analysis')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='sentiment_analysis',
            text_column='text',
            label_column='sentiment',
            model_type='sklearn',
            label_encoder=metadata.get('label_encoder')
        )
        
        # Test batch prediction
        test_texts = [
            'I really enjoy this product',
            'Not satisfied at all',
            'It works as expected',
            'Hate it so much',
            'Pretty good overall'
        ]
        
        predictions = model_trainer.predict(result['model_id'], test_texts)
        
        assert len(predictions) == len(test_texts)
        assert all(isinstance(p, str) for p in predictions)
        assert all(p in ['positive', 'negative', 'neutral'] for p in predictions)
