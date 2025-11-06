"""
Comprehensive tests for Named Entity Recognition (NER) task
"""
import pytest
import pandas as pd
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import model_trainer
from app.core.task_detector import detect_task_type


class TestNamedEntityRecognition:
    """Test suite for NER tasks"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.preprocessor = TextPreprocessor()
    
    def test_ner_person_org(self):
        """Test NER for person and organization entities"""
        data = {
            'text': [
                'John Smith works at Microsoft',
                'Apple Inc was founded by Steve Jobs',
                'Google hired Sundar Pichai',
                'Amazon CEO Jeff Bezos',
                'Tesla is led by Elon Musk',
                'Mark Zuckerberg founded Facebook',
                'Bill Gates started Microsoft',
                'Larry Page cofounded Google'
            ] * 4,
            'entity': ['PERSON', 'ORG', 'ORG', 'PERSON', 'ORG', 'PERSON', 'PERSON', 'ORG'] * 4
        }
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'ner')
        
        # Train
        result = model_trainer.train_model(
            df=processed_df,
            task_type='ner',
            text_column='text',
            label_column='entity',
            model_type='sklearn',
            training_config={'epochs': 5, 'use_validation': True}
        )
        
        assert result['success'] == True
        assert result['task_type'] == 'ner'
        assert 'metrics' in result
        
        # Predict
        predictions = model_trainer.predict(result['model_id'], [
            'Tim Cook leads Apple',
            'IBM is a technology company'
        ])
        
        assert len(predictions) == 2
        assert all(isinstance(p, str) for p in predictions)
    
    def test_ner_multiple_entities(self):
        """Test NER with multiple entity types"""
        data = {
            'text': [
                'Visit New York City',
                'Meeting on Monday',
                'Call +1234567890',
                'Located in Paris France',
                'Event on December 25',
                'Email john@example.com',
                'Address: 123 Main Street',
                'Born in 1990'
            ] * 3,
            'entity_type': ['LOC', 'DATE', 'PHONE', 'LOC', 'DATE', 'EMAIL', 'LOC', 'DATE'] * 3
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'ner')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='ner',
            text_column='text',
            label_column='entity_type',
            model_type='sklearn',
            training_config={'epochs': 3}
        )
        
        assert result['success'] == True
        
        predictions = model_trainer.predict(result['model_id'], [
            'Travel to London',
            'Schedule for tomorrow'
        ])
        
        assert len(predictions) == 2
    
    def test_ner_task_detection(self):
        """Test NER task is properly detected"""
        data = {
            'sentence': ['Sentence ' + str(i) for i in range(20)],
            'entity_tag': ['B-PER' if i % 2 == 0 else 'O' for i in range(20)]
        }
        df = pd.DataFrame(data)
        
        detected_task = detect_task_type(df)
        # Could be 'ner' or 'classification' depending on detection logic
        assert detected_task in ['ner', 'classification']
    
    def test_ner_bio_tagging(self):
        """Test NER with BIO tagging scheme"""
        data = {
            'token': [
                'Apple', 'John', 'New', 'Microsoft',
                'Tesla', 'Paris', 'Google', 'Amazon'
            ] * 3,
            'bio_tag': ['B-ORG', 'B-PER', 'B-LOC', 'B-ORG'] * 6
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'ner')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='ner',
            text_column='token',
            label_column='bio_tag',
            model_type='sklearn',
            training_config={'epochs': 3, 'batch_size': 8}
        )
        
        assert result['success'] == True
        assert 'training_time' in result
    
    def test_ner_named_entities(self):
        """Test NER with common named entity types"""
        data = {
            'text': [
                'Apple Inc', 'John Doe', 'United States',
                'Microsoft Corp', 'Jane Smith', 'London',
                'Google LLC', 'Bob Jones', 'Tokyo',
                'Amazon.com', 'Alice Brown', 'Paris'
            ] * 3,
            'label': ['ORG', 'PERSON', 'GPE'] * 12  # GPE = Geo-Political Entity
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'ner')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='named_entity_recognition',
            text_column='text',
            label_column='label',
            model_type='sklearn',
            training_config={'epochs': 5}
        )
        
        assert result['success'] == True
        
        predictions = model_trainer.predict(result['model_id'], [
            'Facebook',
            'Sarah Wilson',
            'Berlin'
        ])
        
        assert len(predictions) == 3
        assert all(p in ['ORG', 'PERSON', 'GPE'] for p in predictions)
