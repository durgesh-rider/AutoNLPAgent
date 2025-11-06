"""
Comprehensive tests for Question Answering (QA) task
"""
import pytest
import pandas as pd
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import model_trainer
from app.core.task_detector import detect_task_type


class TestQuestionAnswering:
    """Test suite for QA tasks"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.preprocessor = TextPreprocessor()
    
    def test_simple_qa(self):
        """Test simple question-answering"""
        data = {
            'question': [
                'What is the capital of France?',
                'Who invented the telephone?',
                'What is the largest ocean?',
                'Who wrote Romeo and Juliet?',
                'What is the speed of light?',
                'Who painted the Mona Lisa?'
            ] * 5,
            'answer': [
                'Paris',
                'Alexander Graham Bell',
                'Pacific Ocean',
                'William Shakespeare',
                '299,792,458 meters per second',
                'Leonardo da Vinci'
            ] * 5
        }
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'question_answering')
        
        # Train
        result = model_trainer.train_model(
            df=processed_df,
            task_type='question_answering',
            text_column='question',
            label_column='answer',
            model_type='sklearn',
            training_config={'epochs': 3, 'use_validation': True}
        )
        
        assert result['success'] == True
        assert result['task_type'] == 'qa'
        assert 'metrics' in result
        
        # Predict
        predictions = model_trainer.predict(result['model_id'], [
            'What is the capital of France?',
            'Who invented the telephone?'
        ])
        
        assert len(predictions) == 2
        assert all(isinstance(p, str) for p in predictions)
    
    def test_qa_with_context(self):
        """Test QA with context passages"""
        data = {
            'context': [
                'Python is a high-level programming language.',
                'The Earth orbits around the Sun.',
                'Water boils at 100 degrees Celsius.',
                'Shakespeare was born in 1564.',
                'Mount Everest is the tallest mountain.',
                'DNA stands for Deoxyribonucleic Acid.'
            ] * 4,
            'answer': [
                'Python',
                'Sun',
                '100 degrees Celsius',
                '1564',
                'Mount Everest',
                'Deoxyribonucleic Acid'
            ] * 4
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'qa')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='qa',
            text_column='context',
            label_column='answer',
            model_type='sklearn',
            training_config={'epochs': 5}
        )
        
        assert result['success'] == True
        
        predictions = model_trainer.predict(result['model_id'], [
            'Python is a programming language',
            'Earth orbits the Sun'
        ])
        
        assert len(predictions) == 2
    
    def test_qa_task_detection(self):
        """Test QA task is properly detected"""
        data = {
            'question': ['What is ' + str(i) + '?' for i in range(20)],
            'answer': ['Answer ' + str(i) for i in range(20)]
        }
        df = pd.DataFrame(data)
        
        detected_task = detect_task_type(df)
        assert detected_task == 'question_answering'
    
    def test_qa_factoid_questions(self):
        """Test QA with factoid questions"""
        data = {
            'question': [
                'What color is the sky?',
                'How many days in a week?',
                'What is 2+2?',
                'What is ice made of?',
                'Where is the Eiffel Tower?',
                'When did WWII end?'
            ] * 4,
            'answer': [
                'blue',
                '7',
                '4',
                'water',
                'Paris',
                '1945'
            ] * 4
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'question_answering')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='question_answering',
            text_column='question',
            label_column='answer',
            model_type='sklearn',
            training_config={'epochs': 3, 'batch_size': 8}
        )
        
        assert result['success'] == True
        assert 'qa_pairs' in result
    
    def test_qa_empty_questions(self):
        """Test QA handles empty questions gracefully"""
        data = {
            'question': ['Question ' + str(i) for i in range(30)],
            'answer': ['Answer ' + str(i) for i in range(30)]
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'qa')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='qa',
            text_column='question',
            label_column='answer',
            model_type='sklearn'
        )
        
        # Test with empty questions
        predictions = model_trainer.predict(result['model_id'], ['', '  ', 'Normal question?'])
        
        assert len(predictions) == 3
        assert all(isinstance(p, str) for p in predictions)
    
    def test_qa_multiple_choice(self):
        """Test QA with limited answer options"""
        data = {
            'question': [
                'Is Python a programming language?',
                'Is the Earth flat?',
                'Is water wet?',
                'Is AI useful?'
            ] * 6,
            'answer': ['yes', 'no', 'yes', 'yes'] * 6
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'qa')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='question_answering',
            text_column='question',
            label_column='answer',
            model_type='sklearn',
            training_config={'epochs': 5}
        )
        
        assert result['success'] == True
        
        predictions = model_trainer.predict(result['model_id'], [
            'Is Java a programming language?',
            'Is the moon made of cheese?'
        ])
        
        assert len(predictions) == 2
