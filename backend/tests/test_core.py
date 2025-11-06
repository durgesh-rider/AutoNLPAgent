"""
Comprehensive tests for core functionality
"""
import pytest
import pandas as pd
import numpy as np
from app.core.task_detector import detect_task_type, analyze_dataset_structure
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import ModelTrainer
from app.core.evaluator import ModelEvaluator


class TestTaskDetector:
    """Test task detection functionality"""
    
    def test_sentiment_detection(self):
        """Test sentiment analysis task detection"""
        df = pd.DataFrame({
            'text': ['I love this!', 'This is terrible', 'Good product'],
            'sentiment': ['positive', 'negative', 'positive']
        })
        task = detect_task_type(df)
        assert task == 'sentiment_analysis'
    
    def test_classification_detection(self):
        """Test classification task detection"""
        df = pd.DataFrame({
            'text': ['News article 1', 'Sports update', 'Tech review'],
            'category': ['news', 'sports', 'tech']
        })
        task = detect_task_type(df)
        assert task == 'classification'
    
    def test_qa_detection(self):
        """Test question answering task detection"""
        df = pd.DataFrame({
            'question': ['What is AI?', 'Where is Paris?'],
            'answer': ['Artificial Intelligence', 'France'],
            'context': ['AI is...', 'Paris is the capital...']
        })
        task = detect_task_type(df)
        assert task == 'question_answering'
    
    def test_ner_detection(self):
        """Test NER task detection"""
        df = pd.DataFrame({
            'text': ['John lives in Paris', 'Apple Inc. released new iPhone'],
            'entity_tags': ['B-PER O O B-LOC', 'B-ORG O O O B-PROD']
        })
        task = detect_task_type(df)
        assert task == 'ner'
    
    def test_summarization_detection(self):
        """Test summarization task detection"""
        df = pd.DataFrame({
            'text': ['Long article text here...', 'Another article...'],
            'summary': ['Short summary', 'Brief summary']
        })
        task = detect_task_type(df)
        assert task == 'summarization'
    
    def test_dataset_analysis(self):
        """Test dataset structure analysis"""
        df = pd.DataFrame({
            'text': ['Sample text'] * 10,
            'label': ['A', 'B'] * 5
        })
        analysis = analyze_dataset_structure(df)
        
        assert analysis['num_rows'] == 10
        assert analysis['num_columns'] == 2
        assert 'text' in analysis['columns']
        assert 'label' in analysis['columns']
    
    def test_unknown_task(self):
        """Test unknown task detection"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        task = detect_task_type(df)
        assert task == 'unknown'


class TestPreprocessor:
    """Test data preprocessing"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return TextPreprocessor()
    
    def test_text_cleaning(self, preprocessor):
        """Test text preprocessing"""
        df = pd.DataFrame({
            'text': ['Hello World!', 'This is GREAT!!!', 'Bad product...'],
            'label': ['positive', 'positive', 'negative']
        })
        
        result, metadata = preprocessor.preprocess_dataset(df, task_type='classification')
        
        assert result is not None
        assert len(result) == 3
        assert metadata['task_type'] == 'classification'
        assert 'preprocessing_steps' in metadata
    
    def test_missing_value_handling(self, preprocessor):
        """Test handling of missing values"""
        df = pd.DataFrame({
            'text': ['Hello', None, 'World', ''],
            'label': ['A', 'B', None, 'A']
        })
        
        result, metadata = preprocessor.preprocess_dataset(df, 'classification')
        
        # Should handle missing values appropriately
        assert result is not None
        assert metadata['original_shape'][0] == 4
    
    def test_label_encoding(self, preprocessor):
        """Test label encoding"""
        df = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': ['A', 'B', 'A']
        })
        
        result, metadata = preprocessor.preprocess_dataset(df, 'classification')
        
        # Check if label column was encoded
        assert 'label_column' in metadata
        if metadata['label_column']:
            assert result[metadata['label_column']].dtype in [np.int32, np.int64, 'int64', 'int32']
    
    def test_text_feature_extraction(self, preprocessor):
        """Test text feature extraction"""
        df = pd.DataFrame({
            'review': ['This is great!', 'Not good'],
            'category': ['pos', 'neg']
        })
        
        result, metadata = preprocessor.preprocess_dataset(df, 'sentiment_analysis')
        
        assert result is not None
        assert 'text_columns' in metadata
        assert len(metadata['text_columns']) > 0
    
    def test_column_identification(self, preprocessor):
        """Test automatic column identification"""
        df = pd.DataFrame({
            'review_text': ['Great product', 'Poor quality'],
            'rating': ['good', 'bad']
        })
        
        result, metadata = preprocessor.preprocess_dataset(df, 'classification')
        
        assert 'text_columns' in metadata
        assert 'label_column' in metadata


class TestModelTrainer:
    """Test model training"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        texts = [f"Sample text number {i} for testing" for i in range(100)]
        labels = np.random.choice(['positive', 'negative'], 100)
        
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance"""
        return ModelTrainer()
    
    def test_sklearn_training(self, trainer, sample_data):
        """Test sklearn model training"""
        result = trainer.train_model(
            df=sample_data,
            task_type='classification',
            text_column='text',
            label_column='label',
            model_type='sklearn'
        )
        
        assert result['success'] is True
        assert 'model_id' in result
        assert 'training_time' in result
        assert result['model_id'] is not None
    
    def test_model_storage(self, trainer, sample_data):
        """Test model storage and retrieval"""
        result = trainer.train_model(
            df=sample_data,
            task_type='classification',
            text_column='text',
            label_column='label'
        )
        
        model_id = result['model_id']
        assert model_id in trainer.models
        
        model_info = trainer.models[model_id]
        assert model_info is not None
        assert model_info['model_id'] == model_id
    
    def test_classification_model(self, trainer, sample_data):
        """Test classification model training"""
        result = trainer.train_model(
            df=sample_data,
            task_type='classification',
            text_column='text',
            label_column='label'
        )
        
        assert result['success'] is True
        assert result['task_type'] == 'classification'
        assert 'training_samples' in result
        assert 'test_samples' in result
    
    def test_sentiment_model(self, trainer):
        """Test sentiment analysis model"""
        df = pd.DataFrame({
            'text': ['I love it!', 'Terrible', 'Great!', 'Bad'] * 25,
            'sentiment': ['positive', 'negative', 'positive', 'negative'] * 25
        })
        
        result = trainer.train_model(
            df=df,
            task_type='sentiment_analysis',
            text_column='text',
            label_column='sentiment'
        )
        
        assert result['success'] is True


class TestModelEvaluator:
    """Test model evaluation"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return ModelEvaluator()
    
    @pytest.fixture
    def mock_model_info(self):
        """Create mock model info with predictions"""
        return {
            'model_id': 'test_model',
            'predictions': np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            'true_labels': np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0])
        }
    
    def test_metrics_calculation(self, evaluator, mock_model_info):
        """Test evaluation metrics calculation"""
        result = evaluator.evaluate_model(mock_model_info)
        
        assert result['success'] is True
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 'precision_macro' in result['metrics']
        assert 'recall_macro' in result['metrics']
        assert 'f1_macro' in result['metrics']
    
    def test_confusion_matrix(self, evaluator, mock_model_info):
        """Test confusion matrix generation"""
        result = evaluator.evaluate_model(mock_model_info)
        
        assert 'confusion_matrix' in result
        assert isinstance(result['confusion_matrix'], list)
        assert len(result['confusion_matrix']) > 0
    
    def test_classification_report(self, evaluator, mock_model_info):
        """Test classification report generation"""
        result = evaluator.evaluate_model(mock_model_info)
        
        assert 'classification_report' in result
        assert isinstance(result['classification_report'], dict)
    
    def test_visualization_generation(self, evaluator, mock_model_info):
        """Test visualization creation"""
        result = evaluator.evaluate_model(mock_model_info)
        
        assert result['success'] is True
        assert 'charts' in result
    
    def test_evaluation_storage(self, evaluator, mock_model_info):
        """Test evaluation result storage"""
        result = evaluator.evaluate_model(mock_model_info)
        
        evaluation_id = result['evaluation_id']
        assert evaluation_id in evaluator.evaluation_results
        
        stored_result = evaluator.evaluation_results[evaluation_id]
        assert stored_result['model_id'] == mock_model_info['model_id']


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_pipeline(self):
        """Test complete pipeline from detection to evaluation"""
        # Create sample dataset
        df = pd.DataFrame({
            'text': [
                'Great product!', 'Terrible quality', 'Love it!', 'Awful',
                'Excellent', 'Poor', 'Amazing', 'Bad', 'Wonderful', 'Worst'
            ] * 10,
            'sentiment': [
                'positive', 'negative', 'positive', 'negative',
                'positive', 'negative', 'positive', 'negative',
                'positive', 'negative'
            ] * 10
        })
        
        # Step 1: Task Detection
        task = detect_task_type(df)
        assert task in ['sentiment_analysis', 'classification']
        
        # Step 2: Preprocessing
        preprocessor = TextPreprocessor()
        processed_df, metadata = preprocessor.preprocess_dataset(df, task)
        assert processed_df is not None
        assert len(processed_df) > 0
        
        # Step 3: Model Training
        trainer = ModelTrainer()
        result = trainer.train_model(
            df=processed_df,
            task_type=task,
            text_column=metadata['text_columns'][0] if metadata['text_columns'] else 'text',
            label_column=metadata['label_column'] if metadata['label_column'] else 'sentiment',
            model_type='sklearn'
        )
        assert result['success'] is True
        
        # Step 4: Evaluation
        evaluator = ModelEvaluator()
        eval_result = evaluator.evaluate_model(result)
        assert eval_result['success'] is True
        assert 'metrics' in eval_result
    
    def test_different_data_sizes(self):
        """Test with different dataset sizes"""
        sizes = [10, 50, 100]
        
        for size in sizes:
            df = pd.DataFrame({
                'text': [f'Sample text {i}' for i in range(size)],
                'label': np.random.choice(['A', 'B'], size)
            })
            
            task = detect_task_type(df)
            assert task in ['classification', 'sentiment_analysis']
            
            preprocessor = TextPreprocessor()
            processed_df, _ = preprocessor.preprocess_dataset(df, task)
            assert len(processed_df) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
