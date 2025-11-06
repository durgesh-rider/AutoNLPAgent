"""
Comprehensive tests for Text Summarization task
"""
import pytest
import pandas as pd
from app.core.preprocessor import TextPreprocessor
from app.core.trainer import model_trainer
from app.core.task_detector import detect_task_type


class TestTextSummarization:
    """Test suite for text summarization tasks"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.preprocessor = TextPreprocessor()
    
    def test_extractive_summarization(self):
        """Test extractive summarization"""
        data = {
            'document': [
                'The quick brown fox jumps over the lazy dog. This is a common pangram. It contains every letter of the alphabet.',
                'Machine learning is a subset of artificial intelligence. It enables computers to learn from data. Applications include image recognition.',
                'Climate change is a global issue. It affects weather patterns worldwide. Immediate action is needed to address it.',
                'Python is a versatile programming language. It is used in web development. Data science also relies heavily on Python.',
                'The human brain is complex. It contains billions of neurons. These neurons communicate through synapses.'
            ] * 6,
            'summary': [
                'The quick brown fox jumps over the lazy dog.',
                'Machine learning is a subset of artificial intelligence.',
                'Climate change is a global issue.',
                'Python is a versatile programming language.',
                'The human brain is complex.'
            ] * 6
        }
        df = pd.DataFrame(data)
        
        # Preprocess
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'summarization')
        
        # Train
        result = model_trainer.train_model(
            df=processed_df,
            task_type='summarization',
            text_column='document',
            label_column='summary',
            model_type='extractive',
            training_config={'epochs': 3, 'use_validation': True}
        )
        
        assert result['success'] == True
        assert result['task_type'] == 'summarization'
        assert 'metrics' in result
        assert 'compression_ratio' in result['metrics']
        
        # Predict
        long_text = 'This is a long document. It has multiple sentences. Each sentence provides information. The goal is to create a shorter version. Summarization helps with this task.'
        predictions = model_trainer.predict(result['model_id'], [long_text])
        
        assert len(predictions) == 1
        assert isinstance(predictions[0], str)
        assert len(predictions[0]) < len(long_text)
    
    def test_news_summarization(self):
        """Test summarization of news articles"""
        data = {
            'article': [
                'Breaking news today. A major event occurred. Authorities are investigating. More details will follow soon. Stay tuned for updates.',
                'In sports news. The championship game was played. Team A won against Team B. The final score was close. Fans celebrated the victory.',
                'Weather forecast shows rain. Temperatures will drop tomorrow. Wind speeds increase overnight. Prepare for stormy conditions ahead.',
                'Stock market update today. Prices fluctuated significantly. Tech stocks led gains. Investors remain cautious. Trading volume was high.'
            ] * 7,
            'headline': [
                'Breaking news: Major event under investigation',
                'Sports: Team A wins championship',
                'Weather: Rain and storms expected',
                'Markets: Tech stocks lead gains'
            ] * 7
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'summarization')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='text_summarization',
            text_column='article',
            label_column='headline',
            model_type='extractive',
            training_config={'epochs': 2}
        )
        
        assert result['success'] == True
        assert 'avg_doc_length' in result['metrics']
        assert 'avg_summary_length' in result['metrics']
        
        # Test summarization
        test_article = 'New technology emerged. It promises innovation. Companies are investing heavily. The future looks promising. Experts are optimistic.'
        predictions = model_trainer.predict(result['model_id'], [test_article])
        
        assert len(predictions) == 1
        assert len(predictions[0]) > 0
    
    def test_summarization_task_detection(self):
        """Test summarization task is properly detected"""
        data = {
            'full_text': ['Long text ' + str(i) * 10 for i in range(20)],
            'summary': ['Short ' + str(i) for i in range(20)]
        }
        df = pd.DataFrame(data)
        
        detected_task = detect_task_type(df)
        assert detected_task == 'summarization'
    
    def test_multi_sentence_summarization(self):
        """Test summarization with multiple sentences"""
        data = {
            'text': [
                'Sentence one provides context. Sentence two adds detail. Sentence three concludes the thought. Sentence four adds more. Sentence five wraps up.',
                'First point is important. Second point builds on it. Third point provides evidence. Fourth point summarizes. Fifth point concludes.',
                'Introduction sets the stage. Development explains the concept. Evidence supports the claim. Analysis interprets results. Conclusion ties together.'
            ] * 10,
            'summary': [
                'Sentence one provides context. Sentence two adds detail.',
                'First point is important. Second point builds on it.',
                'Introduction sets the stage. Development explains the concept.'
            ] * 10
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'summarization')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='summarization',
            text_column='text',
            label_column='summary',
            model_type='extractive',
            training_config={'epochs': 3, 'batch_size': 8}
        )
        
        assert result['success'] == True
        assert result['metrics']['compression_ratio'] > 0
        assert result['metrics']['compression_ratio'] < 1
    
    def test_summarization_single_sentence(self):
        """Test summarization handles single sentence"""
        data = {
            'text': ['This is a single sentence document.'] * 20 + ['A. B. C. D. E.'] * 10,
            'summary': ['This is a single sentence document.'] * 20 + ['A. B.'] * 10
        }
        df = pd.DataFrame(data)
        
        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'summarization')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='summarization',
            text_column='text',
            label_column='summary',
            model_type='extractive'
        )
        
        assert result['success'] == True
        
        # Single sentence should return itself
        predictions = model_trainer.predict(result['model_id'], ['One sentence only.'])
        assert len(predictions) == 1
        assert predictions[0] == 'One sentence only.'
    
    def test_summarization_compression_ratio(self):
        """Test summarization maintains reasonable compression ratio"""
        # Create docs with varying lengths
        data = {
            'document': [
                'The company reported strong earnings this quarter. Sales increased by 25 percent. ' * 20,
                'Scientists discovered a new species in the Amazon. The creature has unique features. ' * 15,
                'The government announced new policies today. Citizens reacted positively to the changes. ' * 10,
                'Technology advances continue to accelerate. New innovations emerge every day. ' * 8
            ] * 7,
            'summary': [
                'The company reported strong earnings this quarter. Sales increased by 25 percent. ' * 6,
                'Scientists discovered a new species in the Amazon. The creature has unique features. ' * 4,
                'The government announced new policies today. Citizens reacted positively to the changes. ' * 3,
                'Technology advances continue to accelerate. New innovations emerge every day. ' * 2
            ] * 7
        }
        df = pd.DataFrame(data)

        processed_df, metadata = self.preprocessor.preprocess_dataset(df, 'summarization')
        
        result = model_trainer.train_model(
            df=processed_df,
            task_type='summarization',
            text_column='document',
            label_column='summary',
            model_type='extractive',
            training_config={'epochs': 2}
        )
        
        assert result['success'] == True# Check compression ratio is reasonable
        compression = result['metrics']['compression_ratio']
        assert 0.1 < compression < 0.9
        
        # Test prediction respects compression
        long_doc = 'Word. ' * 50
        predictions = model_trainer.predict(result['model_id'], [long_doc])
        
        assert len(predictions[0]) < len(long_doc)
