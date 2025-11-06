import pandas as pd
from typing import Dict, Any, Tuple
import re


def detect_task_type(df: pd.DataFrame) -> str:
    """
    Automatically detect the NLP task type from the dataset structure and content.

    Returns:
        str: Detected task type ('classification', 'sentiment_analysis', 'ner',
             'summarization', 'question_answering', 'regression', 'unknown')
    """
    columns = df.columns.tolist()
    num_rows = len(df)

    # Check for common NLP task patterns

    # 1. Sentiment Analysis
    sentiment_keywords = ['sentiment', 'polarity', 'emotion', 'feeling', 'mood']
    if any(keyword in ' '.join(columns).lower() for keyword in sentiment_keywords):
        return 'sentiment_analysis'

    # Check column patterns for different tasks
    text_columns = []
    label_columns = []

    for col in columns:
        col_lower = col.lower()
        # Identify text columns
        if any(keyword in col_lower for keyword in ['text', 'content', 'message', 'review', 'comment', 'description']):
            text_columns.append(col)
        # Identify label/target columns
        elif any(keyword in col_lower for keyword in ['label', 'target', 'class', 'category', 'type']):
            label_columns.append(col)

    # If we have text and label columns, likely classification
    if text_columns and label_columns:
        # Check if labels are categorical (classification) or numeric (regression)
        label_col = label_columns[0]
        unique_labels = df[label_col].nunique()

        # If few unique labels or string labels, it's classification
        if unique_labels <= 20 or df[label_col].dtype == 'object':
            return 'classification'
        else:
            return 'regression'

    # 2. Named Entity Recognition (NER)
    ner_keywords = ['entity', 'ner', 'tag', 'bio', 'iob']
    if any(keyword in ' '.join(columns).lower() for keyword in ner_keywords):
        return 'ner'

    # 3. Question Answering
    qa_keywords = ['question', 'answer', 'context', 'qa']
    if any(keyword in ' '.join(columns).lower() for keyword in qa_keywords):
        return 'question_answering'

    # 4. Text Summarization
    summary_keywords = ['summary', 'abstract', 'summarize']
    if any(keyword in ' '.join(columns).lower() for keyword in summary_keywords):
        return 'summarization'

    # Analyze content patterns
    if len(columns) >= 2:
        # Sample some rows to analyze content
        sample_size = min(100, num_rows)
        sample_df = df.head(sample_size)

        # Check for classification patterns
        for col in columns:
            if df[col].dtype == 'object':
                unique_values = sample_df[col].unique()
                # If column has few unique values and looks like labels
                if len(unique_values) <= 10 and len(unique_values) > 1:
                    # Check if values look like class labels
                    if all(isinstance(val, str) and len(val.strip()) < 50 for val in unique_values):
                        return 'classification'

        # Check for sentiment patterns in text content
        text_cols = [col for col in columns if df[col].dtype == 'object']
        if text_cols:
            sample_texts = []
            for col in text_cols[:2]:  # Check first 2 text columns
                col_texts = sample_df[col].dropna().astype(str).tolist()[:10]
                sample_texts.extend(col_texts)

            # Look for sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor']

            has_sentiment = False
            for text in sample_texts:
                text_lower = text.lower()
                if any(word in text_lower for word in positive_words + negative_words):
                    has_sentiment = True
                    break

            if has_sentiment:
                return 'sentiment_analysis'

    # Default to classification if we have text and some other column
    if len(columns) >= 2 and any(df[col].dtype == 'object' for col in columns):
        return 'classification'

    return 'unknown'


def analyze_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset structure and provide insights"""
    analysis = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': {},
        'data_types': {},
        'missing_values': {},
        'unique_values': {}
    }

    for col in df.columns:
        analysis['columns'][col] = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
        }

        if df[col].dtype == 'object':
            unique_vals = df[col].nunique()
            analysis['unique_values'][col] = unique_vals
            # Sample unique values for categorical columns
            if unique_vals <= 20:
                analysis['columns'][col]['unique_samples'] = df[col].dropna().unique().tolist()

    return analysis