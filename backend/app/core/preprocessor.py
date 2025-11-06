import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoders = {}

    def preprocess_dataset(self, df: pd.DataFrame, task_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess the dataset based on the detected task type.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Processed dataframe and preprocessing metadata
        """
        preprocessing_steps = []
        metadata = {
            'original_shape': df.shape,
            'task_type': task_type,
            'preprocessing_steps': preprocessing_steps
        }

        # Handle missing values
        df, missing_handled = self._handle_missing_values(df)
        if missing_handled > 0:
            preprocessing_steps.append(f"Handled {missing_handled} missing values")

        # Identify text and label columns
        text_columns, label_column = self._identify_columns(df, task_type)

        # Preprocess text columns
        for col in text_columns:
            df[col] = df[col].apply(self._preprocess_text)
            preprocessing_steps.append(f"Preprocessed text column: {col}")

        # Encode labels for classification tasks
        if task_type in ['classification', 'sentiment_analysis'] and label_column:
            df, label_encoder = self._encode_labels(df, label_column)
            metadata['label_encoder'] = label_encoder
            preprocessing_steps.append(f"Encoded labels in column: {label_column}")

        # Feature engineering for text data
        if text_columns:
            df = self._extract_text_features(df, text_columns)
            preprocessing_steps.append("Extracted text features")

        metadata['processed_shape'] = df.shape
        metadata['text_columns'] = text_columns
        metadata['label_column'] = label_column

        return df, metadata

    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Handle missing values in the dataset"""
        missing_count = df.isnull().sum().sum()

        if missing_count > 0:
            # For text columns, fill with empty string
            text_cols = df.select_dtypes(include=['object']).columns
            df[text_cols] = df[text_cols].fillna('')

            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())

        return df, missing_count

    def _identify_columns(self, df: pd.DataFrame, task_type: str) -> Tuple[List[str], str]:
        """Identify text and label columns based on task type"""
        columns = df.columns.tolist()

        # Common text column names
        text_keywords = ['text', 'content', 'message', 'review', 'comment', 'description', 'sentence', 'document']
        text_columns = [col for col in columns if any(keyword in col.lower() for keyword in text_keywords)]

        # If no text columns found by keywords, assume first object column is text
        if not text_columns:
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                text_columns = [object_cols[0]]

        # Identify label column
        label_keywords = ['label', 'target', 'class', 'category', 'sentiment', 'type']
        label_column = None

        for col in columns:
            if col not in text_columns and any(keyword in col.lower() for keyword in label_keywords):
                label_column = col
                break

        # If no label column found and we have classification task, assume last non-text column
        if not label_column and task_type in ['classification', 'sentiment_analysis']:
            remaining_cols = [col for col in columns if col not in text_columns]
            if remaining_cols:
                label_column = remaining_cols[-1]

        return text_columns, label_column

    def _preprocess_text(self, text: str) -> str:
        """Preprocess individual text"""
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Join back to string
        return ' '.join(tokens)

    def _encode_labels(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, LabelEncoder]:
        """Encode categorical labels"""
        label_encoder = LabelEncoder()
        df[label_column] = label_encoder.fit_transform(df[label_column].astype(str))
        self.label_encoders[label_column] = label_encoder
        return df, label_encoder

    def _extract_text_features(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Extract additional features from text columns"""
        for col in text_columns:
            # Add text length features
            df[f'{col}_length'] = df[col].apply(len)
            df[f'{col}_word_count'] = df[col].apply(lambda x: len(x.split()))

        return df

    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the preprocessing"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'text_columns_count': len([col for col in df.columns if col.endswith('_length')]),
            'missing_values_remaining': df.isnull().sum().sum(),
            'label_distributions': {}
        }