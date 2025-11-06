import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from datetime import datetime
import os

# Optional imports for transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None


class ModelTrainer:
    def __init__(self):
        self.models = {}
        if TRANSFORMERS_AVAILABLE and torch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

    def train_model(self, df: pd.DataFrame, task_type: str, text_column: str,
                   label_column: str, model_type: str = None, training_config: dict = None, 
                   label_encoder=None) -> Dict[str, Any]:
        """
        Train a model for the specified NLP task.

        Args:
            df: Preprocessed dataframe
            task_type: Type of NLP task
            text_column: Name of text column
            label_column: Name of label column
            model_type: Preferred model type (optional)
            training_config: Training configuration (epochs, batch_size, etc.)
            label_encoder: LabelEncoder for decoding predictions (optional)

        Returns:
            Dict containing training results and model info
        """
        # Default config
        if training_config is None:
            training_config = {
                'epochs': 3,
                'batch_size': 16,
                'use_validation': True
            }
        
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            if task_type in ['classification', 'sentiment_analysis']:
                return self._train_classification_model(
                    df, model_id, text_column, label_column, model_type, training_config, label_encoder
                )
            elif task_type in ['ner', 'named_entity_recognition']:
                return self._train_ner_model(
                    df, model_id, text_column, label_column, model_type, training_config
                )
            elif task_type in ['qa', 'question_answering']:
                return self._train_qa_model(
                    df, model_id, text_column, label_column, model_type, training_config
                )
            elif task_type in ['summarization', 'text_summarization']:
                return self._train_summarization_model(
                    df, model_id, text_column, label_column, model_type, training_config
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}. Supported tasks: classification, sentiment_analysis, ner, question_answering, summarization")

        except Exception as e:
            return {
                'model_id': model_id,
                'success': False,
                'error': str(e),
                'training_time': 0
            }

    def _train_classification_model(self, df: pd.DataFrame, model_id: str, text_column: str,
                                  label_column: str, model_type: str,
                                  training_config: dict, label_encoder=None) -> Dict[str, Any]:
        """Train a classification model"""
        start_time = datetime.now()

        # Prepare data and remove NaN values
        X = df[text_column].fillna('')
        y = df[label_column]
        
        # Remove rows where label is NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) == 0:
            return {
                'success': False,
                'error': 'No valid labels found in the dataset. All label values are missing.'
            }
        
        # Check minimum dataset size
        min_samples = 10
        if len(y) < min_samples:
            return {
                'success': False,
                'error': f'Dataset too small. Need at least {min_samples} samples, but only have {len(y)}.'
            }

        # Use validation split if configured
        test_size = 0.2 if training_config.get('use_validation', True) else 0.1
        
        # Check if we can stratify (need at least 2 samples per class)
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        can_stratify = min_class_count >= 2 and len(y) * test_size >= len(class_counts)
        
        if not can_stratify and min_class_count == 1:
            return {
                'success': False,
                'error': f'Some classes have only 1 sample. Each class needs at least 2 samples for training. Class distribution: {class_counts.to_dict()}'
            }
        
        # Adjust test_size if dataset is very small
        if len(y) < 20:
            test_size = 0.1  # Use smaller test set for very small datasets
        
        # Split data
        try:
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            else:
                # Don't stratify if we can't maintain class distribution
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
        except ValueError as e:
            return {
                'success': False,
                'error': f'Cannot split dataset: {str(e)}. Try using a larger dataset or ensure all classes have multiple samples.'
            }

        model_info = {
            'model_id': model_id,
            'task_type': 'classification',
            'model_type': model_type or 'auto',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'config': training_config,
            'label_encoder': label_encoder,  # Store label encoder for decoding predictions
            'text_column': text_column,  # Store for user reference
            'label_column': label_column  # Store for user reference
        }

        # Choose model based on data size and preference
        if model_type == 'transformer' or (model_type is None and len(X_train) > 1000):
            # Use transformer model for larger datasets or if explicitly requested
            result = self._train_transformer_model(
                X_train, y_train, X_test, y_test, model_info, training_config
            )
        else:
            # Use traditional ML model for smaller datasets
            result = self._train_sklearn_model(
                X_train, y_train, X_test, y_test, model_info
            )

        result['training_time'] = (datetime.now() - start_time).total_seconds()
        result['success'] = True

        # Store model
        self.models[model_id] = result

        return result

    def _train_sklearn_model(self, X_train, y_train, X_test, y_test, model_info: Dict) -> Dict[str, Any]:
        """Train sklearn model"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Vectorize text
        try:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        except ValueError as e:
            # Handle empty vocabulary or insufficient text
            if 'empty vocabulary' in str(e).lower():
                return {
                    **model_info,
                    'success': False,
                    'error': 'Text data is too simple or repetitive. Please provide more diverse text content.'
                }
            else:
                return {
                    **model_info,
                    'success': False,
                    'error': f'Text vectorization failed: {str(e)}'
                }

        # Choose model
        if len(np.unique(y_train)) > 10:  # Multi-class
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:  # Binary or small multi-class
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train model
        model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = model.predict(X_test_vec)
        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            **model_info,
            'model': model,
            'vectorizer': vectorizer,
            'metrics': metrics,
            'model_type': 'sklearn'
        }

    def _train_transformer_model(self, X_train, y_train, X_test, y_test, model_info: Dict, 
                                training_config: dict) -> Dict[str, Any]:
        """Train transformer model using Hugging Face"""
        try:
            # Load pre-trained model
            model_name = "distilbert-base-uncased"  # Smaller, faster model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(np.unique(y_train))
            )

            # Prepare datasets
            train_dataset = self._create_dataset(X_train.tolist(), y_train.tolist(), tokenizer)
            test_dataset = self._create_dataset(X_test.tolist(), y_test.tolist(), tokenizer)

            # Training arguments - use config values
            epochs = training_config.get('epochs', 3)
            batch_size = training_config.get('batch_size', 16)
            
            training_args = TrainingArguments(
                output_dir=f"./results/{model_info['model_id']}",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"./logs/{model_info['model_id']}",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Train
            trainer.train()

            # Evaluate
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            metrics = self._calculate_metrics(y_test, y_pred)

            return {
                **model_info,
                'model': model,
                'tokenizer': tokenizer,
                'trainer': trainer,
                'metrics': metrics,
                'model_type': 'transformer'
            }

        except Exception as e:
            # Fallback to sklearn if transformer training fails
            print(f"Transformer training failed: {e}, falling back to sklearn")
            return self._train_sklearn_model(X_train, y_train, X_test, y_test, model_info)

    def _create_dataset(self, texts: list, labels: list, tokenizer):
        """Create dataset for transformer training"""
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )

                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        return TextDataset(texts, labels, tokenizer)

    def _compute_metrics(self, eval_pred):
        """Compute metrics for transformer training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'f1': f1
        }

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _train_ner_model(self, df: pd.DataFrame, model_id: str, text_column: str, label_column: str,
                        model_type: str, training_config: dict) -> Dict[str, Any]:
        """Train a Named Entity Recognition model"""
        start_time = datetime.now()

        # For NER, we expect text and entity labels (e.g., BIO format)
        # Simplified implementation using classification on tokens
        X = df[text_column].fillna('')
        y = df[label_column]
        
        # Remove rows where label is NaN
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) == 0:
            return {
                'success': False,
                'error': 'No valid entity labels found in the dataset. All label values are missing.'
            }
        
        # Check minimum dataset size
        min_samples = 10
        if len(y) < min_samples:
            return {
                'success': False,
                'error': f'Dataset too small for NER. Need at least {min_samples} samples, but only have {len(y)}.'
            }

        # Split data
        test_size = training_config.get('test_size', 0.2)
        
        # Adjust test_size for small datasets
        if len(y) < 20:
            test_size = 0.1
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        except ValueError as e:
            return {
                'success': False,
                'error': f'Cannot split dataset: {str(e)}. Try using a larger dataset.'
            }

        # Use sklearn for simplified NER (token classification)
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), analyzer='char_wb', min_df=1)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        except ValueError as e:
            return {
                'model_id': model_id,
                'task_type': 'ner',
                'success': False,
                'error': f'Text vectorization failed: {str(e)}. Ensure text data contains sufficient diversity.',
                'training_time': (datetime.now() - start_time).total_seconds()
            }

        # Use RandomForest for NER
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = model.predict(X_test_vec)
        metrics = self._calculate_metrics(y_test, y_pred)

        model_info = {
            'model_id': model_id,
            'task_type': 'ner',
            'model_type': 'sklearn',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model': model,
            'vectorizer': vectorizer,
            'metrics': metrics,
            'config': training_config,
            'text_column': text_column,  # Store for user reference
            'label_column': label_column  # Store for user reference
        }

        result = {
            **model_info,
            'success': True,
            'training_time': (datetime.now() - start_time).total_seconds()
        }

        self.models[model_id] = result
        return result

    def _train_qa_model(self, df: pd.DataFrame, model_id: str, text_column: str,
                       label_column: str, model_type: str,
                       training_config: dict) -> Dict[str, Any]:
        """Train a Question Answering model"""
        start_time = datetime.now()

        # For QA, we expect question and answer columns
        # Simplified implementation: treat as text generation/retrieval
        X = df[text_column].fillna('')  # Questions or context
        y = df[label_column].fillna('')  # Answers
        
        # Remove rows where either question or answer is empty
        valid_mask = (X.str.strip() != '') & (y.str.strip() != '')
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            return {
                'success': False,
                'error': 'No valid question-answer pairs found in the dataset.'
            }
        
        # Check minimum dataset size
        min_samples = 10
        if len(X) < min_samples:
            return {
                'success': False,
                'error': f'Dataset too small for QA. Need at least {min_samples} question-answer pairs, but only have {len(X)}.'
            }

        # Split data
        test_size = training_config.get('test_size', 0.2)
        
        # Adjust test_size for small datasets
        if len(X) < 20:
            test_size = 0.1
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        except ValueError as e:
            return {
                'success': False,
                'error': f'Cannot split dataset: {str(e)}. Try using a larger dataset.'
            }

        # Use TF-IDF for question-answer matching
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=1)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
        except ValueError as e:
            return {
                'model_id': model_id,
                'task_type': 'qa',
                'success': False,
                'error': f'Question vectorization failed: {str(e)}. Ensure questions contain sufficient diversity.',
                'training_time': (datetime.now() - start_time).total_seconds()
            }

        # Store questions and answers for retrieval
        # For classification-based QA, encode answers
        from sklearn.preprocessing import LabelEncoder
        answer_encoder = LabelEncoder()
        
        # Try to fit, handle if answers are too diverse
        try:
            y_train_encoded = answer_encoder.fit_transform(y_train.astype(str))
            y_test_encoded = answer_encoder.transform(y_test.astype(str))
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_vec, y_train_encoded)
            
            y_pred = model.predict(X_test_vec)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test_encoded, y_pred)
        except:
            # If answers are too diverse, use a simpler approach
            model = None
            answer_encoder = None
            metrics = {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}

        model_info = {
            'model_id': model_id,
            'task_type': 'qa',
            'model_type': 'sklearn',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model': model,
            'vectorizer': vectorizer,
            'answer_encoder': answer_encoder,
            'metrics': metrics,
            'config': training_config,
            'qa_pairs': list(zip(X_train.tolist(), y_train.tolist()))[:100],  # Store sample Q&A pairs
            'text_column': text_column,  # Store for user reference (questions)
            'label_column': label_column  # Store for user reference (answers)
        }

        result = {
            **model_info,
            'success': True,
            'training_time': (datetime.now() - start_time).total_seconds()
        }

        self.models[model_id] = result
        return result

    def _train_summarization_model(self, df: pd.DataFrame, model_id: str, text_column: str,
                                   label_column: str, model_type: str,
                                   training_config: dict) -> Dict[str, Any]:
        """Train a Text Summarization model"""
        start_time = datetime.now()

        # For summarization, we expect full text and summary
        X = df[text_column].fillna('')  # Full documents
        y = df[label_column].fillna('')  # Summaries
        
        # Remove rows where either document or summary is empty
        valid_mask = (X.str.strip() != '') & (y.str.strip() != '')
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            return {
                'success': False,
                'error': 'No valid document-summary pairs found in the dataset.'
            }
        
        # Check minimum dataset size
        min_samples = 10
        if len(X) < min_samples:
            return {
                'success': False,
                'error': f'Dataset too small for summarization. Need at least {min_samples} document-summary pairs, but only have {len(X)}.'
            }

        # Split data
        test_size = training_config.get('test_size', 0.2)
        
        # Adjust test_size for small datasets
        if len(X) < 20:
            test_size = 0.1
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        except ValueError as e:
            return {
                'success': False,
                'error': f'Cannot split dataset: {str(e)}. Try using a larger dataset.'
            }

        # Simplified extractive summarization using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
            X_train_vec = vectorizer.fit_transform(X_train)
        except ValueError as e:
            return {
                'model_id': model_id,
                'task_type': 'summarization',
                'success': False,
                'error': f'Document vectorization failed: {str(e)}. Ensure documents contain sufficient diversity.',
                'training_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Store document-summary pairs for retrieval-based summarization
        # Calculate basic statistics
        avg_doc_length = X_train.str.len().mean()
        avg_summary_length = y_train.str.len().mean()
        compression_ratio = avg_summary_length / avg_doc_length if avg_doc_length > 0 else 0.3

        model_info = {
            'model_id': model_id,
            'task_type': 'summarization',
            'model_type': 'extractive',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'vectorizer': vectorizer,
            'metrics': {
                'avg_doc_length': float(avg_doc_length),
                'avg_summary_length': float(avg_summary_length),
                'compression_ratio': float(compression_ratio)
            },
            'config': training_config,
            'summary_pairs': list(zip(X_train.tolist()[:50], y_train.tolist()[:50])),  # Sample pairs
            'text_column': text_column,  # Store for user reference (documents)
            'label_column': label_column  # Store for user reference (summaries)
        }

        result = {
            **model_info,
            'model': None,  # Extractive summarization doesn't need a trained model
            'success': True,
            'training_time': (datetime.now() - start_time).total_seconds()
        }

        self.models[model_id] = result
        return result

    def get_model(self, model_id: str):
        """Get trained model by ID"""
        return self.models.get(model_id)

    def predict(self, model_id: str, texts: list, return_probabilities: bool = False) -> list:
        """Make predictions using trained model and decode to original labels"""
        model_info = self.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")

        task_type = model_info.get('task_type', 'classification')
        model = model_info.get('model')
        label_encoder = model_info.get('label_encoder')

        # Handle different task types
        if task_type == 'summarization':
            # Extractive summarization
            return self._predict_summarization(texts, model_info)
        elif task_type == 'qa':
            # Question answering
            return self._predict_qa(texts, model_info)
        elif task_type == 'ner':
            # Named entity recognition
            return self._predict_ner(texts, model_info)
        else:
            # Classification and sentiment analysis
            if model_info['model_type'] == 'sklearn':
                vectorizer = model_info['vectorizer']
                X_vec = vectorizer.transform(texts)
                predictions = model.predict(X_vec)
                
                # Get probabilities if requested and model supports it
                probabilities = None
                confidence_scores = None
                if return_probabilities and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_vec)
                    confidence_scores = probabilities.max(axis=1).tolist()
                    probabilities = probabilities.tolist()
            else:
                # Transformer model
                tokenizer = model_info['tokenizer']
                predictions = []
                probabilities = []
                confidence_scores = []

                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        
                        if return_probabilities:
                            probs = torch.softmax(logits, dim=1)[0]
                            probabilities.append(probs.tolist())
                            confidence_scores.append(probs.max().item())
                        
                        pred = torch.argmax(logits, dim=1).item()
                        predictions.append(pred)

            # Decode predictions back to original labels if label encoder is available
            if label_encoder is not None:
                try:
                    decoded_predictions = label_encoder.inverse_transform(predictions)
                    decoded_predictions = decoded_predictions.tolist()
                except Exception as e:
                    # If decoding fails, return original predictions with warning
                    print(f"Warning: Could not decode predictions: {str(e)}")
                    decoded_predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            else:
                decoded_predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            
            # Return with probabilities if requested
            if return_probabilities:
                result = {
                    'predictions': decoded_predictions,
                    'probabilities': probabilities if probabilities else [],
                    'confidence_scores': confidence_scores if confidence_scores else []
                }
                return result
            
            return decoded_predictions

    def _predict_ner(self, texts: list, model_info: Dict) -> list:
        """Predict named entities in texts"""
        model = model_info['model']
        vectorizer = model_info['vectorizer']
        
        predictions = []
        for text in texts:
            X_vec = vectorizer.transform([text])
            pred = model.predict(X_vec)[0]
            predictions.append(pred)
        
        return predictions

    def _predict_qa(self, questions: list, model_info: Dict) -> list:
        """Answer questions using QA model"""
        model = model_info.get('model')
        vectorizer = model_info['vectorizer']
        answer_encoder = model_info.get('answer_encoder')
        
        if model and answer_encoder:
            # Use trained model
            predictions = []
            for question in questions:
                X_vec = vectorizer.transform([question])
                pred_encoded = model.predict(X_vec)[0]
                answer = answer_encoder.inverse_transform([pred_encoded])[0]
                predictions.append(answer)
        else:
            # Fallback to retrieval from stored Q&A pairs
            qa_pairs = model_info.get('qa_pairs', [])
            predictions = []
            for question in questions:
                # Simple retrieval: return most similar answer
                if qa_pairs:
                    # Return first answer as fallback
                    predictions.append(qa_pairs[0][1] if len(qa_pairs[0]) > 1 else "No answer available")
                else:
                    predictions.append("No answer available")
        
        return predictions

    def _predict_summarization(self, texts: list, model_info: Dict) -> list:
        """Generate summaries for texts"""
        vectorizer = model_info['vectorizer']
        compression_ratio = model_info['metrics'].get('compression_ratio', 0.3)
        
        summaries = []
        for text in texts:
            # Simple extractive summarization
            sentences = text.split('.')
            if len(sentences) <= 1:
                summaries.append(text)
                continue
            
            # Take first N sentences based on compression ratio
            num_sentences = max(1, int(len(sentences) * compression_ratio))
            summary = '. '.join(sentences[:num_sentences]).strip()
            if summary and not summary.endswith('.'):
                summary += '.'
            summaries.append(summary)
        
        return summaries


# Global trainer instance
model_trainer = ModelTrainer()