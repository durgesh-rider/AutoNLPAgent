from typing import Dict, Any, Optional
from app.core.trainer import model_trainer
from app.models.response import TrainingResponse, EvaluationResponse


class MLService:
    def __init__(self):
        self.trained_models = {}

    def train_model(self, dataset_id: str, df, task_type: str,
                   text_column: str, label_column: str, model_type: str = None, 
                   training_config: dict = None, preprocessing_metadata: dict = None) -> TrainingResponse:
        """Train a model for the given dataset"""
        try:
            # Default config if not provided
            if training_config is None:
                training_config = {
                    'epochs': 3,
                    'batch_size': 16,
                    'use_validation': True
                }
            
            # Extract label encoder from preprocessing metadata if available
            label_encoder = None
            if preprocessing_metadata and 'label_encoder' in preprocessing_metadata:
                label_encoder = preprocessing_metadata['label_encoder']
            
            result = model_trainer.train_model(
                df, task_type, text_column, label_column, model_type, training_config, label_encoder
            )

            # Store model info
            self.trained_models[result['model_id']] = {
                'dataset_id': dataset_id,
                'task_type': task_type,
                'model_info': result
            }

            return TrainingResponse(
                success=result.get('success', False),
                message="Model training completed" if result.get('success') else f"Training failed: {result.get('error', 'Unknown error')}",
                model_id=result['model_id'],
                dataset_id=dataset_id,
                model_type=result.get('model_type', 'unknown'),
                training_time=result.get('training_time', 0),
                status="completed" if result.get('success') else "failed"
            )
        except Exception as e:
            return TrainingResponse(
                success=False,
                message=f"Training failed: {str(e)}",
                model_id="",
                dataset_id=dataset_id,
                model_type="unknown",
                training_time=0,
                status="failed"
            )

    def evaluate_model(self, model_id: str, test_data: Optional[Dict] = None) -> EvaluationResponse:
        """Evaluate a trained model"""
        try:
            model_info = self.trained_models.get(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")

            trainer_result = model_info['model_info']

            # Get metrics from training result
            metrics = trainer_result.get('metrics', {})

            # Create confusion matrix if available
            confusion_matrix = None
            classification_report = None

            # For sklearn models, we can generate more detailed metrics
            if trainer_result.get('model_type') == 'sklearn':
                # Additional evaluation could be done here if needed
                pass

            return EvaluationResponse(
                success=True,
                message="Model evaluation completed",
                model_id=model_id,
                metrics=metrics,
                confusion_matrix=confusion_matrix,
                classification_report=classification_report
            )
        except Exception as e:
            return EvaluationResponse(
                success=False,
                message=f"Evaluation failed: {str(e)}",
                model_id=model_id,
                metrics={},
                confusion_matrix=None,
                classification_report=None
            )

    def predict(self, model_id: str, texts: list, return_probabilities: bool = False) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        try:
            predictions = model_trainer.predict(model_id, texts, return_probabilities=return_probabilities)
            
            result = {
                'success': True,
                'predictions': predictions if not return_probabilities else predictions.get('predictions', []),
                'count': len(texts)
            }
            
            # Add probabilities if requested
            if return_probabilities and isinstance(predictions, dict):
                result['probabilities'] = predictions.get('probabilities', [])
                result['confidence_scores'] = predictions.get('confidence_scores', [])
            
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a trained model"""
        return self.trained_models.get(model_id)

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all trained models"""
        return self.trained_models

    def delete_model(self, model_id: str) -> bool:
        """Delete a trained model"""
        if model_id in self.trained_models:
            del self.trained_models[model_id]
            # Also remove from trainer
            if hasattr(model_trainer, 'models') and model_id in model_trainer.models:
                del model_trainer.models[model_id]
            return True
        return False


# Global ML service instance
ml_service = MLService()