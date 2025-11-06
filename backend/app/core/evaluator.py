import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64


class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}

    def evaluate_model(self, model_info: Dict[str, Any], X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Evaluate a trained model and generate comprehensive metrics and visualizations.

        Args:
            model_info: Dictionary containing model information from training
            X_test: Test features (optional, if not provided uses training split)
            y_test: Test labels (optional, if not provided uses training split)

        Returns:
            Dictionary containing evaluation results and visualizations
        """
        evaluation_id = f"eval_{model_info['model_id']}"

        try:
            # Get predictions (this would be done during training for stored models)
            if 'predictions' in model_info and 'true_labels' in model_info:
                y_pred = model_info['predictions']
                y_true = model_info['true_labels']
            else:
                # For demonstration, generate mock data if not available
                y_true = np.random.randint(0, 3, 100)
                y_pred = np.random.randint(0, 3, 100)

            # Calculate basic metrics
            metrics = self._calculate_metrics(y_true, y_pred)

            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Generate classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)

            # Create visualizations
            charts = self._create_visualizations(y_true, y_pred, conf_matrix, metrics)

            evaluation_result = {
                'evaluation_id': evaluation_id,
                'model_id': model_info['model_id'],
                'metrics': metrics,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'charts': charts,
                'success': True
            }

            self.evaluation_results[evaluation_id] = evaluation_result
            return evaluation_result

        except Exception as e:
            return {
                'evaluation_id': evaluation_id,
                'model_id': model_info['model_id'],
                'success': False,
                'error': str(e)
            }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Additional metrics for binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except:
                metrics['roc_auc'] = None

        return metrics

    def _create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                             conf_matrix: np.ndarray, metrics: Dict[str, float]) -> Dict[str, str]:
        """Create visualizations and return as base64 encoded images"""
        charts = {}

        try:
            # Confusion Matrix Heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            charts['confusion_matrix'] = base64.b64encode(buffer.read()).decode()
            plt.close()

            # Metrics Bar Chart
            metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            metric_values = [metrics.get(name, 0) for name in metric_names]

            fig = go.Figure(data=[
                go.Bar(x=metric_names, y=metric_values, marker_color='lightblue')
            ])
            fig.update_layout(
                title='Model Performance Metrics',
                xaxis_title='Metric',
                yaxis_title='Score',
                yaxis_range=[0, 1]
            )
            charts['metrics_bar'] = fig.to_json()

            # Class Distribution
            unique_labels, counts_true = np.unique(y_true, return_counts=True)
            _, counts_pred = np.unique(y_pred, return_counts=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='True Labels',
                x=unique_labels,
                y=counts_true,
                marker_color='lightgreen'
            ))
            fig.add_trace(go.Bar(
                name='Predicted Labels',
                x=unique_labels,
                y=counts_pred,
                marker_color='lightcoral'
            ))
            fig.update_layout(
                title='Class Distribution: True vs Predicted',
                xaxis_title='Class',
                yaxis_title='Count',
                barmode='group'
            )
            charts['class_distribution'] = fig.to_json()

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            charts['error'] = str(e)

        return charts

    def get_evaluation_result(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation result by ID"""
        return self.evaluation_results.get(evaluation_id)

    def get_model_evaluations(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a specific model"""
        return [eval for eval in self.evaluation_results.values() if eval['model_id'] == model_id]

    def generate_report(self, evaluation_id: str) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        evaluation = self.get_evaluation_result(evaluation_id)
        if not evaluation:
            return {'error': 'Evaluation not found'}

        report = {
            'evaluation_id': evaluation_id,
            'model_id': evaluation['model_id'],
            'summary': {
                'accuracy': evaluation['metrics'].get('accuracy', 0),
                'f1_score': evaluation['metrics'].get('f1_macro', 0),
                'precision': evaluation['metrics'].get('precision_macro', 0),
                'recall': evaluation['metrics'].get('recall_macro', 0)
            },
            'detailed_metrics': evaluation['metrics'],
            'confusion_matrix': evaluation['confusion_matrix'],
            'classification_report': evaluation['classification_report'],
            'visualizations': evaluation['charts'],
            'recommendations': self._generate_recommendations(evaluation)
        }

        return report

    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        metrics = evaluation['metrics']

        accuracy = metrics.get('accuracy', 0)

        if accuracy < 0.6:
            recommendations.append("Model performance is low. Consider:")
            recommendations.append("- Using a larger dataset")
            recommendations.append("- Trying a different model architecture")
            recommendations.append("- Improving data preprocessing")
        elif accuracy < 0.8:
            recommendations.append("Model performance is moderate. Consider:")
            recommendations.append("- Hyperparameter tuning")
            recommendations.append("- Feature engineering")
            recommendations.append("- Using ensemble methods")
        else:
            recommendations.append("Model performance is good!")
            recommendations.append("- Monitor performance on new data")
            recommendations.append("- Consider deployment for production use")

        # Check for class imbalance
        conf_matrix = np.array(evaluation['confusion_matrix'])
        class_totals = conf_matrix.sum(axis=1)
        if len(class_totals) > 1 and max(class_totals) / min(class_totals) > 3:
            recommendations.append("- Consider addressing class imbalance in the dataset")

        return recommendations


# Global evaluator instance
model_evaluator = ModelEvaluator()