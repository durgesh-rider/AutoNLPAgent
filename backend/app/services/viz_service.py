from typing import Dict, Any, Optional
from app.core.evaluator import model_evaluator
from app.models.response import VisualizationResponse


class VisualizationService:
    def __init__(self):
        self.visualizations = {}

    def generate_visualizations(self, evaluation_id: str) -> VisualizationResponse:
        """Generate visualizations for a model evaluation"""
        try:
            evaluation = model_evaluator.get_evaluation_result(evaluation_id)
            if not evaluation:
                raise ValueError(f"Evaluation {evaluation_id} not found")

            # Generate comprehensive report with visualizations
            report = model_evaluator.generate_report(evaluation_id)

            # Store visualization data
            viz_id = f"viz_{evaluation_id}"
            self.visualizations[viz_id] = {
                'evaluation_id': evaluation_id,
                'report': report,
                'charts': evaluation.get('charts', {})
            }

            return VisualizationResponse(
                success=True,
                message="Visualizations generated successfully",
                result_id=viz_id,
                charts=list(report.get('visualizations', {}).keys()),
                download_urls=self._generate_download_urls(viz_id)
            )

        except Exception as e:
            return VisualizationResponse(
                success=False,
                message=f"Visualization generation failed: {str(e)}",
                result_id="",
                charts=[],
                download_urls=None
            )

    def get_visualization(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """Get visualization data by ID"""
        return self.visualizations.get(viz_id)

    def get_report_data(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """Get complete report data for a visualization"""
        viz_data = self.get_visualization(viz_id)
        if viz_data:
            return viz_data.get('report')
        return None

    def _generate_download_urls(self, viz_id: str) -> Dict[str, str]:
        """Generate download URLs for reports and charts"""
        # In a real implementation, these would be actual URLs to downloadable files
        return {
            'pdf_report': f'/download/{viz_id}/report.pdf',
            'csv_metrics': f'/download/{viz_id}/metrics.csv',
            'png_confusion_matrix': f'/download/{viz_id}/confusion_matrix.png',
            'png_metrics_chart': f'/download/{viz_id}/metrics.png'
        }

    def export_report(self, viz_id: str, format: str = 'json') -> Dict[str, Any]:
        """Export visualization report in different formats"""
        report = self.get_report_data(viz_id)
        if not report:
            return {'error': 'Visualization not found'}

        if format == 'json':
            return report
        elif format == 'summary':
            return {
                'model_id': report.get('model_id'),
                'accuracy': report.get('summary', {}).get('accuracy'),
                'f1_score': report.get('summary', {}).get('f1_score'),
                'recommendations': report.get('recommendations', [])
            }
        else:
            return {'error': f'Unsupported format: {format}'}


# Global visualization service instance
viz_service = VisualizationService()