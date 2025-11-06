from fastapi import APIRouter, HTTPException
from app.services.ml_service import ml_service
from app.services.viz_service import viz_service
from app.models.response import EvaluationResponse, VisualizationResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/{model_id}", response_model=EvaluationResponse)
async def evaluate_model(model_id: str):
    """Evaluate a trained model and return metrics"""
    try:
        result = ml_service.evaluate_model(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/visualize/{evaluation_id}", response_model=VisualizationResponse)
async def generate_visualizations(evaluation_id: str):
    """Generate visualizations for model evaluation results"""
    try:
        result = viz_service.generate_visualizations(evaluation_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")


@router.get("/visualizations/{viz_id}")
async def get_visualization(viz_id: str):
    """Get visualization data by ID"""
    try:
        viz_data = viz_service.get_visualization(viz_id)
        if not viz_data:
            raise HTTPException(status_code=404, detail="Visualization not found")
        return viz_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{viz_id}")
async def get_report(viz_id: str, format: str = "json"):
    """Get complete evaluation report"""
    try:
        report = viz_service.export_report(viz_id, format)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/metrics/{model_id}")
async def get_model_metrics(model_id: str):
    """Get evaluation metrics for a specific model"""
    try:
        # Get model info
        model_info = ml_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get evaluation results
        evaluation = ml_service.evaluate_model(model_id)

        return {
            'model_id': model_id,
            'metrics': evaluation.metrics if evaluation.success else {},
            'evaluation_status': 'success' if evaluation.success else 'failed'
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))