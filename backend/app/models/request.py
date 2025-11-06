from pydantic import BaseModel
from typing import Optional, List
from fastapi import UploadFile


class UploadRequest(BaseModel):
    file: UploadFile
    task_type: Optional[str] = None  # Auto-detect if not provided


class PreprocessingRequest(BaseModel):
    dataset_id: str
    options: Optional[dict] = None


class TrainingRequest(BaseModel):
    dataset_id: str
    model_type: Optional[str] = None  # Auto-select if not provided
    hyperparameters: Optional[dict] = None


class EvaluationRequest(BaseModel):
    model_id: str
    test_data: Optional[str] = None  # Use validation split if not provided


class VisualizationRequest(BaseModel):
    result_id: str
    chart_types: Optional[List[str]] = None