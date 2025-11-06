from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC


class BaseResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UploadResponse(BaseResponse):
    dataset_id: str
    filename: str
    file_size: int
    columns: List[str]
    row_count: int
    detected_task: str


class TaskDetectionResponse(BaseResponse):
    dataset_id: str
    detected_task: str
    confidence: float
    reasoning: str


class PreprocessingResponse(BaseResponse):
    dataset_id: str
    preprocessing_steps: List[str]
    processed_data_shape: tuple
    missing_values_handled: int
    text_cleaned: bool


class TrainingResponse(BaseResponse):
    model_id: str
    dataset_id: str
    model_type: str
    training_time: float
    status: str  # 'completed', 'running', 'failed'


class EvaluationResponse(BaseResponse):
    model_id: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None


class VisualizationResponse(BaseResponse):
    result_id: str
    charts: List[Dict[str, Any]]
    download_urls: Optional[Dict[str, str]] = None


class DatasetInfo(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    size: int
    columns: List[str]
    row_count: int
    task_type: str
    status: str  # 'uploaded', 'processing', 'ready', 'error'


class ModelInfo(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    created_date: datetime
    status: str
    metrics: Optional[Dict[str, float]] = None


class ProjectSummary(BaseModel):
    total_datasets: int
    total_models: int
    recent_uploads: List[DatasetInfo]
    recent_models: List[ModelInfo]