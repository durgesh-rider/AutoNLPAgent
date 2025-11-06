from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class DatasetConfiguration(BaseModel):
    """User configuration for dataset processing"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_columns": ["review_text", "title"],
                "label_column": "sentiment",
                "task_type": "sentiment_analysis",
                "exclude_columns": ["id", "timestamp"],
                "combine_text_columns": True
            }
        }
    )
    
    text_columns: List[str] = Field(
        description="Column(s) containing text data. Multiple columns will be concatenated."
    )
    label_column: Optional[str] = Field(
        default=None,
        description="Column containing target labels for classification/sentiment tasks"
    )
    task_type: str = Field(
        description="NLP task type: classification, sentiment_analysis, ner, qa, summarization"
    )
    exclude_columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to exclude from processing"
    )
    combine_text_columns: bool = Field(
        default=True,
        description="If true, concatenate multiple text columns with space separator"
    )


class ColumnInfo(BaseModel):
    """Information about a dataset column"""
    name: str
    dtype: str
    unique_count: int
    missing_count: int
    sample_values: List[str]
    is_text: bool
    is_numeric: bool
    is_categorical: bool


class DatasetColumnsResponse(BaseModel):
    """Response containing dataset column information"""
    success: bool
    dataset_id: str
    columns: List[ColumnInfo]
    recommended_text_columns: List[str]
    recommended_label_column: Optional[str]
    auto_detected_task: str
