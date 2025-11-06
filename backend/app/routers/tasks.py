from fastapi import APIRouter, HTTPException
from typing import Optional
import pandas as pd
from app.services.file_service import file_service
from app.services.nlp_service import nlp_service
from app.models.response import TaskDetectionResponse, PreprocessingResponse
from app.models.dataset_config import DatasetConfiguration, DatasetColumnsResponse, ColumnInfo

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/detect/{dataset_id}", response_model=TaskDetectionResponse)
async def detect_task(dataset_id: str):
    """Detect NLP task type for uploaded dataset"""
    try:
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = dataset.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="Dataset not properly loaded")

        result = nlp_service.detect_task(df, dataset_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task detection failed: {str(e)}")


@router.post("/preprocess/{dataset_id}", response_model=PreprocessingResponse)
async def preprocess_dataset(dataset_id: str):
    """Preprocess dataset for NLP tasks"""
    try:
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = dataset.get('dataframe')
        task_type = dataset.get('task_type', 'classification')

        if df is None:
            raise HTTPException(status_code=400, detail="Dataset not properly loaded")

        result = nlp_service.preprocess_data(df, dataset_id, task_type)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.get("/supported")
async def get_supported_tasks():
    """Get list of supported NLP tasks"""
    return {
        "tasks": [
            {
                "name": "classification",
                "description": "Text classification into predefined categories"
            },
            {
                "name": "sentiment_analysis",
                "description": "Determine positive, negative, or neutral sentiment"
            },
            {
                "name": "named_entity_recognition",
                "description": "Extract entities like names, dates, organizations"
            },
            {
                "name": "question_answering",
                "description": "Answer questions based on provided context"
            },
            {
                "name": "text_summarization",
                "description": "Generate concise summaries of longer texts"
            }
        ]
    }


@router.get("/columns/{dataset_id}", response_model=DatasetColumnsResponse)
async def get_dataset_columns(dataset_id: str):
    """
    Get detailed information about dataset columns including recommendations
    for text and label columns, and auto-detected task type.
    """
    try:
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = dataset.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="Dataset not properly loaded")
        
        # Analyze each column
        columns_info = []
        text_keywords = ['text', 'content', 'message', 'review', 'comment', 'description', 'sentence', 'document', 'title', 'body']
        label_keywords = ['label', 'target', 'class', 'category', 'sentiment', 'type', 'rating']
        
        recommended_text = []
        recommended_label = None
        
        for col in df.columns:
            col_data = df[col]
            dtype = str(col_data.dtype)
            unique_count = col_data.nunique()
            missing_count = col_data.isnull().sum()
            
            # Get sample values (non-null, up to 5)
            sample_values = col_data.dropna().head(5).astype(str).tolist()
            
            # Determine column type
            is_numeric = col_data.dtype in ['int64', 'float64', 'int32', 'float32']
            is_text = col_data.dtype == 'object' and col_data.apply(lambda x: isinstance(x, str) if pd.notna(x) else False).any()
            is_categorical = (unique_count / len(col_data)) < 0.1 and unique_count > 1  # Less than 10% unique
            
            column_info = ColumnInfo(
                name=col,
                dtype=dtype,
                unique_count=int(unique_count),
                missing_count=int(missing_count),
                sample_values=sample_values,
                is_text=bool(is_text),
                is_numeric=bool(is_numeric),
                is_categorical=bool(is_categorical)
            )
            columns_info.append(column_info)
            
            # Recommend text columns
            if is_text and any(keyword in col.lower() for keyword in text_keywords):
                recommended_text.append(col)
            elif is_text and not recommended_text:
                recommended_text.append(col)  # Fallback to first text column
            
            # Recommend label column
            if not recommended_label:
                if any(keyword in col.lower() for keyword in label_keywords):
                    if 2 <= unique_count <= 50:  # Reasonable number of classes
                        recommended_label = col
                elif is_categorical and 2 <= unique_count <= 50:
                    recommended_label = col
        
        # Auto-detect task type
        auto_task = dataset.get('task_type', 'classification')
        
        return DatasetColumnsResponse(
            success=True,
            dataset_id=dataset_id,
            columns=columns_info,
            recommended_text_columns=recommended_text,
            recommended_label_column=recommended_label,
            auto_detected_task=auto_task
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column analysis failed: {str(e)}")


@router.post("/configure/{dataset_id}")
async def configure_dataset(dataset_id: str, config: DatasetConfiguration):
    """
    Configure dataset with user-selected columns and task type.
    This overrides auto-detection and stores user preferences.
    """
    try:
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = dataset.get('dataframe')
        if df is None:
            raise HTTPException(status_code=400, detail="Dataset not properly loaded")
        
        # Validate selected columns exist
        all_columns = df.columns.tolist()
        for col in config.text_columns:
            if col not in all_columns:
                raise HTTPException(status_code=400, detail=f"Text column '{col}' not found in dataset")
        
        if config.label_column and config.label_column not in all_columns:
            raise HTTPException(status_code=400, detail=f"Label column '{config.label_column}' not found")
        
        # Store configuration with dataset
        dataset['user_config'] = {
            'text_columns': config.text_columns,
            'label_column': config.label_column,
            'task_type': config.task_type,
            'exclude_columns': config.exclude_columns or [],
            'combine_text_columns': config.combine_text_columns
        }
        
        # Update task type
        dataset['task_type'] = config.task_type
        
        file_service.datasets[dataset_id] = dataset
        
        return {
            "success": True,
            "message": "Dataset configuration saved successfully",
            "config": dataset['user_config']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")
