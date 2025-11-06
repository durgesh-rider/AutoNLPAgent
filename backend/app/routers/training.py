from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
import traceback
from app.services.file_service import file_service
from app.services.ml_service import ml_service
from app.models.response import TrainingResponse, EvaluationResponse

router = APIRouter(prefix="/training", tags=["training"])


class TrainingConfig(BaseModel):
    """Configuration for model training with user control"""
    model_type: Optional[str] = Field(
        default=None, 
        description="Model type: 'sklearn' (traditional ML) or 'transformer' (deep learning). Auto-selected based on data size if not specified."
    )
    epochs: int = Field(
        default=3, 
        ge=1, 
        le=100, 
        description="Number of training epochs (1-100). More epochs = longer training but potentially better results."
    )
    batch_size: int = Field(
        default=16, 
        ge=1, 
        le=128, 
        description="Batch size for training (1-128). Larger batches = faster training but more memory."
    )
    learning_rate: Optional[float] = Field(
        default=None, 
        gt=0, 
        lt=1, 
        description="Learning rate for training (0-1). Controls how quickly the model learns. Auto-selected if not specified."
    )
    use_validation: bool = Field(
        default=True, 
        description="Use validation split (20% of data) for better evaluation. Recommended: True"
    )
    max_length: Optional[int] = Field(
        default=128, 
        ge=32, 
        le=512, 
        description="Maximum sequence length for text (32-512). Longer = more context but slower training."
    )
    early_stopping: bool = Field(
        default=False, 
        description="Stop training early if validation performance stops improving. Prevents overfitting."
    )
    test_size: float = Field(
        default=0.2, 
        gt=0, 
        lt=0.5, 
        description="Proportion of data to use for testing (0-0.5). Default 0.2 = 20%"
    )
    text_column: Optional[str] = Field(
        default=None, 
        description="Name of the text column to use. Auto-detected if not specified."
    )
    label_column: Optional[str] = Field(
        default=None, 
        description="Name of the label column to use. Auto-detected if not specified."
    )


class PredictionRequest(BaseModel):
    texts: List[str] = Field(
        default=None,
        description="List of text strings to predict. Use this for direct text input."
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset ID to use for batch prediction. If provided, text_column must be specified."
    )
    text_column: Optional[str] = Field(
        default=None,
        description="Column name containing text to predict when using dataset_id."
    )
    return_probabilities: bool = Field(
        default=False,
        description="Return prediction probabilities/confidence scores."
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for processing predictions."
    )


@router.post("/{dataset_id}", response_model=TrainingResponse)
async def train_model(dataset_id: str, config: Optional[TrainingConfig] = None):
    """
    Train a model on the uploaded dataset with customizable parameters.
    
    Provides full control over:
    - Model type (traditional ML vs. deep learning)
    - Training parameters (epochs, batch size, learning rate)
    - Text processing (max length, columns to use)
    - Validation strategy (validation split, early stopping)
    """
    try:
        # Use default config if not provided
        if config is None:
            config = TrainingConfig()
        
        # Get dataset
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

        df = dataset.get('dataframe')
        task_type = dataset.get('task_type', 'classification')
        preprocessing_metadata = dataset.get('preprocessing_metadata', {})
        user_config = dataset.get('user_config', {})  # Get user configuration if available

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Dataset not properly loaded or is empty")

        # Identify columns - Priority: config param > user_config > preprocessing_metadata > auto-detect
        if config.text_column and config.label_column:
            # User specified in training config
            text_column = config.text_column
            label_column = config.label_column
        elif user_config.get('text_columns') and user_config.get('label_column'):
            # User configured via /tasks/configure endpoint
            text_cols = user_config['text_columns']
            label_column = user_config['label_column']
            
            # If multiple text columns, combine them
            if len(text_cols) > 1 and user_config.get('combine_text_columns', True):
                # Create combined column
                df['_combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
                text_column = '_combined_text'
            else:
                text_column = text_cols[0]
        else:
            # Auto-detect columns (fallback)
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if not text_columns:
                raise HTTPException(status_code=400, detail="No text columns found in dataset")
            
            text_column = text_columns[0]
            
            # Try to find label column
            if preprocessing_metadata.get('label_column'):
                label_column = preprocessing_metadata['label_column']
            else:
                # Use last column as label
                label_column = df.columns[-1]

        # Build training configuration from user parameters
        training_config = {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'use_validation': config.use_validation,
            'max_length': config.max_length,
            'early_stopping': config.early_stopping,
            'test_size': config.test_size
        }
        
        # Add learning rate if specified
        if config.learning_rate:
            training_config['learning_rate'] = config.learning_rate

        # Train the model
        result = ml_service.train_model(
            dataset_id, 
            df, 
            task_type, 
            text_column, 
            label_column, 
            config.model_type,
            training_config,
            preprocessing_metadata
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/evaluate/{model_id}", response_model=EvaluationResponse)
async def evaluate_model(model_id: str):
    """Evaluate a trained model"""
    try:
        result = ml_service.evaluate_model(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/predict/{model_id}")
async def predict(model_id: str, request: PredictionRequest):
    """
    Make predictions using a trained model.
    
    Two modes:
    1. Direct text input: Provide 'texts' array
    2. Dataset prediction: Provide 'dataset_id' and 'text_column'
    
    Options:
    - return_probabilities: Get confidence scores with predictions
    - batch_size: Control prediction batch size
    """
    try:
        # Validate request
        if not request.texts and not request.dataset_id:
            raise HTTPException(
                status_code=400, 
                detail="Either 'texts' or 'dataset_id' must be provided"
            )
        
        if request.dataset_id and not request.text_column:
            raise HTTPException(
                status_code=400,
                detail="'text_column' is required when using 'dataset_id'"
            )
        
        # Get texts from dataset if dataset_id is provided
        if request.dataset_id:
            from app.services.file_service import file_service
            dataset = file_service.get_dataset(request.dataset_id)
            
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            df = dataset.get('dataframe')
            if df is None:
                raise HTTPException(status_code=400, detail="Dataset not properly loaded")
            
            if request.text_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{request.text_column}' not found in dataset. Available columns: {df.columns.tolist()}"
                )
            
            # Extract texts from specified column
            texts = df[request.text_column].fillna('').astype(str).tolist()
        else:
            texts = request.texts
        
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided for prediction")
        
        # Make predictions
        result = ml_service.predict(
            model_id, 
            texts,
            return_probabilities=request.return_probabilities
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models")
async def get_models():
    """Get all trained models"""
    try:
        models = ml_service.get_all_models()
        
        # Return only serializable model info
        serializable_models = {}
        for model_id, model_data in models.items():
            serializable_models[model_id] = {
                'dataset_id': model_data.get('dataset_id'),
                'task_type': model_data.get('task_type'),
                'model_info': {
                    'model_id': model_data.get('model_info', {}).get('model_id'),
                    'model_type': model_data.get('model_info', {}).get('model_type'),
                    'task_type': model_data.get('model_info', {}).get('task_type'),
                    'training_samples': model_data.get('model_info', {}).get('training_samples'),
                    'test_samples': model_data.get('model_info', {}).get('test_samples'),
                    'metrics': model_data.get('model_info', {}).get('metrics', {}),
                    'training_time': model_data.get('model_info', {}).get('training_time')
                }
            }
        
        return {"models": serializable_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    Get detailed model information by ID.
    
    Returns:
    - Model metadata (task type, training samples, etc.)
    - Training configuration
    - Column names used during training (text_column, label_column)
    - Performance metrics
    """
    try:
        model_info = ml_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Extract model_info from nested structure
        inner_model_info = model_info.get('model_info', {})
        
        # Return comprehensive serializable data
        serializable_info = {
            'dataset_id': model_info.get('dataset_id'),
            'task_type': model_info.get('task_type'),
            'model_info': {
                'model_id': inner_model_info.get('model_id'),
                'model_type': inner_model_info.get('model_type'),
                'task_type': inner_model_info.get('task_type'),
                'training_samples': inner_model_info.get('training_samples'),
                'test_samples': inner_model_info.get('test_samples'),
                'metrics': inner_model_info.get('metrics', {}),
                'training_time': inner_model_info.get('training_time'),
                'text_column': inner_model_info.get('text_column'),  # Column used for text
                'label_column': inner_model_info.get('label_column'),  # Column used for labels
                'config': inner_model_info.get('config', {})
            }
        }
        
        return serializable_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    try:
        success = ml_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"message": "Model deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download a trained model as a zip file"""
    from fastapi.responses import FileResponse
    import os
    import zipfile
    import tempfile
    
    try:
        # Get model info
        model_info = ml_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_path = model_info.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Create a temporary zip file
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, f"model_{model_id}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isdir(model_path):
                # If it's a directory (transformer models), zip all files
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        zipf.write(file_path, arcname)
            else:
                # If it's a single file (sklearn models)
                zipf.write(model_path, os.path.basename(model_path))
        
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=f"model_{model_id}.zip"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")