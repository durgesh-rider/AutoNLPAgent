from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.services.file_service import file_service
from app.models.response import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    task_type: Optional[str] = Form(None)
):
    """
    Upload a dataset file (CSV, TXT, Excel).

    - **file**: The dataset file to upload
    - **task_type**: Optional task type override (auto-detected if not provided)
    """
    try:
        result = await file_service.upload_dataset(file, task_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets")
async def get_datasets():
    """Get all uploaded datasets"""
    try:
        datasets = file_service.get_all_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset by ID"""
    try:
        dataset = file_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Return dataset info without the dataframe (too large for JSON)
        return {
            'dataset_id': dataset_id,
            'file_path': dataset.get('file_path'),
            'filename': dataset.get('filename'),
            'columns': dataset.get('columns', []),
            'row_count': dataset.get('row_count', 0),
            'task_type': dataset.get('task_type', 'unknown')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete dataset by ID"""
    try:
        success = file_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))