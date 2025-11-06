import os
from typing import Dict, Any
from pathlib import Path
from app.utils.file_utils import (
    generate_dataset_id,
    validate_file,
    save_uploaded_file,
    load_dataset,
    get_dataset_info,
    cleanup_file
)
from app.core.task_detector import detect_task_type
from app.models.response import UploadResponse


class FileService:
    def __init__(self):
        self.uploaded_datasets: Dict[str, Dict[str, Any]] = {}

    async def upload_dataset(self, file, task_type: str = None) -> UploadResponse:
        """Upload and process a dataset file"""
        # Validate file
        validate_file(file)

        # Generate dataset ID
        dataset_id = generate_dataset_id()

        # Save file
        file_path = save_uploaded_file(file, dataset_id)

        try:
            # Get dataset info
            row_count, columns = get_dataset_info(file_path)

            # Load dataset for task detection
            df, _ = load_dataset(file_path)

            # Detect task type if not provided
            if not task_type:
                task_type = detect_task_type(df)

            # Store dataset info
            self.uploaded_datasets[dataset_id] = {
                'file_path': file_path,
                'filename': file.filename,
                'columns': columns,
                'row_count': row_count,
                'task_type': task_type,
                'dataframe': df
            }

            return UploadResponse(
                success=True,
                message="Dataset uploaded successfully",
                dataset_id=dataset_id,
                filename=file.filename,
                file_size=os.path.getsize(file_path),
                columns=columns,
                row_count=row_count,
                detected_task=task_type
            )

        except Exception as e:
            # Cleanup on error
            cleanup_file(file_path)
            raise e

    def get_dataset(self, dataset_id: str):
        """Get dataset by ID"""
        if dataset_id not in self.uploaded_datasets:
            return None
        
        dataset_info = self.uploaded_datasets[dataset_id]
        
        # Reload dataframe if it's not in memory
        if 'dataframe' not in dataset_info or dataset_info['dataframe'] is None:
            try:
                df, _ = load_dataset(dataset_info['file_path'])
                dataset_info['dataframe'] = df
                self.uploaded_datasets[dataset_id] = dataset_info
            except Exception as e:
                print(f"Error reloading dataset {dataset_id}: {e}")
                return None
        
        return dataset_info

    def get_all_datasets(self):
        """Get all uploaded datasets"""
        # Return dataset info without dataframe for listing
        datasets = []
        for dataset_id, dataset_info in self.uploaded_datasets.items():
            datasets.append({
                'dataset_id': dataset_id,
                'file_path': dataset_info.get('file_path'),
                'filename': dataset_info.get('filename'),
                'columns': dataset_info.get('columns', []),
                'row_count': dataset_info.get('row_count', 0),
                'task_type': dataset_info.get('task_type', 'unknown')
            })
        return datasets

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset by ID"""
        if dataset_id in self.uploaded_datasets:
            dataset_info = self.uploaded_datasets[dataset_id]
            cleanup_file(dataset_info['file_path'])
            del self.uploaded_datasets[dataset_id]
            return True
        return False


# Global file service instance
file_service = FileService()