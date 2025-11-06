import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from fastapi import UploadFile, HTTPException
from app.config import settings


def generate_dataset_id() -> str:
    """Generate a unique dataset ID"""
    return str(uuid.uuid4())


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to beginning

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024)}MB"
        )

    # Check file extension
    ext = get_file_extension(file.filename)
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}"
        )


def save_uploaded_file(file: UploadFile, dataset_id: str) -> str:
    """Save uploaded file to disk and return file path"""
    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)

    # Generate file path
    file_extension = get_file_extension(file.filename)
    file_path = upload_dir / f"{dataset_id}{file_extension}"

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    return str(file_path)


def load_dataset(file_path: str) -> Tuple[pd.DataFrame, list]:
    """Load dataset from file and return DataFrame and column names"""
    file_extension = get_file_extension(file_path)

    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.txt':
            # Assume tab-separated or space-separated text
            df = pd.read_csv(file_path, sep='\t', header=0)
            if len(df.columns) == 1:  # Try space separation if only one column
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
        elif file_extension == '.pdf':
            # For PDF files, we'll need to extract text first
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"

                    # Convert to simple dataframe with text column
                    df = pd.DataFrame({'text': [text]})
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF processing requires PyPDF2. Install with: pip install PyPDF2")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Basic data validation
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        columns = df.columns.tolist()
        return df, columns

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")


def get_dataset_info(file_path: str) -> Tuple[int, list]:
    """Get dataset information without loading full data"""
    df, columns = load_dataset(file_path)
    return len(df), columns


def cleanup_file(file_path: str) -> None:
    """Remove file from disk"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Ignore cleanup errors