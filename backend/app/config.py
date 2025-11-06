from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")
    
    # Application settings
    app_name: str = "AutoNLP-Agent"
    debug: bool = True
    version: str = "1.0.0"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # File upload settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = [".csv", ".txt", ".xlsx", ".xls", ".pdf"]
    upload_dir: str = "uploads"

    # ML settings
    max_training_time: int = 3600  # 1 hour
    default_batch_size: int = 16
    default_epochs: int = 3

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


settings = Settings()