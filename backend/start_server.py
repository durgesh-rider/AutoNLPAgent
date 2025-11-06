"""
Start the AutoNLP backend server
This script ensures the correct Python path is set before starting uvicorn
"""
import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variable
os.environ['PYTHONPATH'] = str(backend_dir)

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("AutoNLP-Agent - Backend Server")
    print("=" * 60)
    print()
    print(f"Backend directory: {backend_dir}")
    print(f"Python path: {sys.path[0]}")
    print()
    print("Access Points:")
    print("  API Server:     http://localhost:8000")
    print("  API Docs:       http://localhost:8000/docs")
    print("  ReDoc:          http://localhost:8000/redoc")
    print("  Health Check:   http://localhost:8000/health")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
