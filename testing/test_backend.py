"""
Test script to verify backend functionality
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

print("Testing backend imports...")

try:
    from app.main import app
    print("✓ Main app imported successfully")
    
    from app.services.file_service import file_service
    print("✓ File service imported successfully")
    
    from app.services.ml_service import ml_service
    print("✓ ML service imported successfully")
    
    from app.core.trainer import model_trainer
    print("✓ Model trainer imported successfully")
    
    print("\n✅ All backend components loaded successfully!")
    print("\nYou can now start the server with:")
    print("  py start_server.py")
    
except Exception as e:
    print(f"\n❌ Error loading backend: {e}")
    import traceback
    traceback.print_exc()
