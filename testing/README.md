# Testing Scripts

This folder contains test scripts and demo files for the AutoNLP-Agent application.

## Files

### Test Scripts

- **`test_backend.py`** - Basic backend API tests
  - Tests file upload
  - Tests dataset retrieval
  - Tests model training endpoint
  
- **`test_full_backend.py`** - End-to-end integration tests
  - Creates a test dataset
  - Uploads it to the backend
  - Trains a model
  - Makes predictions
  - Validates results

- **`run_tests.ps1`** - PowerShell script to run the pytest test suite

### Demo Scripts

- **`demo.py`** - Complete workflow demonstration
  - Demonstrates the full ML pipeline
  - Shows API usage examples
  - Useful for testing after deployment

### Notebooks

- **`colab_autonlp_agent.ipynb`** - Google Colab notebook
  - For running the application in Google Colab
  - Useful for cloud-based testing

## Running Tests

### Backend API Tests

```bash
# Run from the backend directory
cd ../backend
python -m pytest tests/ -v
```

### Full Backend Test

```bash
# Run from this directory
python test_full_backend.py
```

### Demo Script

```bash
# Make sure the backend is running first
# Start backend: cd ../backend && py start_server.py
python demo.py
```

### PowerShell Test Runner

```powershell
.\run_tests.ps1
```

## Prerequisites

- Backend server must be running (for integration tests and demo)
- All backend dependencies must be installed (`pip install -r ../backend/requirements.txt`)
- pytest must be installed for running the test suite

## Expected Output

All tests should pass if the application is set up correctly. Example output:

```
✅ Server is healthy
✅ Dataset uploaded successfully
✅ Model trained successfully
✅ Predictions working correctly
```

## Troubleshooting

If tests fail:

1. **Check if backend is running**: Visit http://localhost:8000/health
2. **Check dependencies**: Ensure all packages are installed
3. **Check uploads folder**: Ensure `backend/uploads/` exists
4. **Check logs**: View backend terminal for error messages
