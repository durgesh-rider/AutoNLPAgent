# Test Runner for AutoNLP-Agent
# This script runs all tests and generates a summary

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AutoNLP-Agent - Test Suite Runner" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to backend directory
Set-Location -Path "r:\WEB\autonlp\autonlp\backend"

# Run tests
Write-Host "[*] Running Core Functionality Tests..." -ForegroundColor Yellow
Write-Host ""
python -m pytest tests/test_core.py -v --tb=short

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# Get test results
$testOutput = python -m pytest tests/test_core.py -v --tb=line 2>&1
$passed = ($testOutput | Select-String "passed").ToString()

Write-Host ""
Write-Host "$passed" -ForegroundColor Green
Write-Host ""
Write-Host "[OK] All core tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "To view detailed coverage report, run:" -ForegroundColor Yellow
Write-Host "    python -m pytest tests/ --cov=app --cov-report=html" -ForegroundColor Cyan
Write-Host "    Then open: htmlcov/index.html" -ForegroundColor Cyan
Write-Host ""
