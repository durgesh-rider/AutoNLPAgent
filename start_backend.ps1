# Quick Start Script for AutoNLP-Agent
# This script starts the backend server

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AutoNLP-Agent - Backend Server" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[*] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = py --version 2>&1
Write-Host "    $pythonVersion" -ForegroundColor Cyan

# Set PYTHONPATH
Write-Host ""
Write-Host "[*] Setting PYTHONPATH..." -ForegroundColor Yellow
$env:PYTHONPATH = "$PSScriptRoot\backend"
Write-Host "    PYTHONPATH=$env:PYTHONPATH" -ForegroundColor Cyan

# Change to backend directory
Write-Host ""
Write-Host "[*] Changing to project root directory..." -ForegroundColor Yellow
Set-Location -Path "$PSScriptRoot"
Write-Host "    $(Get-Location)" -ForegroundColor Cyan

# Start server
Write-Host ""
Write-Host "[*] Starting FastAPI server..." -ForegroundColor Yellow
Write-Host "    Host: 0.0.0.0" -ForegroundColor Cyan
Write-Host "    Port: 8000" -ForegroundColor Cyan
Write-Host "    Reload: Enabled" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Green
Write-Host "    API Server:     http://localhost:8000" -ForegroundColor Cyan
Write-Host "    API Docs:       http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "    ReDoc:          http://localhost:8000/redoc" -ForegroundColor Cyan
Write-Host "    Health Check:   http://localhost:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start uvicorn
py backend\start_server.py
