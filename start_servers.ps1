# Start AutoNLP-Agent Servers
# This script starts both backend and frontend in separate windows

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Starting AutoNLP-Agent Full Stack" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start Backend in a new window
Write-Host "[1/2] Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; py start_server.py"
Write-Host "      Backend starting in separate window..." -ForegroundColor Green

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start Frontend in a new window
Write-Host ""
Write-Host "[2/2] Starting Frontend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run serve"
Write-Host "      Frontend starting in separate window..." -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Servers Starting Successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access Points:" -ForegroundColor White
Write-Host "  Backend API:   http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs:      http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  Frontend App:  http://localhost:8081" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop servers, close the PowerShell windows" -ForegroundColor Yellow
Write-Host ""
