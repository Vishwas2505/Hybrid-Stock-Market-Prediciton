@echo off
title HIMM Stock Prediction Server
color 0b

echo ============================================================
echo   Hybrid Information Mixing Module Stock Prediction System
echo   CVR College of Engineering - CSE B.Tech Mini Project
echo ============================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.7+ and try again.
    pause
    exit /b
)

echo Python is available. Starting server...
echo.
echo Application will be available at: http://localhost:8000
echo Press Ctrl+C in this window to stop the server.
echo.
echo ============================================================

python backend.py

pause
