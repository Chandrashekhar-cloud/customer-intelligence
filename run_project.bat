@echo off
setlocal enabledelayedexpansion
title E-Commerce Customer Churn Intelligence System

echo =====================================================================
echo  E-Commerce Customer Churn Prediction & Analytics System
echo  End-to-End Machine Learning Pipeline Execution (Assignment 2)
echo =====================================================================
echo.

:: 1. Verify Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    pause
    exit /b 1
)

:: 2. Check and Install Requirements
echo [*] Step 1: Checking dependencies...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [!] Warning: Some dependencies could not be installed automatically. Continuing...
) else (
    echo [+] Dependencies verified successfully.
)
echo.

:: 3. Generate Synthetic Dataset (Milestone 1)
echo [*] Step 2: Generating Synthetic Dataset (Milestone 1)...
python src/data_generation.py
if %errorlevel% neq 0 (
    echo [ERROR] Milestone 1 failed!
    pause
    exit /b 1
)
echo.

:: 4. Run Exploratory Data Analysis (Milestone 2)
echo [*] Step 3: Running Exploratory Data Analysis & Visualizations (Milestone 2)...
python src/eda.py
if %errorlevel% neq 0 (
    echo [ERROR] Milestone 2 failed!
    pause
    exit /b 1
)
echo.

:: 5. Train Models & Optimize Hyperparameters (Milestone 3)
echo [*] Step 4: Training ML Models & Optimizing Hyperparameters (Milestone 3)...
python src/train.py
if %errorlevel% neq 0 (
    echo [ERROR] Milestone 3 failed!
    pause
    exit /b 1
)
echo.

:: 6. Run Automated Test Suite (Milestone 4)
echo [*] Step 5: Executing Pytest Automated Test Suite (Milestone 4)...
pytest tests/ -v
if %errorlevel% neq 0 (
    echo [ERROR] Automated tests failed!
    pause
    exit /b 1
)
echo.

:: 7. Launch Services
echo =====================================================================
echo  ALL 5 MILESTONES VERIFIED SUCCESSFULLY!
echo =====================================================================
echo.
echo Starting Flask REST API Backend on http://127.0.0.1:5000...
start "Flask Churn API" cmd /k "python api/app.py"

echo.
echo To view the interactive dashboard, open frontend/index.html in your browser:
start "" "frontend/index.html"

echo.
echo [+] Backend API running in separate window.
echo [+] Frontend dashboard launched in default browser.
echo.
echo Press any key to exit this runner...
pause >nul
