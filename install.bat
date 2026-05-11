@echo off
REM Installation script for Anaconda users
REM Run this from Anaconda Prompt

echo ========================================
echo Hybrid LLM Hallucination Detection System
echo Installation Script
echo ========================================
echo.

echo Step 1: Installing PyTorch (CPU version)...
call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    echo Please run this script from Anaconda Prompt
    pause
    exit /b 1
)

echo.
echo Step 2: Installing other dependencies...
call pip install transformer-lens transformers sentence-transformers streamlit plotly scikit-learn matplotlib numpy
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run the system:
echo   1. Streamlit UI:  streamlit run app.py
echo   2. Example:       python example.py
echo   3. CLI:           python main.py --prompt "Your question"
echo.
pause
