@echo off
title Vehicle Exterior Damage Detection AI
echo =====================================================================
echo       VEHICLE EXTERIOR DAMAGE DETECTION - AI INSPECTOR
echo       Model: Prototypical Mask R-CNN (ResNet-50-FPN)
echo =====================================================================
echo.

if not exist "latest_model.pth" (
    echo [ERROR] 'latest_model.pth' was not found in the current folder!
    echo Please make sure latest_model.pth is present.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Starting Web Application with %PYTHON_EXE%...
echo Web UI will launch at http://127.0.0.1:7860
echo.
"%PYTHON_EXE%" app.py --port 7860 --inbrowser

pause
