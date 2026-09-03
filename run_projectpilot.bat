@echo off
title PROJECTPILOT AI Launcher
echo ==================================================
echo         STARTING PROJECTPILOT AI PLATFORM         
echo ==================================================
echo.

cd /d "%~dp0"

echo [1/2] Launching FastAPI REST Backend API (Port 8000)...
start /b python backend_server.py

timeout /t 3 >nul

echo [2/2] Launching Streamlit Web App (Port 8501)...
start http://localhost:8501
python -m streamlit run app.py --server.port 8501

pause
