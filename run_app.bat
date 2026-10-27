@echo off
title ApexRetention AI - Online Gaming Retention Suite
echo ======================================================================
echo   🎮 Starting ApexRetention AI • Gaming Player Retention Platform
echo   🎨 Theme: Emerald Green, Metallic Gold, Obsidian Black, Pure White
echo ======================================================================
cd /d "%~dp0"
streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Streamlit command failed. Trying via Python module...
    python -m streamlit run app.py
)
pause

