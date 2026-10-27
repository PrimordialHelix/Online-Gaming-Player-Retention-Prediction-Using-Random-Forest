@echo off
title ApexRetention AI - Desktop Application
echo ======================================================================
echo   🎮 Launching ApexRetention AI • Standalone Desktop Application
echo   🎨 Emerald-Gold Hardware-Accelerated Animation Suite
echo ======================================================================
cd /d "%~dp0"
python desktop_launcher.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FALLBACK] Direct launch via py launcher...
    py -3.11 desktop_launcher.py
)

