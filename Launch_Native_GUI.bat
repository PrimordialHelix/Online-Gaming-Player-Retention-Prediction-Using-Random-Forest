@echo off
title ApexRetention AI - Native CustomTkinter GUI
echo ======================================================================
echo   🎮 Launching ApexRetention AI • Native Windows GUI Application
echo   🎨 Framework: CustomTkinter (Emerald & Gold Dark Theme)
echo ======================================================================
cd /d "%~dp0"
python desktop_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FALLBACK] Direct launch via py launcher...
    py -3.11 desktop_gui.py
)
pause

