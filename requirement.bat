@echo off
title ApexRetention AI - Environment and Requirements Installer
color 0A

echo ==============================================================================
echo   APEX RETENTION AI - ENVIRONMENT AND DEPENDENCY INSTALLER
echo   Theme: Emerald Green, Metallic Gold, Obsidian Black, Crisp White
echo ==============================================================================
echo.

cd /d "%~dp0"

:: -----------------------------------------------------------------------------
:: Step 1: Detect Python Command
:: -----------------------------------------------------------------------------
echo [1/4] Detecting Python installation...

set "PY_CMD="

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

py --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py"
    goto :PYTHON_FOUND
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :PYTHON_FOUND
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :PYTHON_FOUND
)

if exist "%ProgramFiles%\Python311\python.exe" (
    set "PY_CMD=%ProgramFiles%\Python311\python.exe"
    goto :PYTHON_FOUND
)

:PYTHON_NOT_FOUND
color 0C
echo [ERROR] Python was not found in PATH or standard install directories!
echo Please install Python 3.9+ from https://www.python.org/
echo Check the box "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:PYTHON_FOUND
for /f "tokens=*" %%i in ('"%PY_CMD%" --version 2^>^&1') do set "PY_VER=%%i"
echo       Found: %PY_VER%
echo       Executable: "%PY_CMD%"
echo.

:: -----------------------------------------------------------------------------
:: Step 2: Check Pip
:: -----------------------------------------------------------------------------
echo [2/4] Checking pip package manager...
"%PY_CMD%" -m pip install --upgrade pip --quiet >nul 2>&1
echo       Pip check completed.
echo.

:: -----------------------------------------------------------------------------
:: Step 3: Install Required Dependencies
:: -----------------------------------------------------------------------------
echo [3/4] Installing project requirements from requirements.txt...
echo       Packages: pandas, numpy, matplotlib, seaborn, scikit-learn,
echo                 streamlit, plotly, customtkinter, nbformat, ipykernel
echo.

if exist "requirements.txt" (
    "%PY_CMD%" -m pip install -r requirements.txt
) else (
    "%PY_CMD%" -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly customtkinter nbformat ipykernel
)
echo.

:: -----------------------------------------------------------------------------
:: Step 4: Verify Installation of Core Libraries
:: -----------------------------------------------------------------------------
echo [4/4] Verifying installed packages...
"%PY_CMD%" -c "import sys; mods = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'streamlit', 'plotly', 'customtkinter']; missing = []; [print(f'      [OK] {m:15s} verified') for m in mods if __import__(m)]; sys.exit(0)"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==============================================================================
    echo   ALL DEPENDENCIES SUCCESSFULLY INSTALLED AND VERIFIED!
    echo ==============================================================================
    echo.
    echo   You can now launch the application using any of the following modes:
    echo.
    echo   1. Standalone Desktop Window App - Recommended:
    echo      Double-click: Launch_Desktop_App.bat
    echo.
    echo   2. Native Windows CustomTkinter GUI:
    echo      Double-click: Launch_Native_GUI.bat
    echo.
    echo   3. Web Browser Mode - Streamlit:
    echo      Double-click: run_app.bat
    echo.
    echo   4. Jupyter Notebook Research:
    echo      Command: jupyter notebook online_gaming_player_retention.ipynb
    echo.
) else (
    color 0C
    echo.
    echo [WARNING] Some modules could not be verified. Check the messages above.
)

pause

