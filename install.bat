@echo off
REM sticky2jira - Automated Installation Script
REM This script sets up the complete development environment

echo ========================================
echo sticky2jira Installation
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/11] Checking Python version...
python --version

REM Create virtual environment
echo.
echo [2/11] Creating Python virtual environment (.venv)...
if exist .venv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

REM Activate virtual environment
echo.
echo [3/11] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo [4/11] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install pip-tools
echo.
echo [5/11] Installing pip-tools...
pip install pip-tools --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install pip-tools
    pause
    exit /b 1
)

REM Compile requirements.txt
echo.
echo [6/11] Compiling requirements.txt from requirements.in...
pip-compile requirements.in --strip-extras --quiet
if errorlevel 1 (
    echo [ERROR] Failed to compile requirements.txt
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo [7/11] Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo Dependencies installed successfully

REM Download frontend libraries
echo.
echo [8/11] Downloading frontend libraries (Bootstrap, jQuery, DataTables, Socket.IO)...
call download_libs.bat
if errorlevel 1 (
    echo [WARNING] Frontend library download failed
    echo You can run download_libs.bat manually later
)

REM Append to .gitignore
echo.
echo [9/11] Updating .gitignore with project-specific entries...
echo.>> .gitignore
echo # Project-specific>> .gitignore
echo bin/>> .gitignore
echo .encryption_key>> .gitignore
echo uploads/>> .gitignore
echo session.db>> .gitignore
echo jira_templates/>> .gitignore
echo app.log>> .gitignore
echo errors.log>> .gitignore
echo launch.bat>> .gitignore
echo static/libs/>> .gitignore

REM Create launch.bat
echo.
echo [10/11] Creating launch.bat...
(
echo @echo off
echo REM sticky2jira - Launch Script
echo call .venv\Scripts\activate.bat
echo python app.py
echo pause
) > launch.bat

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: launch.bat
echo 2. Browser will open automatically to http://localhost:5000
echo 3. Configure Jira connection in Setup tab
echo.
echo For updates, run: pip-compile requirements.in
echo.
pause
