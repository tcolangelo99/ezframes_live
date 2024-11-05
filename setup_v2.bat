@echo off
setlocal

REM Set the target directory to Program Files
set "target_dir=%ProgramFiles%\ezframes_live"

REM Check if Git is installed
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Please install Git and try again.
    pause
    exit /b
)

REM Clone or update the GitHub repository
if exist "%target_dir%\.git" (
    echo Repository already exists, pulling latest updates...
    cd "%target_dir%"
    git pull
) else (
    echo Cloning the repository to Program Files...
    cd "%ProgramFiles%"
    git clone https://github.com/tcolangelo99/ezframes_live.git
    cd ezframes_live
)

REM Check if the Visual Studio C++ Redistributable is installed
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Visual Studio C++ Redistributable is not installed. Installing...

    REM Set a temporary directory for downloading the installer
    set "temp_dir=%TEMP%\vcredist_installer"
    mkdir "%temp_dir%"
    cd /d "%temp_dir%"

    REM Download the Visual Studio C++ Redistributable installer using curl
    echo Downloading Visual Studio C++ Redistributable installer...
    curl -o "vc_redist.x64.exe" "https://aka.ms/vs/17/release/vc_redist.x64.exe"

    REM Install Visual Studio C++ Redistributable silently
    echo Installing Visual Studio C++ Redistributable silently...
    start /wait "" vc_redist.x64.exe /quiet /norestart

    REM Clean up the installer
    del vc_redist.x64.exe
    cd /d "%~dp0"
    rmdir /s /q "%temp_dir%"
)

REM Run the application after pulling or downloading everything
echo Running launcher.exe...
start "" "%target_dir%\launcher.exe"

pause
