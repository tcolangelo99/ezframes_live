@echo off
setlocal

REM Set the target directory to Program Files
set "target_dir=%ProgramFiles%\ezframes_live"

REM Function to check if Git is installed
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Installing Git...

    REM Inform the user
    echo This script will download and install Git for Windows.
    echo Press any key to continue or close this window to cancel.
    pause >nul

    REM Set a temporary directory for downloading the installer
    set "temp_dir=%TEMP%\git_installer"
    mkdir "%temp_dir%"
    cd /d "%temp_dir%"

    REM Download the Git for Windows installer
    echo Downloading Git installer...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.1/Git-2.42.0-64-bit.exe -OutFile Git-Installer.exe"

    REM Install Git silently
    echo Installing Git silently...
    start /wait "" Git-Installer.exe /VERYSILENT /NORESTART /SP- /NOICONS

    REM Clean up the installer
    del Git-Installer.exe
    cd /d "%~dp0"
    rmdir /s /q "%temp_dir%"

    REM Update the PATH variable to include Git (this change is temporary for this script)
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
)

REM Check for admin privileges to write to Program Files
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo This script requires administrative privileges to write to Program Files.
    echo Please re-run this script as an administrator.
    pause
    exit /b
)

REM Check if the Visual Studio C++ Redistributable is installed
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Visual Studio C++ Redistributable is not installed. Installing...

    REM Set a temporary directory for downloading the installer
    set "temp_dir=%TEMP%\vcredist_installer"
    mkdir "%temp_dir%"
    cd /d "%temp_dir%"

    REM Download the Visual Studio C++ Redistributable installer
    echo Downloading Visual Studio C++ Redistributable installer...
    powershell -Command "Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile vc_redist.x64.exe"

    REM Install Visual Studio C++ Redistributable silently
    echo Installing Visual Studio C++ Redistributable silently...
    start /wait "" vc_redist.x64.exe /quiet /norestart

    REM Clean up the installer
    del vc_redist.x64.exe
    cd /d "%~dp0"
    rmdir /s /q "%temp_dir%"
)

REM Proceed with cloning or pulling the repository
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

REM Run the application after pulling or cloning
echo Running launcher.exe...
start "" "%target_dir%\launcher.exe"

pause
