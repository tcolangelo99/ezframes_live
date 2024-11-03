@echo off
setlocal

REM Set the target directory to Program Files
set "target_dir=%ProgramFiles%\ezframes_live"
set "internal_folder=%target_dir%\_internal"
set "tar_file_name=%target_dir%\_internal.tar"
set "s3_url=https://ezframesinternal.s3.eu-north-1.amazonaws.com/_internal.tar"
set "download_path=%USERPROFILE%\Desktop\_internal.tar"

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

REM Check if the _internal folder already exists
if exist "%internal_folder%" (
    echo _internal folder already exists. Skipping download and extraction.
) else (
    REM Check if _internal.tar already exists
    if exist "%tar_file_name%" (
        echo _internal.tar already exists. Proceeding to extraction...
    ) else (
        REM Download _internal.tar using curl
        echo Downloading _internal.tar to the Desktop using curl with parallel connections...
        curl --parallel -o "%download_path%" "%s3_url%"

        REM Move the tar file to the target directory
        if exist "%download_path%" (
            echo Download completed successfully.
            move /Y "%download_path%" "%tar_file_name%"
        ) else (
            echo Error: Failed to download _internal.tar. Access is denied or another issue occurred.
            pause
            exit /b
        )
    )

    REM Extract the tar file using tar
    echo Extracting files using tar...
    tar -xf "%tar_file_name%" -C "%target_dir%"
    if %ERRORLEVEL% EQU 0 (
        echo Extraction completed successfully.
        del "%tar_file_name%"
    ) else (
        echo Error: Failed to extract _internal.tar. Ensure tar is available on your system.
        pause
        exit /b
    )
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
