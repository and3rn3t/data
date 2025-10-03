@echo off
REM Windows batch script for production deployment

setlocal enabledelayedexpansion

set "COMPOSE_FILE=docker-compose.prod.yml"
set "APP_NAME=data-science-sandbox"

echo.
echo 🚀 Data Science Sandbox - Production Deployment (Windows)
echo ============================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not in PATH
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker Compose is not available
        exit /b 1
    )
    set "COMPOSE_CMD=docker compose"
) else (
    set "COMPOSE_CMD=docker-compose"
)

if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="status" goto status
if "%1"=="backup" goto backup

:deploy
echo 📦 Building Docker images...
%COMPOSE_CMD% -f %COMPOSE_FILE% build --no-cache

echo 🛑 Stopping existing containers...
%COMPOSE_CMD% -f %COMPOSE_FILE% down --remove-orphans

echo 🚀 Starting production deployment...
%COMPOSE_CMD% -f %COMPOSE_FILE% up -d

echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

echo 🏥 Checking service health...
curl -f -s http://localhost:8503/_stcore/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Enhanced dashboard may still be starting...
) else (
    echo ✅ Enhanced gamification dashboard is healthy
)

goto status

:stop
echo 🛑 Stopping production deployment...
%COMPOSE_CMD% -f %COMPOSE_FILE% down
echo ✅ Deployment stopped
goto end

:restart
echo 🔄 Restarting production deployment...
%COMPOSE_CMD% -f %COMPOSE_FILE% restart
echo ✅ Deployment restarted
goto end

:logs
echo 📄 Showing logs...
if "%2"=="" (
    %COMPOSE_CMD% -f %COMPOSE_FILE% logs -f
) else (
    %COMPOSE_CMD% -f %COMPOSE_FILE% logs -f %2
)
goto end

:status
echo 📊 Deployment Status:
echo.
%COMPOSE_CMD% -f %COMPOSE_FILE% ps
echo.
echo 🌐 Service URLs:
echo 🎮 Enhanced Gamification Dashboard: http://localhost:8503
echo 📊 Standard Dashboard: http://localhost:8501
echo 📱 Modern Dashboard: http://localhost:8502
echo 🔬 MLflow Tracking: http://localhost:5000
echo 🌐 Nginx Proxy: http://localhost
echo.
echo 🔗 Service Access via Nginx:
echo 🎮 Default (Gamification): http://localhost/
echo 📊 Standard: http://standard.localhost/
echo 📱 Modern: http://modern.localhost/
echo 🔬 MLflow: http://mlflow.localhost/
echo.
echo ✅ Production deployment is running!
goto end

:backup
echo 💾 Creating backup...
set "BACKUP_DIR=backups\%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "BACKUP_DIR=%BACKUP_DIR: =0%"
mkdir "%BACKUP_DIR%" 2>nul
if exist "data" xcopy "data" "%BACKUP_DIR%\data\" /s /e /i /q
if exist "mlruns" xcopy "mlruns" "%BACKUP_DIR%\mlruns\" /s /e /i /q
if exist "logs" xcopy "logs" "%BACKUP_DIR%\logs\" /s /e /i /q
echo ✅ Backup created at %BACKUP_DIR%
goto end

:help
echo Usage: %0 [command]
echo.
echo Commands:
echo   (no args) - Deploy the application (default)
echo   stop      - Stop the deployment
echo   restart   - Restart the deployment
echo   logs      - Show logs
echo   status    - Show deployment status
echo   backup    - Create backup of data
echo   help      - Show this help
goto end

:end
echo.
