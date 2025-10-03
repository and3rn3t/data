# PowerShell script for production deployment
# Usage: .\scripts\deploy-prod.ps1 [command]

param(
  [string]$Command = "deploy"
)

$ComposeFile = "docker-compose.prod.yml"
$AppName = "data-science-sandbox"

Write-Host ""
Write-Host "🚀 Data Science Sandbox - Production Deployment (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
  $dockerVersion = docker --version
  Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
}
catch {
  Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
  exit 1
}

# Check if Docker Compose is available
try {
  $composeVersion = docker-compose --version 2>$null
  if (-not $composeVersion) {
    $composeVersion = docker compose version
    $ComposeCmd = "docker compose"
  }
  else {
    $ComposeCmd = "docker-compose"
  }
  Write-Host "✅ Docker Compose found: $composeVersion" -ForegroundColor Green
}
catch {
  Write-Host "❌ Docker Compose is not available" -ForegroundColor Red
  exit 1
}

switch ($Command.ToLower()) {
  "deploy" {
    Write-Host "📦 Building Docker images..." -ForegroundColor Yellow
    & $ComposeCmd -f $ComposeFile build --no-cache

    if ($LASTEXITCODE -ne 0) {
      Write-Host "❌ Build failed" -ForegroundColor Red
      exit 1
    }

    Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
    & $ComposeCmd -f $ComposeFile down --remove-orphans

    Write-Host "🚀 Starting production deployment..." -ForegroundColor Yellow
    & $ComposeCmd -f $ComposeFile up -d

    if ($LASTEXITCODE -eq 0) {
      Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
      Start-Sleep -Seconds 30

      Write-Host "🏥 Checking service health..." -ForegroundColor Yellow
      try {
        $response = Invoke-WebRequest -Uri "http://localhost:8503/_stcore/health" -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
          Write-Host "✅ Enhanced gamification dashboard is healthy" -ForegroundColor Green
        }
        else {
          Write-Host "⚠️  Enhanced dashboard may still be starting..." -ForegroundColor Yellow
        }
      }
      catch {
        Write-Host "⚠️  Enhanced dashboard may still be starting..." -ForegroundColor Yellow
      }

      # Show status
      & $MyInvocation.MyCommand.Path "status"
    }
    else {
      Write-Host "❌ Deployment failed" -ForegroundColor Red
      exit 1
    }
  }

  "stop" {
    Write-Host "🛑 Stopping production deployment..." -ForegroundColor Yellow
    & $ComposeCmd -f $ComposeFile down
    Write-Host "✅ Deployment stopped" -ForegroundColor Green
  }

  "restart" {
    Write-Host "🔄 Restarting production deployment..." -ForegroundColor Yellow
    & $ComposeCmd -f $ComposeFile restart
    Write-Host "✅ Deployment restarted" -ForegroundColor Green
  }

  "logs" {
    Write-Host "📄 Showing logs..." -ForegroundColor Yellow
    & $ComposeCmd -f $ComposeFile logs -f
  }

  "status" {
    Write-Host "📊 Deployment Status:" -ForegroundColor Cyan
    Write-Host ""
    & $ComposeCmd -f $ComposeFile ps
    Write-Host ""

    Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
    Write-Host "🎮 Enhanced Gamification Dashboard: http://localhost:8503" -ForegroundColor White
    Write-Host "📊 Standard Dashboard: http://localhost:8501" -ForegroundColor White
    Write-Host "📱 Modern Dashboard: http://localhost:8502" -ForegroundColor White
    Write-Host "🔬 MLflow Tracking: http://localhost:5000" -ForegroundColor White
    Write-Host "🌐 Nginx Proxy: http://localhost" -ForegroundColor White
    Write-Host ""

    Write-Host "🔗 Service Access via Nginx:" -ForegroundColor Cyan
    Write-Host "🎮 Default (Gamification): http://localhost/" -ForegroundColor White
    Write-Host "📊 Standard: http://standard.localhost/" -ForegroundColor White
    Write-Host "📱 Modern: http://modern.localhost/" -ForegroundColor White
    Write-Host "🔬 MLflow: http://mlflow.localhost/" -ForegroundColor White
    Write-Host ""
    Write-Host "✅ Production deployment is running!" -ForegroundColor Green
  }

  "backup" {
    Write-Host "💾 Creating backup..." -ForegroundColor Yellow
    $BackupDir = "backups\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

    if (Test-Path "data") { Copy-Item "data" "$BackupDir\" -Recurse }
    if (Test-Path "mlruns") { Copy-Item "mlruns" "$BackupDir\" -Recurse }
    if (Test-Path "logs") { Copy-Item "logs" "$BackupDir\" -Recurse }

    Write-Host "✅ Backup created at $BackupDir" -ForegroundColor Green
  }

  default {
    Write-Host "Usage: .\scripts\deploy-prod.ps1 [command]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  deploy  - Deploy the application (default)" -ForegroundColor White
    Write-Host "  stop    - Stop the deployment" -ForegroundColor White
    Write-Host "  restart - Restart the deployment" -ForegroundColor White
    Write-Host "  logs    - Show logs" -ForegroundColor White
    Write-Host "  status  - Show deployment status" -ForegroundColor White
    Write-Host "  backup  - Create backup of data" -ForegroundColor White
    exit 1
  }
}

Write-Host ""
