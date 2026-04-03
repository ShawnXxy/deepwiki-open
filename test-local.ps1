# Local Docker Testing Script for CodeWiki v2
# Builds and runs the container locally for testing.
#
# Architecture (v2):
#   - Next.js on :3000 (frontend viewer — reads wiki cache from disk)
#   - FastAPI on :8001 (optional — Ask/Chat only)
#   - No nginx needed (frontend is self-contained)
#
# USAGE:
#   .\test-local.ps1                           # Reads API key from backend/.env
#   .\test-local.ps1 -ApiKey "your-api-key"    # Uses provided API key

param(
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"

# ============================================
# Load .env file
# ============================================
$envFile = Join-Path $PSScriptRoot "backend/.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*?)\s*=\s*(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            if (-not [string]::IsNullOrEmpty($value) -and $value -ne "your-api-key-here") {
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
            }
        }
    }
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CodeWiki v2 - Local Docker Testing" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================
# Configuration
# ============================================
$IMAGE_NAME = "codewiki-local-v2"
$CONTAINER_NAME = "codewiki-v2"
$BACKEND_PORT = 8001
$FRONTEND_PORT = 3000

# Check if API key is provided via parameter or environment variable (including from .env)
if ([string]::IsNullOrEmpty($ApiKey)) {
    $ApiKey = $env:AZURE_OPENAI_API_KEY
}

$usingApiKey = -not [string]::IsNullOrEmpty($ApiKey)
if ($usingApiKey) {
    Write-Host "🔑 Using API Key authentication (from .env or parameter)" -ForegroundColor Green
} else {
    Write-Host "⚠️  No API key found. Please configure .env file:" -ForegroundColor Yellow
    Write-Host "   1. Edit .env file in project root" -ForegroundColor DarkGray
    Write-Host "   2. Set AZURE_OPENAI_API_KEY=your-key-here" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "   Or run with: .\test-local.ps1 -ApiKey `"your-key`"" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "🔐 Falling back to Azure CLI authentication..." -ForegroundColor Yellow
}

# ============================================
# Check Docker
# ============================================
Write-Host "🔍 Checking Docker..." -ForegroundColor Yellow
$dockerVersion = docker --version 2>$null
if (-not $dockerVersion) {
    Write-Host "❌ Docker is not installed or not running." -ForegroundColor Red
    Write-Host "   Please install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor White
    exit 1
}
Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green

# ============================================
# Stop existing container
# ============================================
Write-Host ""
Write-Host "🛑 Stopping existing container (if any)..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME 2>$null
docker rm $CONTAINER_NAME 2>$null
Write-Host "✅ Cleaned up" -ForegroundColor Green

# ============================================
# Build Image
# ============================================
Write-Host ""
Write-Host "🐳 Building Docker image..." -ForegroundColor Yellow
Write-Host "   This may take 5-10 minutes on first build..." -ForegroundColor DarkGray

$buildStart = Get-Date
docker build -t $IMAGE_NAME -f Dockerfile .
$buildDuration = (Get-Date) - $buildStart

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Image built in $([math]::Round($buildDuration.TotalMinutes, 1)) minutes" -ForegroundColor Green

# ============================================
# Create local config (disable blob storage and App Insights)
# ============================================
# Config Strategy:
#   - Local terminal: reads backend/config/ directly (blob/appinsights per infra.json)
#   - Local Docker: uses backend/config/.local/ (blob/appinsights DISABLED)
#   - Azure Cloud: uses backend/config/.cloud/ (blob/appinsights ENABLED)
Write-Host ""
Write-Host "📝 Creating .local config (disabling blob storage and App Insights)..." -ForegroundColor Yellow

$localConfigDir = Join-Path $PSScriptRoot "backend/config/.local"
if (-not (Test-Path $localConfigDir)) {
    New-Item -ItemType Directory -Path $localConfigDir -Force | Out-Null
}

# Read original infra.json and modify for local Docker testing
$infraPath = Join-Path $PSScriptRoot "backend/config/infra.json"
$localInfraPath = Join-Path $localConfigDir "infra.json"

$infra = Get-Content $infraPath | ConvertFrom-Json
$infra.azure_blob_storage.enabled = $false
$infra.azure_application_insights.enabled = $false
$infra | ConvertTo-Json -Depth 10 | Set-Content $localInfraPath

# Copy other config files to .local
$configFiles = @("excluded.json", "included.json", "lang.json", "embedder.json")
foreach ($configFile in $configFiles) {
    $sourcePath = Join-Path $PSScriptRoot "backend/config/$configFile"
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $localConfigDir -Force
    }
}

Write-Host "✅ Local config created with blob storage and App Insights disabled" -ForegroundColor Green

# ============================================
# Run Container
# ============================================
Write-Host ""
Write-Host "🚀 Starting container..." -ForegroundColor Yellow

# Build docker run command
# Note: 8GB memory limit for large repository embedding (5000+ files)
$dockerArgs = @(
    "run", "-d",
    "--name", $CONTAINER_NAME,
    "--memory", "8g",
    "-p", "${BACKEND_PORT}:${BACKEND_PORT}",
    "-p", "${FRONTEND_PORT}:3000",
    "-e", "PORT=$BACKEND_PORT",
    "-e", "NODE_ENV=production",
    "-e", "LOG_LEVEL=DEBUG",
    "-e", "SERVER_BASE_URL=http://localhost:$BACKEND_PORT",
    "-v", "$HOME/.adalflow:/root/.adalflow",
    "-v", "$PWD/backend/config/.local:/app/backend/config:ro"
)

# Add API key if provided, otherwise mount Azure CLI volume
if ($usingApiKey) {
    $dockerArgs += @("-e", "AZURE_OPENAI_API_KEY=$ApiKey")
    Write-Host "   ✅ API key will be passed to container" -ForegroundColor DarkGray
}

# Pass REPO_ACCESS_TOKEN if set (for private repo cloning)
$pat = $env:REPO_ACCESS_TOKEN
if (-not [string]::IsNullOrEmpty($pat)) {
    $dockerArgs += @("-e", "REPO_ACCESS_TOKEN=$pat")
    Write-Host "   ✅ REPO_ACCESS_TOKEN will be passed to container" -ForegroundColor DarkGray
}

# Mount backend/.env into container (it's excluded by .dockerignore)
$backendEnv = Join-Path $PSScriptRoot "backend/.env"
if (Test-Path $backendEnv) {
    $dockerArgs += @("-v", "${backendEnv}:/app/backend/.env:ro")
    Write-Host "   ✅ backend/.env mounted into container" -ForegroundColor DarkGray
}

if (-not $usingApiKey -and -not (Test-Path $backendEnv)) {
    $dockerArgs += @("-v", "codewiki-azure-cli:/root/.azure")
    Write-Host "   📁 Mounting Azure CLI credentials volume" -ForegroundColor DarkGray
}

$dockerArgs += $IMAGE_NAME

# Run the container
docker @dockerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start container!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Container started" -ForegroundColor Green

# ============================================
# Wait for startup
# ============================================
Write-Host ""
Write-Host "⏳ Waiting for application to start..." -ForegroundColor Yellow

$maxAttempts = 30
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    Start-Sleep -Seconds 2
    $attempt++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$BACKEND_PORT/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
        }
    } catch {
        Write-Host "   Waiting... ($attempt/$maxAttempts)" -ForegroundColor DarkGray
    }
}

if (-not $ready) {
    Write-Host "⚠️  Application may still be starting. Check logs with:" -ForegroundColor Yellow
    Write-Host "   docker logs $CONTAINER_NAME" -ForegroundColor DarkGray
} else {
    Write-Host "✅ Application is ready!" -ForegroundColor Green
}

# ============================================
# Check Azure authentication
# ============================================
Write-Host ""
Write-Host "🔐 Checking Azure authentication..." -ForegroundColor Yellow

if ($usingApiKey) {
    Write-Host "✅ Using API key authentication - ready to use Azure OpenAI" -ForegroundColor Green
} else {
    $azureAccount = docker exec $CONTAINER_NAME az account show --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Azure CLI not logged in inside container." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   Option 1: Use API key (recommended for local dev)" -ForegroundColor Cyan
        Write-Host "   Stop this container and re-run with:" -ForegroundColor DarkGray
        Write-Host "   .\test-local.ps1 -ApiKey `"your-api-key`"" -ForegroundColor White
        Write-Host ""
        Write-Host "   To get your API key:" -ForegroundColor DarkGray
        Write-Host "   Azure Portal -> Azure OpenAI -> Keys and Endpoint -> Key 1" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "   Option 2: Try Azure CLI login" -ForegroundColor Cyan
        Write-Host "   docker exec -it $CONTAINER_NAME az login --use-device-code" -ForegroundColor White
        Write-Host "   (May be blocked by corporate Conditional Access policies)" -ForegroundColor DarkGray
        Write-Host ""
    } else {
        Write-Host "✅ Azure CLI logged in" -ForegroundColor Green
    }
}

# ============================================
# Summary
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  🎉 Local Testing Ready!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
Write-Host "  Frontend:     http://localhost:$FRONTEND_PORT" -ForegroundColor White
Write-Host "  Backend API:  http://localhost:$BACKEND_PORT" -ForegroundColor White
Write-Host "  Health Check: http://localhost:$BACKEND_PORT/health" -ForegroundColor White
Write-Host ""
Write-Host "Commands:" -ForegroundColor Cyan
Write-Host "  View logs:    docker logs -f $CONTAINER_NAME" -ForegroundColor DarkGray
Write-Host "  Stop:         docker stop $CONTAINER_NAME" -ForegroundColor DarkGray
Write-Host "  Shell:        docker exec -it $CONTAINER_NAME bash" -ForegroundColor DarkGray
if (-not $usingApiKey) {
    Write-Host "  Azure login:  docker exec -it $CONTAINER_NAME az login --use-device-code" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:$FRONTEND_PORT"
