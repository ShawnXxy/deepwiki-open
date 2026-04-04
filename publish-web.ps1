# Azure Web App Deployment for DeepWiki
# Builds and deploys the Docker container to Azure App Service (Web App for Containers)
# Uses Managed Identity for ACR authentication

$ErrorActionPreference = "Stop"

# ============================================
# Configuration - Load from Deployments/config.py
# ============================================

function Get-ConfigValue {
    param([string]$Key)
    $configPath = Join-Path $PSScriptRoot "Deployments/config.py"
    $content = Get-Content $configPath -Raw
    if ($content -match "(?m)^$Key\s*=\s*[`"'](.+?)[`"']") {
        return $matches[1]
    }
    elseif ($content -match "(?m)^$Key\s*=\s*(\S+)") {
        return $matches[1]
    }
    return $null
}

# Load configuration values from config.py
$RESOURCE_GROUP = Get-ConfigValue "resource_group"
$LOCATION = Get-ConfigValue "location"
$APP_SERVICE_NAME = Get-ConfigValue "app_service_name"
$ACR_NAME = Get-ConfigValue "container_registry_name"
$CONTAINER_IMAGE_NAME = Get-ConfigValue "container_image_name"
$CONTAINER_IMAGE_TAG = "latest"  # Always use latest
$MSI_NAME = Get-ConfigValue "dri_copilot_identity_name"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Azure Web App Deployment for DeepWiki" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Configuration (from Deployments/config.py):" -ForegroundColor Yellow
Write-Host "   Resource Group:    $RESOURCE_GROUP"
Write-Host "   Location:          $LOCATION"
Write-Host "   Web App Name:      $APP_SERVICE_NAME"
Write-Host "   ACR Name:          $ACR_NAME"
Write-Host "   Image Name:        $CONTAINER_IMAGE_NAME"
Write-Host "   Image Tag:         $CONTAINER_IMAGE_TAG"
Write-Host "   Managed Identity:  $MSI_NAME"
Write-Host ""

# ============================================
# Pre-flight Checks
# ============================================
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Cyan

# Check if logged in to Azure
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "❌ Not logged in to Azure. Please run 'az login' first." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green
Write-Host "✅ Subscription: $($account.name)" -ForegroundColor Green

# Check if Docker is available
$dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
if (-not $dockerVersion) {
    Write-Host "❌ Docker is not running or not installed." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker version: $dockerVersion" -ForegroundColor Green

# ============================================
# Step 1: Ensure ACR Exists
# ============================================
Write-Host ""
Write-Host "🔧 Step 1: Checking Azure Container Registry..." -ForegroundColor Cyan

$acrExists = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP 2>$null
if ($acrExists) {
    Write-Host "✅ ACR already exists: $ACR_NAME" -ForegroundColor Green
} else {
    Write-Host "   Creating ACR: $ACR_NAME..." -ForegroundColor Yellow
    az acr create `
        --resource-group $RESOURCE_GROUP `
        --name $ACR_NAME `
        --sku Basic `
        --admin-enabled false `
        --location $LOCATION
    Write-Host "✅ ACR created: $ACR_NAME" -ForegroundColor Green
}

$ACR_LOGIN_SERVER = az acr show --name $ACR_NAME --query loginServer -o tsv
Write-Host "   ACR Login Server: $ACR_LOGIN_SERVER" -ForegroundColor Gray

# ============================================
# Step 2: Grant ACR Pull Permission to Managed Identity
# ============================================
Write-Host ""
Write-Host "🔐 Step 2: Ensuring Managed Identity has ACR Pull permission..." -ForegroundColor Cyan

$MSI_PRINCIPAL_ID = az identity show `
    --name $MSI_NAME `
    --resource-group $RESOURCE_GROUP `
    --query principalId -o tsv 2>$null

if (-not $MSI_PRINCIPAL_ID) {
    Write-Host "❌ Managed Identity '$MSI_NAME' not found in resource group '$RESOURCE_GROUP'" -ForegroundColor Red
    Write-Host "   Please run the deployment notebook first to create the managed identity." -ForegroundColor Yellow
    exit 1
}

$ACR_RESOURCE_ID = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query id -o tsv

# Check if role assignment already exists
$existingRole = az role assignment list `
    --assignee $MSI_PRINCIPAL_ID `
    --scope $ACR_RESOURCE_ID `
    --role "AcrPull" `
    --query "[0]" 2>$null | ConvertFrom-Json

if ($existingRole) {
    Write-Host "✅ AcrPull role already assigned to Managed Identity" -ForegroundColor Green
} else {
    Write-Host "   Assigning AcrPull role to Managed Identity..." -ForegroundColor Yellow
    az role assignment create `
        --assignee $MSI_PRINCIPAL_ID `
        --scope $ACR_RESOURCE_ID `
        --role "AcrPull" | Out-Null
    Write-Host "✅ AcrPull role assigned" -ForegroundColor Green
}

# ============================================
# Step 3: Create Cloud Config (enable blob storage, AI Search, and App Insights)
# ============================================
Write-Host ""
Write-Host "📝 Step 3: Creating .cloud config (enabling cloud services)..." -ForegroundColor Cyan

$cloudConfigDir = Join-Path $PSScriptRoot "backend/config/.cloud"
if (-not (Test-Path $cloudConfigDir)) {
    New-Item -ItemType Directory -Path $cloudConfigDir -Force | Out-Null
}

# Read original infra.json and ensure cloud settings are enabled
$infraPath = Join-Path $PSScriptRoot "backend/config/infra.json"
$cloudInfraPath = Join-Path $cloudConfigDir "infra.json"

$infra = Get-Content $infraPath | ConvertFrom-Json
$infra.azure_blob_storage.enabled = $true
$infra.azure_ai_search.enabled = $true
$infra.azure_application_insights.enabled = $true
$infra | ConvertTo-Json -Depth 10 | Set-Content $cloudInfraPath

# Copy other config files to .cloud
$configFiles = @("excluded.json", "included.json", "generator.json", "lang.json", "embedder.json")
foreach ($configFile in $configFiles) {
    $sourcePath = Join-Path $PSScriptRoot "backend/config/$configFile"
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $cloudConfigDir -Force
    }
}

Write-Host "✅ Cloud config created (blob, AI Search, App Insights enabled)" -ForegroundColor Green

# ============================================
# Step 4: Build and Push Container Image
# ============================================
Write-Host ""
Write-Host "🐳 Step 4: Building and pushing container image..." -ForegroundColor Cyan

# Login to ACR
Write-Host "   Logging into ACR..." -ForegroundColor Gray
az acr login --name $ACR_NAME

$IMAGE_TAG = "${ACR_LOGIN_SERVER}/${CONTAINER_IMAGE_NAME}:${CONTAINER_IMAGE_TAG}"

# Build locally with --no-cache to ensure latest code is included
Write-Host "   Building Docker image locally (--no-cache for fresh build)..." -ForegroundColor Gray
docker build --no-cache -t $IMAGE_TAG -f Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

# Push to ACR
Write-Host "   Pushing to ACR..." -ForegroundColor Gray
docker push $IMAGE_TAG

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Image pushed to: $IMAGE_TAG" -ForegroundColor Green

# ============================================
# Step 5: Verify Web App Exists
# ============================================
Write-Host ""
Write-Host "🌐 Step 5: Checking Web App..." -ForegroundColor Cyan

$webAppExists = az webapp show --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP 2>$null
if (-not $webAppExists) {
    Write-Host "❌ Web App '$APP_SERVICE_NAME' not found in resource group '$RESOURCE_GROUP'" -ForegroundColor Red
    Write-Host "   Please run the deployment notebook (deploy_required.ipynb) first to create the Web App." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   The Web App template has been updated. Run the WEB deployment cell in the notebook" -ForegroundColor Yellow
    Write-Host "   to create a Web App configured for container deployment." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Web App exists: $APP_SERVICE_NAME" -ForegroundColor Green

# ============================================
# Step 6: Update Web App Container Settings
# ============================================
Write-Host ""
Write-Host "🔄 Step 6: Updating Web App container settings..." -ForegroundColor Cyan

# Get MSI Client ID for AZURE_CLIENT_ID
$MSI_CLIENT_ID = az identity show `
    --name $MSI_NAME `
    --resource-group $RESOURCE_GROUP `
    --query clientId -o tsv

# Update container settings
Write-Host "   Configuring container image: $IMAGE_TAG" -ForegroundColor Gray

az webapp config container set `
    --name $APP_SERVICE_NAME `
    --resource-group $RESOURCE_GROUP `
    --container-image-name $IMAGE_TAG `
    --container-registry-url "https://$ACR_LOGIN_SERVER"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to update container settings!" -ForegroundColor Red
    exit 1
}

# Enable managed identity for ACR pull
Write-Host "   Enabling Managed Identity for ACR pull..." -ForegroundColor Gray

# Use az resource update which handles JSON properly
$SUBSCRIPTION_ID = (az account show --query id -o tsv)
$webConfigResourceId = "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$APP_SERVICE_NAME/config/web"

az resource update `
    --ids $webConfigResourceId `
    --set properties.acrUseManagedIdentityCreds=true properties.acrUserManagedIdentityID=$MSI_CLIENT_ID `
    --output none

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Warning: Failed to set ACR managed identity config via az resource update" -ForegroundColor Yellow
}

# Update app settings
Write-Host "   Updating app settings..." -ForegroundColor Gray
az webapp config appsettings set `
    --name $APP_SERVICE_NAME `
    --resource-group $RESOURCE_GROUP `
    --settings `
        AZURE_CLIENT_ID=$MSI_CLIENT_ID `
        DEEPWIKI_CONFIG_DIR=backend/config/.cloud `
        WEBSITES_ENABLE_APP_SERVICE_STORAGE=false `
        DOCKER_REGISTRY_SERVER_URL="https://$ACR_LOGIN_SERVER" `
        WEBSITES_PORT=3000 `
        FASTAPI_PORT=8001 `
        NODE_ENV=production `
        LOG_LEVEL=INFO

Write-Host "✅ Container settings updated" -ForegroundColor Green

# ============================================
# Step 7: Restart Web App
# ============================================
Write-Host ""
Write-Host "🔁 Step 7: Restarting Web App to apply changes..." -ForegroundColor Cyan

az webapp restart --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP

Write-Host "✅ Web App restarted" -ForegroundColor Green

# ============================================
# Step 8: Get Deployment Info
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Deployment Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

$webAppUrl = az webapp show --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP --query defaultHostName -o tsv
Write-Host ""
Write-Host "🌐 Web App URL: https://$webAppUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Deployment Summary:" -ForegroundColor Yellow
Write-Host "   - Container Image: $IMAGE_TAG"
Write-Host "   - ACR: $ACR_LOGIN_SERVER"
Write-Host "   - Managed Identity: $MSI_NAME"
Write-Host ""
Write-Host "⏳ Note: The container may take 2-5 minutes to fully start." -ForegroundColor Yellow
Write-Host "   You can monitor startup logs with:" -ForegroundColor Gray
Write-Host "   az webapp log tail --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP" -ForegroundColor Gray
Write-Host ""
Write-Host "🔍 To check container health:" -ForegroundColor Yellow
Write-Host "   az webapp show --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP --query state -o tsv" -ForegroundColor Gray
Write-Host ""
