# Azure Container Apps Deployment for CodeWiki
# Using the unified Dockerfile with Managed Identity Authentication

$ErrorActionPreference = "Stop"

# ============================================
# Configuration - Update these values
# ============================================
$RESOURCE_GROUP = "RG-ORCAS-DEEPWIKI"
$LOCATION = "eastasia"
$ENVIRONMENT_NAME = "codewiki-env"
$ACR_NAME = "acrorcascodewiki"  # Must be globally unique, lowercase, alphanumeric only
$APP_NAME = "orcascodewiki"

# Managed Identity (from your infra.json)
$MSI_NAME = "mid-orcas-deepwiki"

# Storage account name from infra.json
$STORAGE_ACCOUNT_NAME = "bloborcasdeepwiki"

# Log Analytics Workspace (existing workspace to use for Container Apps Environment)
# Leave empty to let Azure auto-create a new workspace
$LOG_ANALYTICS_WORKSPACE_NAME = "log-orcas-deepwiki-ea"

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

$SUBSCRIPTION_ID = $account.id

# ============================================
# Step 1: Create Azure Container Registry
# ============================================
Write-Host ""
Write-Host "🔧 Step 1: Creating Azure Container Registry..." -ForegroundColor Cyan

$acrExists = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP 2>$null
if ($acrExists) {
    Write-Host "✅ ACR already exists: $ACR_NAME" -ForegroundColor Green
} else {
    az acr create `
        --resource-group $RESOURCE_GROUP `
        --name $ACR_NAME `
        --sku Basic `
        --admin-enabled true `
        --location $LOCATION
    Write-Host "✅ ACR created: $ACR_NAME" -ForegroundColor Green
}

$ACR_LOGIN_SERVER = az acr show --name $ACR_NAME --query loginServer -o tsv

# ============================================
# Step 2: Create cloud config (enable blob storage and App Insights)
# ============================================
# Config Strategy:
#   - Local terminal: reads backend/config/ directly (blob/appinsights per infra.json)
#   - Local Docker: uses backend/config/.local/ (blob/appinsights DISABLED)
#   - Azure Cloud: uses backend/config/.cloud/ (blob/appinsights ENABLED)
Write-Host ""
Write-Host "📝 Step 2: Creating .cloud config (ensuring blob storage and App Insights enabled)..." -ForegroundColor Cyan

$cloudConfigDir = Join-Path $PSScriptRoot "backend/config/.cloud"
if (-not (Test-Path $cloudConfigDir)) {
    New-Item -ItemType Directory -Path $cloudConfigDir -Force | Out-Null
}

# Read original infra.json and ensure cloud settings are enabled
$infraPath = Join-Path $PSScriptRoot "backend/config/infra.json"
$cloudInfraPath = Join-Path $cloudConfigDir "infra.json"

$infra = Get-Content $infraPath | ConvertFrom-Json
$infra.azure_blob_storage.enabled = $true
$infra.azure_application_insights.enabled = $true
$infra | ConvertTo-Json -Depth 10 | Set-Content $cloudInfraPath

# Copy other config files to .cloud
$configFiles = @("repo.json", "generator.json", "lang.json", "embedder.json")
foreach ($configFile in $configFiles) {
    $sourcePath = Join-Path $PSScriptRoot "backend/config/$configFile"
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $cloudConfigDir -Force
    }
}

Write-Host "✅ Cloud config created with blob storage and App Insights enabled" -ForegroundColor Green

# ============================================
# Step 3: Build and Push Container Image (Local Docker)
# ============================================
Write-Host ""
Write-Host "🐳 Step 3: Building and pushing container image..." -ForegroundColor Cyan
Write-Host "   Using local Docker build (faster with .dockerignore)..." -ForegroundColor Yellow

# Login to ACR
Write-Host "   Logging into ACR..." -ForegroundColor Gray
az acr login --name $ACR_NAME

# Build locally with --no-cache to ensure latest code is included
# Docker layer caching can cause stale code to be deployed if layers haven't changed
Write-Host "   Building Docker image locally (--no-cache for fresh build)..." -ForegroundColor Gray
docker build --no-cache -t "${ACR_LOGIN_SERVER}/codewiki:latest" -f Dockerfile .

# Push to ACR
Write-Host "   Pushing to ACR..." -ForegroundColor Gray
docker push "${ACR_LOGIN_SERVER}/codewiki:latest"

Write-Host "✅ Image pushed to: $ACR_LOGIN_SERVER/codewiki:latest" -ForegroundColor Green

# ============================================
# Step 4: Create Container Apps Environment
# ============================================
Write-Host ""
Write-Host "🌐 Step 4: Creating Container Apps Environment..." -ForegroundColor Cyan

$envExists = az containerapp env show --name $ENVIRONMENT_NAME --resource-group $RESOURCE_GROUP 2>$null
if ($envExists) {
    Write-Host "✅ Environment already exists: $ENVIRONMENT_NAME" -ForegroundColor Green
} else {
    # Check if using existing Log Analytics workspace
    if ($LOG_ANALYTICS_WORKSPACE_NAME) {
        Write-Host "   Using existing Log Analytics workspace: $LOG_ANALYTICS_WORKSPACE_NAME" -ForegroundColor Yellow
        
        # Get the Log Analytics workspace resource ID and customer ID
        $LOG_ANALYTICS_WORKSPACE_ID = az monitor log-analytics workspace show `
            --workspace-name $LOG_ANALYTICS_WORKSPACE_NAME `
            --resource-group $RESOURCE_GROUP `
            --query customerId -o tsv
        
        $LOG_ANALYTICS_WORKSPACE_KEY = az monitor log-analytics workspace get-shared-keys `
            --workspace-name $LOG_ANALYTICS_WORKSPACE_NAME `
            --resource-group $RESOURCE_GROUP `
            --query primarySharedKey -o tsv
        
        if ($LOG_ANALYTICS_WORKSPACE_ID -and $LOG_ANALYTICS_WORKSPACE_KEY) {
            az containerapp env create `
                --name $ENVIRONMENT_NAME `
                --resource-group $RESOURCE_GROUP `
                --location $LOCATION `
                --logs-workspace-id $LOG_ANALYTICS_WORKSPACE_ID `
                --logs-workspace-key $LOG_ANALYTICS_WORKSPACE_KEY
            Write-Host "✅ Environment created with existing Log Analytics workspace" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Could not retrieve Log Analytics workspace details. Creating environment with auto-generated workspace..." -ForegroundColor Yellow
            az containerapp env create `
                --name $ENVIRONMENT_NAME `
                --resource-group $RESOURCE_GROUP `
                --location $LOCATION
            Write-Host "✅ Environment created with auto-generated Log Analytics workspace" -ForegroundColor Green
        }
    } else {
        Write-Host "   No existing workspace specified - Azure will auto-create one" -ForegroundColor Yellow
        az containerapp env create `
            --name $ENVIRONMENT_NAME `
            --resource-group $RESOURCE_GROUP `
            --location $LOCATION
        Write-Host "✅ Environment created: $ENVIRONMENT_NAME" -ForegroundColor Green
    }
}

# ============================================
# Step 4b: Add D4 Dedicated Workload Profile
# ============================================
Write-Host ""
Write-Host "🔧 Step 4b: Adding D4 Dedicated Workload Profile..." -ForegroundColor Cyan
Write-Host "   D4 profile provides 4 vCPU / 16GB RAM for large repository embedding" -ForegroundColor DarkGray

$profileExists = az containerapp env workload-profile show `
    --name $ENVIRONMENT_NAME `
    --resource-group $RESOURCE_GROUP `
    --workload-profile-name "D4" 2>$null

if ($profileExists) {
    Write-Host "✅ D4 workload profile already exists" -ForegroundColor Green
} else {
    az containerapp env workload-profile add `
        --name $ENVIRONMENT_NAME `
        --resource-group $RESOURCE_GROUP `
        --workload-profile-name "D4" `
        --workload-profile-type "D4" `
        --min-nodes 0 `
        --max-nodes 1
    Write-Host "✅ D4 workload profile added" -ForegroundColor Green
}

# ============================================
# Step 5: Deploy Container App
# ============================================
Write-Host ""
Write-Host "🚀 Step 5: Deploying Container App..." -ForegroundColor Cyan

# Get ACR credentials
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv

# Generate deployment timestamp to force new revision even if image digest is same
$DEPLOY_TIMESTAMP = Get-Date -Format 'yyyyMMddHHmmss'

# Check if app exists
$appExists = az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP 2>$null
if ($appExists) {
    Write-Host "   Updating existing app (timestamp: $DEPLOY_TIMESTAMP)..." -ForegroundColor Yellow
    # Include DEPLOY_TIMESTAMP env var to force a new revision
    # This ensures Container Apps creates a new revision even if image digest hasn't changed
    az containerapp update `
        --name $APP_NAME `
        --resource-group $RESOURCE_GROUP `
        --image "$ACR_LOGIN_SERVER/codewiki:latest" `
        --set-env-vars "DEPLOY_TIMESTAMP=$DEPLOY_TIMESTAMP"
} else {
    Write-Host "   Creating new app..." -ForegroundColor Yellow
    # Using D4 dedicated workload profile with 4 CPU / 16GB RAM
    # Required for large repository embedding (5000+ files)
    # Consumption tier max is 8GB which causes OOM for large repos
    az containerapp create `
        --name $APP_NAME `
        --resource-group $RESOURCE_GROUP `
        --environment $ENVIRONMENT_NAME `
        --image "$ACR_LOGIN_SERVER/codewiki:latest" `
        --workload-profile-name "D4" `
        --target-port 3000 `
        --ingress external `
        --min-replicas 1 `
        --max-replicas 5 `
        --cpu 4.0 `
        --memory 16.0Gi `
        --registry-server $ACR_LOGIN_SERVER `
        --registry-username $ACR_USERNAME `
        --registry-password $ACR_PASSWORD `
        --env-vars "PORT=8001" "NODE_ENV=production" "LOG_LEVEL=INFO" "DEPLOY_TIMESTAMP=$DEPLOY_TIMESTAMP"
}

# Verify deployment - check that a new revision was created
Write-Host "   Verifying deployment..." -ForegroundColor Gray
$latestRevision = az containerapp show `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --query "properties.latestRevisionName" -o tsv

$revisionInfo = az containerapp revision show `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --revision $latestRevision `
    --query "{name:name, created:properties.createdTime, active:properties.active, replicas:properties.replicas}" -o json | ConvertFrom-Json

Write-Host "   Latest revision: $($revisionInfo.name)" -ForegroundColor Gray
Write-Host "   Created: $($revisionInfo.created)" -ForegroundColor Gray
Write-Host "   Active: $($revisionInfo.active)" -ForegroundColor Gray
Write-Host "   Replicas: $($revisionInfo.replicas)" -ForegroundColor Gray

Write-Host "✅ Container App deployed" -ForegroundColor Green

# ============================================
# Step 6: Configure Managed Identity
# ============================================
Write-Host ""
Write-Host "🔐 Step 6: Configuring Managed Identity..." -ForegroundColor Cyan

# Enable system-assigned managed identity
az containerapp identity assign `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --system-assigned

# Assign user-assigned managed identity
$MSI_RESOURCE_ID = "/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/$MSI_NAME"

az containerapp identity assign `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --user-assigned $MSI_RESOURCE_ID

# Get MSI Client ID
$MSI_CLIENT_ID = az identity show `
    --name $MSI_NAME `
    --resource-group $RESOURCE_GROUP `
    --query clientId -o tsv

# Update environment variable with MSI Client ID (preserve DEPLOY_TIMESTAMP)
az containerapp update `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --set-env-vars "AZURE_CLIENT_ID=$MSI_CLIENT_ID" "DEPLOY_TIMESTAMP=$DEPLOY_TIMESTAMP"

Write-Host "✅ Managed Identity configured" -ForegroundColor Green

# ============================================
# Step 7: Configure Storage Account Firewall
# ============================================
Write-Host ""
Write-Host "🔒 Step 7: Configuring Storage Account Firewall..." -ForegroundColor Cyan

# Check if storage account exists
$storageExists = az storage account show --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP 2>$null
if ($storageExists) {
    # Get Container App's outbound IPs
    Write-Host "   Getting Container App outbound IPs..." -ForegroundColor Gray
    $outboundIps = az containerapp show `
        --name $APP_NAME `
        --resource-group $RESOURCE_GROUP `
        --query "properties.outboundIpAddresses" -o tsv
    
    if ($outboundIps) {
        $ipArray = $outboundIps -split "`n" | Where-Object { $_ -match '\d+\.\d+\.\d+\.\d+' }
        Write-Host "   Found $($ipArray.Count) outbound IPs" -ForegroundColor Gray
        
        # Get current network rules
        $currentRules = az storage account network-rule list `
            --account-name $STORAGE_ACCOUNT_NAME `
            --resource-group $RESOURCE_GROUP `
            --query "ipRules[].ipAddressOrRange" -o tsv
        
        $currentIpSet = @{}
        if ($currentRules) {
            $currentRules -split "`n" | ForEach-Object { $currentIpSet[$_] = $true }
        }
        
        # Add each outbound IP to storage firewall (if not already present)
        $addedCount = 0
        foreach ($ip in $ipArray) {
            $ip = $ip.Trim()
            if ($ip -and -not $currentIpSet.ContainsKey($ip)) {
                Write-Host "   Adding IP: $ip" -ForegroundColor Gray
                az storage account network-rule add `
                    --account-name $STORAGE_ACCOUNT_NAME `
                    --resource-group $RESOURCE_GROUP `
                    --ip-address $ip 2>$null | Out-Null
                $addedCount++
            }
        }
        
        if ($addedCount -gt 0) {
            Write-Host "   Added $addedCount new IPs to storage firewall" -ForegroundColor Gray
        } else {
            Write-Host "   All Container App IPs already in firewall" -ForegroundColor Gray
        }
        
        # Ensure Azure Services bypass is enabled
        az storage account update `
            --name $STORAGE_ACCOUNT_NAME `
            --resource-group $RESOURCE_GROUP `
            --bypass AzureServices 2>$null | Out-Null
        
        Write-Host "✅ Storage firewall configured with Container App IPs" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Could not retrieve Container App outbound IPs" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Storage account '$STORAGE_ACCOUNT_NAME' not found - skipping firewall config" -ForegroundColor Yellow
}

# ============================================
# Step 8: Get App URL
# ============================================
Write-Host ""
Write-Host "🌍 Step 8: Getting App URL..." -ForegroundColor Cyan

$APP_FQDN = az containerapp show `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --query properties.configuration.ingress.fqdn -o tsv

$APP_URL = "https://$APP_FQDN"

# ============================================
# Summary
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Application URL: $APP_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Managed Identity Roles Required:" -ForegroundColor Yellow
Write-Host "   Ensure '$MSI_NAME' has these roles:" -ForegroundColor White
Write-Host "   • Cognitive Services OpenAI User - on Azure OpenAI resource" -ForegroundColor White
Write-Host "   • Storage Blob Data Contributor - on Storage Account" -ForegroundColor White
Write-Host "   • Monitoring Metrics Publisher - on Application Insights" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Useful Commands:" -ForegroundColor Yellow
Write-Host "   View logs:    az containerapp logs show -n $APP_NAME -g $RESOURCE_GROUP --follow" -ForegroundColor DarkGray
Write-Host "   Restart app:  az containerapp revision restart -n $APP_NAME -g $RESOURCE_GROUP" -ForegroundColor DarkGray
Write-Host "   Scale app:    az containerapp update -n $APP_NAME -g $RESOURCE_GROUP --min-replicas 2 --max-replicas 10" -ForegroundColor DarkGray
Write-Host ""
Write-Host "🌐 Test your deployment:" -ForegroundColor Yellow
Write-Host "   Start-Process '$APP_URL'" -ForegroundColor DarkGray
Write-Host ""
