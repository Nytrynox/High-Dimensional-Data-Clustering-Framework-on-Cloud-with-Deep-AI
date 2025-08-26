# Azure Deployment Script
# Deploy the clustering framework to Azure

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$Location = "East US",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [string]$ProjectName = "cluster"
)

Write-Host "🚀 Starting Azure deployment for High-Dimensional Clustering Framework" -ForegroundColor Green

# Check if Azure CLI is installed and logged in
try {
    $account = az account show --query "name" -o tsv
    Write-Host "✅ Azure CLI authenticated as: $account" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI not authenticated. Please run 'az login'" -ForegroundColor Red
    exit 1
}

# Create resource group if it doesn't exist
Write-Host "📦 Creating resource group: $ResourceGroupName" -ForegroundColor Yellow
az group create --name $ResourceGroupName --location $Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create resource group" -ForegroundColor Red
    exit 1
}

# Deploy main infrastructure
Write-Host "🏗️  Deploying infrastructure..." -ForegroundColor Yellow
$deploymentName = "clustering-deployment-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

az deployment group create `
    --resource-group $ResourceGroupName `
    --template-file "infrastructure/main.bicep" `
    --parameters projectName=$ProjectName environment=$Environment location=$Location `
    --name $deploymentName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Infrastructure deployment failed" -ForegroundColor Red
    exit 1
}

# Get deployment outputs
Write-Host "📄 Retrieving deployment outputs..." -ForegroundColor Yellow
$outputs = az deployment group show --resource-group $ResourceGroupName --name $deploymentName --query "properties.outputs" -o json | ConvertFrom-Json

$storageAccountName = $outputs.storageAccountName.value
$containerRegistryName = $outputs.containerRegistryName.value
$mlWorkspaceName = $outputs.mlWorkspaceName.value
$functionAppName = $outputs.functionAppName.value
$apiUrl = $outputs.apiUrl.value

Write-Host "✅ Infrastructure deployed successfully!" -ForegroundColor Green
Write-Host "   Storage Account: $storageAccountName" -ForegroundColor Cyan
Write-Host "   Container Registry: $containerRegistryName" -ForegroundColor Cyan
Write-Host "   ML Workspace: $mlWorkspaceName" -ForegroundColor Cyan
Write-Host "   Function App: $functionAppName" -ForegroundColor Cyan
Write-Host "   API URL: $apiUrl" -ForegroundColor Cyan

# Build and push Docker image
Write-Host "🐳 Building and pushing Docker image..." -ForegroundColor Yellow

# Login to container registry
az acr login --name $containerRegistryName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to login to container registry" -ForegroundColor Red
    exit 1
}

# Build and push image
$imageName = "$containerRegistryName.azurecr.io/clustering-api:latest"
docker build -t $imageName .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

docker push $imageName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker image built and pushed successfully!" -ForegroundColor Green

# Update Container App with new image
Write-Host "🔄 Updating Container App..." -ForegroundColor Yellow
az containerapp update `
    --resource-group $ResourceGroupName `
    --name "$ProjectName-$Environment-api" `
    --image $imageName

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Container App update failed, but continuing..." -ForegroundColor Yellow
}

# Deploy Azure Functions
Write-Host "⚡ Deploying Azure Functions..." -ForegroundColor Yellow

# Create functions deployment package
if (Test-Path "functions.zip") {
    Remove-Item "functions.zip"
}

# Zip the functions code
Compress-Archive -Path "src/functions/*" -DestinationPath "functions.zip"

# Deploy functions
az functionapp deployment source config-zip `
    --resource-group $ResourceGroupName `
    --name $functionAppName `
    --src "functions.zip"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Azure Functions deployed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Azure Functions deployment failed" -ForegroundColor Yellow
}

# Upload sample data to storage
Write-Host "📊 Uploading sample data..." -ForegroundColor Yellow

# Generate sample data if it doesn't exist
if (-not (Test-Path "data/sample_data.csv")) {
    Write-Host "   Generating sample data..." -ForegroundColor Yellow
    python scripts/generate_sample_data.py
}

# Upload to Azure Storage
az storage blob upload-batch `
    --account-name $storageAccountName `
    --destination "clustering-data" `
    --source "data/" `
    --pattern "*.csv"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Sample data uploaded successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Sample data upload failed" -ForegroundColor Yellow
}

# Setup ML workspace
Write-Host "🧠 Setting up ML Workspace..." -ForegroundColor Yellow

# Install Azure ML extension if not already installed
az extension add --name ml --yes --only-show-errors

# Create compute instance
az ml compute create `
    --resource-group $ResourceGroupName `
    --workspace-name $mlWorkspaceName `
    --name "clustering-compute" `
    --type ComputeInstance `
    --size Standard_DS3_v2

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ ML Workspace compute created!" -ForegroundColor Green
} else {
    Write-Host "⚠️  ML Workspace compute creation failed" -ForegroundColor Yellow
}

# Create environment file with deployment details
Write-Host "📝 Creating environment configuration..." -ForegroundColor Yellow

$envConfig = @"
# Azure Deployment Configuration
AZURE_RESOURCE_GROUP=$ResourceGroupName
AZURE_LOCATION=$Location
AZURE_STORAGE_ACCOUNT=$storageAccountName
AZURE_CONTAINER_REGISTRY=$containerRegistryName
AZURE_ML_WORKSPACE=$mlWorkspaceName
AZURE_FUNCTION_APP=$functionAppName
API_URL=$apiUrl

# Generated on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
"@

$envConfig | Out-File -FilePath ".env.azure" -Encoding UTF8

Write-Host "✅ Environment configuration saved to .env.azure" -ForegroundColor Green

# Display final summary
Write-Host ""
Write-Host "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!" -ForegroundColor Green -BackgroundColor Black
Write-Host ""
Write-Host "📋 Deployment Summary:" -ForegroundColor Yellow
Write-Host "   Resource Group: $ResourceGroupName" -ForegroundColor White
Write-Host "   Location: $Location" -ForegroundColor White
Write-Host "   Environment: $Environment" -ForegroundColor White
Write-Host "   API Endpoint: $apiUrl" -ForegroundColor White
Write-Host ""
Write-Host "🔗 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Visit the API: $apiUrl/docs" -ForegroundColor White
Write-Host "   2. Check ML Workspace: https://ml.azure.com" -ForegroundColor White
Write-Host "   3. Monitor with Application Insights" -ForegroundColor White
Write-Host ""
Write-Host "📁 Configuration saved to: .env.azure" -ForegroundColor Cyan

# Cleanup
if (Test-Path "functions.zip") {
    Remove-Item "functions.zip"
}
