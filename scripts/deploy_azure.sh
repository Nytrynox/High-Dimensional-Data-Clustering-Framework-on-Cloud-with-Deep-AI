#!/bin/bash

# Azure Deployment Script for High-Dimensional Clustering Framework
# This script deploys the complete clustering framework to Azure

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
LOCATION="eastus"
ENVIRONMENT="dev"
PROJECT_NAME="cluster"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}📄 $1${NC}"
}

print_header() {
    echo -e "${YELLOW}$1${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -g <resource-group> [-l <location>] [-e <environment>] [-p <project-name>]"
            echo ""
            echo "Options:"
            echo "  -g, --resource-group    Azure resource group name (required)"
            echo "  -l, --location         Azure location (default: eastus)"
            echo "  -e, --environment      Environment (default: dev)"
            echo "  -p, --project-name     Project name prefix (default: cluster)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$RESOURCE_GROUP" ]; then
    print_error "Resource group is required. Use -g or --resource-group"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

echo -e "${GREEN}🚀 Starting Azure deployment for High-Dimensional Clustering Framework${NC}"
echo ""

# Check if Azure CLI is installed and logged in
print_header "🔐 Checking Azure CLI authentication..."
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi

ACCOUNT=$(az account show --query "name" -o tsv 2>/dev/null || echo "")
if [ -z "$ACCOUNT" ]; then
    print_error "Azure CLI not authenticated. Please run 'az login'"
    exit 1
fi

print_status "Azure CLI authenticated as: $ACCOUNT"

# Create resource group if it doesn't exist
print_header "📦 Creating resource group: $RESOURCE_GROUP"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null

if [ $? -eq 0 ]; then
    print_status "Resource group created/verified"
else
    print_error "Failed to create resource group"
    exit 1
fi

# Deploy main infrastructure
print_header "🏗️  Deploying infrastructure..."
DEPLOYMENT_NAME="clustering-deployment-$(date +%Y%m%d-%H%M%S)"

az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "infrastructure/main.bicep" \
    --parameters projectName="$PROJECT_NAME" environment="$ENVIRONMENT" location="$LOCATION" \
    --name "$DEPLOYMENT_NAME" \
    --no-wait

# Wait for deployment to complete
print_info "Waiting for infrastructure deployment to complete..."
az deployment group wait --resource-group "$RESOURCE_GROUP" --name "$DEPLOYMENT_NAME" --created

if [ $? -ne 0 ]; then
    print_error "Infrastructure deployment failed"
    exit 1
fi

# Get deployment outputs
print_header "📄 Retrieving deployment outputs..."
OUTPUTS=$(az deployment group show --resource-group "$RESOURCE_GROUP" --name "$DEPLOYMENT_NAME" --query "properties.outputs" -o json)

STORAGE_ACCOUNT_NAME=$(echo "$OUTPUTS" | jq -r '.storageAccountName.value // empty')
CONTAINER_REGISTRY_NAME=$(echo "$OUTPUTS" | jq -r '.containerRegistryName.value // empty')
ML_WORKSPACE_NAME=$(echo "$OUTPUTS" | jq -r '.mlWorkspaceName.value // empty')
FUNCTION_APP_NAME=$(echo "$OUTPUTS" | jq -r '.functionAppName.value // empty')
API_URL=$(echo "$OUTPUTS" | jq -r '.apiUrl.value // empty')

print_status "Infrastructure deployed successfully!"
print_info "   Storage Account: $STORAGE_ACCOUNT_NAME"
print_info "   Container Registry: $CONTAINER_REGISTRY_NAME"
print_info "   ML Workspace: $ML_WORKSPACE_NAME"
print_info "   Function App: $FUNCTION_APP_NAME"
print_info "   API URL: $API_URL"

# Build and push Docker image
print_header "🐳 Building and pushing Docker image..."

# Login to container registry
az acr login --name "$CONTAINER_REGISTRY_NAME" >/dev/null 2>&1

if [ $? -ne 0 ]; then
    print_error "Failed to login to container registry"
    exit 1
fi

# Build and push image
IMAGE_NAME="$CONTAINER_REGISTRY_NAME.azurecr.io/clustering-api:latest"
docker build -t "$IMAGE_NAME" . >/dev/null

if [ $? -ne 0 ]; then
    print_error "Docker build failed"
    exit 1
fi

docker push "$IMAGE_NAME" >/dev/null

if [ $? -ne 0 ]; then
    print_error "Docker push failed"
    exit 1
fi

print_status "Docker image built and pushed successfully!"

# Update Container App with new image
print_header "🔄 Updating Container App..."
az containerapp update \
    --resource-group "$RESOURCE_GROUP" \
    --name "$PROJECT_NAME-$ENVIRONMENT-api" \
    --image "$IMAGE_NAME" >/dev/null 2>&1

if [ $? -eq 0 ]; then
    print_status "Container App updated successfully!"
else
    print_warning "Container App update failed, but continuing..."
fi

# Deploy Azure Functions
print_header "⚡ Deploying Azure Functions..."

# Create functions deployment package
[ -f "functions.zip" ] && rm "functions.zip"

if [ -d "src/functions" ]; then
    cd src/functions
    zip -r "../../functions.zip" . >/dev/null
    cd ../../
    
    # Deploy functions
    az functionapp deployment source config-zip \
        --resource-group "$RESOURCE_GROUP" \
        --name "$FUNCTION_APP_NAME" \
        --src "functions.zip" >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_status "Azure Functions deployed successfully!"
    else
        print_warning "Azure Functions deployment failed"
    fi
    
    rm "functions.zip"
else
    print_warning "No Azure Functions code found, skipping..."
fi

# Upload sample data to storage
print_header "📊 Uploading sample data..."

# Generate sample data if it doesn't exist
if [ ! -f "data/sample_data.csv" ]; then
    print_info "   Generating sample data..."
    mkdir -p data
    python scripts/generate_sample_data.py >/dev/null 2>&1 || print_warning "Failed to generate sample data"
fi

# Upload to Azure Storage if data exists
if [ -d "data" ] && [ "$(ls -A data/*.csv 2>/dev/null)" ]; then
    az storage blob upload-batch \
        --account-name "$STORAGE_ACCOUNT_NAME" \
        --destination "clustering-data" \
        --source "data/" \
        --pattern "*.csv" >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_status "Sample data uploaded successfully!"
    else
        print_warning "Sample data upload failed"
    fi
else
    print_warning "No sample data found to upload"
fi

# Setup ML workspace
print_header "🧠 Setting up ML Workspace..."

# Install Azure ML extension if not already installed
az extension add --name ml --yes --only-show-errors >/dev/null 2>&1

# Create compute instance
az ml compute create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$ML_WORKSPACE_NAME" \
    --name "clustering-compute" \
    --type ComputeInstance \
    --size Standard_DS3_v2 >/dev/null 2>&1

if [ $? -eq 0 ]; then
    print_status "ML Workspace compute created!"
else
    print_warning "ML Workspace compute creation failed"
fi

# Create environment file with deployment details
print_header "📝 Creating environment configuration..."

cat > .env.azure << EOF
# Azure Deployment Configuration
AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
AZURE_LOCATION=$LOCATION
AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT_NAME
AZURE_CONTAINER_REGISTRY=$CONTAINER_REGISTRY_NAME
AZURE_ML_WORKSPACE=$ML_WORKSPACE_NAME
AZURE_FUNCTION_APP=$FUNCTION_APP_NAME
API_URL=$API_URL

# Generated on $(date '+%Y-%m-%d %H:%M:%S')
EOF

print_status "Environment configuration saved to .env.azure"

# Display final summary
echo ""
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo ""
echo -e "${YELLOW}📋 Deployment Summary:${NC}"
echo -e "   Resource Group: ${CYAN}$RESOURCE_GROUP${NC}"
echo -e "   Location: ${CYAN}$LOCATION${NC}"
echo -e "   Environment: ${CYAN}$ENVIRONMENT${NC}"
echo -e "   API Endpoint: ${CYAN}$API_URL${NC}"
echo ""
echo -e "${YELLOW}🔗 Next Steps:${NC}"
echo -e "   1. Visit the API: ${CYAN}$API_URL/docs${NC}"
echo -e "   2. Check ML Workspace: ${CYAN}https://ml.azure.com${NC}"
echo -e "   3. Monitor with Application Insights"
echo ""
echo -e "${CYAN}📁 Configuration saved to: .env.azure${NC}"

print_status "Deployment completed successfully!"
