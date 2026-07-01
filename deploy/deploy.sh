#!/usr/bin/env bash
# ==============================================================================
# FinDoc Intelligence (FDI) — Automated Azure Deployment Script
# ==============================================================================
# Sets up resource groups, ACR, Container App environment, builds/pushes images,
# configures secrets, and deploys both backend API and frontend Streamlit UI.
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ==============================================================================

set -eo pipefail

# Configuration
RESOURCE_GROUP="kpmg-rag-rg"
LOCATION="centralindia"
REGISTRY="kpmgregistry"
ENV_NAME="kpmg-env"

# Fetch default subscription info
SUBSCRIPTION_NAME=$(az account show --query name -o tsv || echo "unknown")
echo "=== Deploying FinDoc Intelligence to Azure ==="
echo "Active Subscription: $SUBSCRIPTION_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Registry Name: $REGISTRY"
echo "Container Environment: $ENV_NAME"
echo "=============================================="

# Ensure user is logged in
if [[ "$SUBSCRIPTION_NAME" == "unknown" ]]; then
    echo "ERROR: Not logged into Azure. Please run 'az login' first."
    exit 1
fi

# 1. Create Resource Group
echo "--> Creating Resource Group: $RESOURCE_GROUP..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# 2. Create Azure Container Registry (ACR)
echo "--> Creating Container Registry: $REGISTRY..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$REGISTRY" \
  --sku Basic \
  --admin-enabled true

# 3. Create Container Apps Environment
echo "--> Creating Container Apps Environment: $ENV_NAME..."
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

# 4. Build & Push Docker Images using ACR Tasks (no local Docker required)
echo "--> Building backend API image in Azure ACR..."
az acr build \
  --registry "$REGISTRY" \
  --image "fdi-api:latest" \
  --file Dockerfile \
  .

echo "--> Building frontend Streamlit UI image in Azure ACR..."
az acr build \
  --registry "$REGISTRY" \
  --image "fdi-ui:latest" \
  --file Dockerfile.streamlit \
  .

# 5. Deploy Backend Container App (API)
echo "--> Deploying Backend API App..."
az containerapp create \
  --name fdi-app \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "$REGISTRY.azurecr.io/fdi-api:latest" \
  --registry-server "$REGISTRY.azurecr.io" \
  --target-port 8000 \
  --ingress external \
  --cpu 1 \
  --memory 2Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --env-vars \
    CHROMA_DB_PATH=/app/chroma_db \
    DEBUG=False

# Get Backend API FQDN URL
API_URL=$(az containerapp show \
  --name fdi-app \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "--> Backend API live at: https://$API_URL"

if [ -z "$GROQ_API_KEY" ]; then
  echo "Error: GROQ_API_KEY environment variable is not set."
  exit 1
fi
if [ -z "$NVIDIA_API_KEY" ]; then
  echo "Error: NVIDIA_API_KEY environment variable is not set."
  exit 1
fi
az containerapp secret set \
  --name fdi-app \
  --resource-group "$RESOURCE_GROUP" \
  --secrets \
    "groq-api-key=$GROQ_API_KEY" \
    "nvidia-api-key=$NVIDIA_API_KEY"
    
az containerapp update \
  --name fdi-app \
  --resource-group "$RESOURCE_GROUP" \
  --set-env-vars \
    "GROQ_API_KEY=secretref:groq-api-key" \
    "NVIDIA_API_KEY=secretref:nvidia-api-key"

# 7. Deploy Streamlit UI Container App
echo "--> Deploying Streamlit UI App..."
az containerapp create \
  --name fdi-ui \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENV_NAME" \
  --image "$REGISTRY.azurecr.io/fdi-ui:latest" \
  --registry-server "$REGISTRY.azurecr.io" \
  --target-port 8501 \
  --ingress external \
  --cpu 0.5 \
  --memory 1Gi \
  --env-vars \
    API_URL="https://$API_URL"

# Get UI FQDN URL
UI_URL=$(az containerapp show \
  --name fdi-ui \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "=============================================="
echo "Deployment Complete!"
echo "Backend API: https://$API_URL"
echo "Frontend UI: https://$UI_URL"
echo "=============================================="
