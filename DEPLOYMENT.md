# Azure Deployment Guide
## FinDoc Intelligence (FDI) — Azure Container Apps

---

## Prerequisites

```bash
# Install Azure CLI
brew install azure-cli   # macOS
# or: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash  (Linux)

# Login
az login

# Verify student account
az account show
```

---

## Step 1: Create Azure Resources

```bash
# Set variables
export RESOURCE_GROUP="kpmg-rag-rg"
export LOCATION="eastus"
export REGISTRY="kpmgregistry"
export ENV_NAME="kpmg-env"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY \
  --sku Basic \
  --admin-enabled true

# Create Container Apps environment
az containerapp env create \
  --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

---

## Step 2: Build & Push Docker Images

```bash
# Login to ACR
az acr login --name $REGISTRY

# Build API image
docker build -t $REGISTRY.azurecr.io/fdi-api:latest -f Dockerfile .
docker push $REGISTRY.azurecr.io/fdi-api:latest

# Build UI image
docker build -t $REGISTRY.azurecr.io/fdi-ui:latest -f Dockerfile.streamlit .
docker push $REGISTRY.azurecr.io/fdi-ui:latest
```

---

## Step 3: Deploy API Container App

```bash
az containerapp create \
  --name fdi-app \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $REGISTRY.azurecr.io/fdi-api:latest \
  --registry-server $REGISTRY.azurecr.io \
  --target-port 8000 \
  --ingress external \
  --cpu 1 \
  --memory 2Gi \
  --min-replicas 1 \
  --max-replicas 5 \
  --env-vars \
    GROQ_API_KEY=secretref:groq-api-key \
    CHROMA_DB_PATH=/app/chroma_db \
    DEBUG=False

# Get the live URL
az containerapp show \
  --name fdi-app \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv
```

---

## Step 4: Deploy Streamlit UI

```bash
# Get API URL first
API_URL=$(az containerapp show \
  --name fdi-app \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

az containerapp create \
  --name fdi-ui \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $REGISTRY.azurecr.io/fdi-ui:latest \
  --registry-server $REGISTRY.azurecr.io \
  --target-port 8501 \
  --ingress external \
  --cpu 0.5 \
  --memory 1Gi \
  --env-vars \
    API_URL=https://$API_URL
```

---

## Step 5: Configure Secrets (API Keys)

```bash
# Store Groq API key as a secret
az containerapp secret set \
  --name fdi-app \
  --resource-group $RESOURCE_GROUP \
  --secrets groq-api-key=$GROQ_API_KEY

# Update the container to use the secret
az containerapp update \
  --name fdi-app \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars "GROQ_API_KEY=secretref:groq-api-key"
```

---

## Step 6: Set Up CI/CD (GitHub Actions)

```bash
# Create service principal for GitHub Actions
az ad sp create-for-rbac \
  --name "kpmg-rag-github-sp" \
  --role Contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth
```

Copy the JSON output and add it as a GitHub secret:
- Go to GitHub Repo → Settings → Secrets → Actions
- Add secret: `AZURE_CREDENTIALS` = (the JSON from above)

The CI/CD pipeline will automatically:
1. Run tests on every push/PR
2. Build and push Docker images on `main` branch
3. Deploy to Azure Container Apps

---

## Estimated Azure Costs (Student Account)

| Resource | SKU | Est. Cost/Month |
|----------|-----|----------------|
| Container Registry | Basic | ~$5 |
| Container Apps (API) | 1 vCPU, 2GB | ~$15 |
| Container Apps (UI) | 0.5 vCPU, 1GB | ~$8 |
| Storage Account | LRS | ~$2 |
| **Total** | | **~$30/month** |

Student account provides $100/month free credit.

---

## Monitoring

```bash
# View API logs
az containerapp logs show \
  --name fdi-app \
  --resource-group $RESOURCE_GROUP \
  --follow

# View metrics in Azure Portal
# → Resource Group → fdi-app → Metrics → CPU/Memory/Request Count
```
