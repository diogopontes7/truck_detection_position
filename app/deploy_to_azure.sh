#!/bin/bash

# 🚛 Truck Alignment API - Azure Deployment Script

set -e

echo "🚀 Starting Truck Alignment API deployment to Azure..."

# Configuration
RESOURCE_GROUP="truck-alignment-rg"
LOCATION="eastus"
ACR_NAME="truckalignmentacr"
AKS_NAME="truck-alignment-aks"
APP_NAME="truck-alignment-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install it first."
    exit 1
fi

# Login to Azure
print_status "Logging in to Azure..."
az login

# Create Resource Group
print_status "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
print_status "Creating Azure Container Registry: $ACR_NAME"
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
print_status "ACR Login Server: $ACR_LOGIN_SERVER"

# Create AKS cluster
print_status "Creating AKS cluster: $AKS_NAME"
az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_NAME \
    --node-count 2 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --attach-acr $ACR_NAME

# Get AKS credentials
print_status "Getting AKS credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_NAME

# Build and push Docker image
print_status "Building and pushing Docker image..."
az acr build --registry $ACR_NAME --image $APP_NAME:latest .

# Update deployment YAML with correct image
print_status "Updating deployment configuration..."
sed -i "s|truck-alignment-api:latest|$ACR_LOGIN_SERVER/$APP_NAME:latest|g" azure-deployment.yaml

# Deploy to AKS
print_status "Deploying to AKS..."
kubectl apply -f azure-deployment.yaml

# Wait for deployment to be ready
print_status "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$APP_NAME

# Get service IP
print_status "Getting service information..."
kubectl get service truck-alignment-service

# Test the API
print_status "Testing API health endpoint..."
SERVICE_IP=$(kubectl get service truck-alignment-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ ! -z "$SERVICE_IP" ]; then
    print_status "Service IP: $SERVICE_IP"
    print_status "Testing health endpoint..."
    curl -s "http://$SERVICE_IP/health" || print_warning "Health check failed, service might still be starting"
else
    print_warning "Service IP not available yet. Please check with: kubectl get service truck-alignment-service"
fi

print_status "Deployment completed!"
print_status "API endpoints:"
echo "  - Health: http://$SERVICE_IP/health"
echo "  - Config: http://$SERVICE_IP/config"
echo "  - Docs: http://$SERVICE_IP/docs"
echo ""
print_status "Useful commands:"
echo "  - Check pods: kubectl get pods"
echo "  - Check services: kubectl get services"
echo "  - View logs: kubectl logs -l app=$APP_NAME"
echo "  - Delete deployment: kubectl delete -f azure-deployment.yaml" 