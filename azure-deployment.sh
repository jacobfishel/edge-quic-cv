#!/bin/bash

# Edge QUIC CV - Azure Container Apps Deployment Script (AMD64 Only)
# Simplified version for AMD64 architecture
# Works on Intel/AMD machines (Linux, Windows WSL, Intel Macs)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_header "Edge QUIC CV - Azure Deployment (AMD64)"

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v az &> /dev/null; then
    print_error "Azure CLI not found."
    print_info "Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi
print_success "Azure CLI found"

# Check if logged into Azure
print_info "Checking Azure login status..."
if ! az account show &> /dev/null; then
    print_warning "Not logged into Azure. Logging in..."
    az login
else
    print_success "Already logged into Azure"
    CURRENT_ACCOUNT=$(az account show --query name -o tsv)
    print_info "Current subscription: $CURRENT_ACCOUNT"
    
    read -p "Continue with this subscription? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Available subscriptions:"
        az account list --output table
        read -p "Enter subscription name or ID: " SUB_CHOICE
        az account set --subscription "$SUB_CHOICE"
        print_success "Switched to subscription: $SUB_CHOICE"
    fi
fi

# Configuration
print_header "Configuration"

# Get configuration from user or use defaults
read -p "Resource Group name (default: edge-quic-yolo-rg): " RESOURCE_GROUP
RESOURCE_GROUP=${RESOURCE_GROUP:-edge-quic-yolo-rg}

read -p "Azure Region (default: eastus): " LOCATION
LOCATION=${LOCATION:-eastus}

# Generate unique ACR name (lowercase alphanumeric only)
DEFAULT_ACR_NAME="edgequicyolo$(date +%s)"
read -p "Azure Container Registry name (default: $DEFAULT_ACR_NAME): " ACR_NAME
ACR_NAME=${ACR_NAME:-$DEFAULT_ACR_NAME}
ACR_NAME=$(echo "$ACR_NAME" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')

read -p "Container App Environment name (default: edge-quic-yolo-env): " CONTAINER_APP_ENV
CONTAINER_APP_ENV=${CONTAINER_APP_ENV:-edge-quic-yolo-env}

read -p "Container App name (default: edge-quic-yolo-server): " CONTAINER_APP_NAME
CONTAINER_APP_NAME=${CONTAINER_APP_NAME:-edge-quic-yolo-server}

# Confirm configuration
print_info "\nDeployment Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Container Registry: $ACR_NAME"
echo "  Container App Environment: $CONTAINER_APP_ENV"
echo "  Container App: $CONTAINER_APP_NAME"
echo "  Architecture: AMD64 (x86_64)"
echo ""

read -p "Proceed with deployment? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Deployment cancelled"
    exit 0
fi

# Step 1: Create Resource Group
print_header "Step 1: Creating Resource Group"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    print_warning "Resource group '$RESOURCE_GROUP' already exists"
else
    az group create \
        --name $RESOURCE_GROUP \
        --location $LOCATION \
        --output none
    print_success "Resource group created: $RESOURCE_GROUP"
fi

# Step 2: Create Azure Container Registry
print_header "Step 2: Creating Azure Container Registry"
if az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    print_warning "Container Registry '$ACR_NAME' already exists"
else
    print_info "Creating ACR (this may take a minute)..."
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --admin-enabled true \
        --output none
    print_success "Container Registry created: $ACR_NAME"
fi

# Step 3: Check for Dockerfile
print_header "Step 3: Checking for Dockerfile"
if [ ! -f "Dockerfile.server" ]; then
    print_warning "Dockerfile.server not found. Creating all-in-one version (React + Backend)..."
    
    cat > Dockerfile.server << 'EOF'
# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy frontend source
COPY frontend/ ./
COPY public/ ./public/ 2>/dev/null || true

# Build the React frontend
RUN npm run build

# Stage 2: Python application with built React frontend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    openssl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install all Python dependencies (including PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY quic_server.py .

# Copy built React frontend from Stage 1
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Generate SSL certificates
RUN openssl req -new -x509 -days 365 -nodes \
    -out cert.pem -keyout key.pem \
    -subj "/CN=localhost"

# Create directory for YOLO models cache
RUN mkdir -p /root/.config/Ultralytics

# Pre-download YOLO models
RUN python -c "from ultralytics import YOLO; \
    print('Downloading YOLOv8 models...'); \
    YOLO('yolov8n.pt'); \
    YOLO('yolov8n-seg.pt'); \
    YOLO('yolov8n-pose.pt'); \
    print('All YOLO models downloaded successfully')"

# Expose ports
EXPOSE 5005/udp 6000/udp 8080 8081

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD timeout 5 bash -c 'cat < /dev/null > /dev/tcp/127.0.0.1/8080' || exit 1

# Start the server
CMD ["python", "-u", "quic_server.py"]
EOF
    print_success "Dockerfile.server created (all-in-one with React)"
else
    print_success "Dockerfile.server found"
fi

# Create .dockerignore if it doesn't exist
if [ ! -f ".dockerignore" ]; then
    print_info "Creating .dockerignore to speed up builds..."
    cat > .dockerignore << 'EOF'
__pycache__/
*.py[cod]
.git/
.vscode/
.idea/
node_modules/
*.md
.DS_Store
test_data/
deployment-info.txt
.azure/
cert.pem
key.pem
client.py
edge_client.py
udp_client.py
frontend/build/
venv/
*.log
*.pt
*.pth
*.weights
EOF
    print_success ".dockerignore created"
fi

# Step 4: Build Docker Image in Azure
print_header "Step 4: Building Docker Image in Azure (AMD64)"
print_warning "This will take 15-20 minutes (React build + PyTorch + YOLO models)..."
print_info "Building for AMD64 architecture..."

az acr build \
    --registry $ACR_NAME \
    --image quic-yolo-server:latest \
    --file Dockerfile.server \
    --platform linux/amd64 \
    . 

print_success "Docker image built and pushed to ACR"

# Step 5: Install Container Apps Extension
print_header "Step 5: Setting up Container Apps"
print_info "Installing/updating Container Apps extension..."
az extension add --name containerapp --upgrade --only-show-errors 2>/dev/null || true
print_success "Container Apps extension ready"

# Step 6: Create Container Apps Environment
print_header "Step 6: Creating Container Apps Environment"
if az containerapp env show --name $CONTAINER_APP_ENV --resource-group $RESOURCE_GROUP &> /dev/null; then
    print_warning "Container Apps environment '$CONTAINER_APP_ENV' already exists"
else
    print_info "Creating Container Apps environment (this may take 2-3 minutes)..."
    az containerapp env create \
        --name $CONTAINER_APP_ENV \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --output none
    print_success "Container Apps environment created"
fi

# Step 7: Get ACR Credentials
print_header "Step 7: Retrieving ACR Credentials"
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
print_success "ACR credentials retrieved"

# Step 8: Deploy Container App
print_header "Step 8: Deploying Container App"

if az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    print_warning "Container App '$CONTAINER_APP_NAME' already exists. Updating..."
    az containerapp update \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --image $ACR_NAME.azurecr.io/quic-yolo-server:latest \
        --output none
    print_success "Container App updated"
else
    print_info "Creating Container App (this may take 2-3 minutes)..."
    az containerapp create \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --environment $CONTAINER_APP_ENV \
        --image $ACR_NAME.azurecr.io/quic-yolo-server:latest \
        --registry-server $ACR_NAME.azurecr.io \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --target-port 8080 \
        --ingress external \
        --min-replicas 1 \
        --max-replicas 2 \
        --cpu 2.0 \
        --memory 4.0Gi \
        --env-vars \
            PYTHONUNBUFFERED=1 \
        --output none
    print_success "Container App created"
fi

# Step 9: Get Application URL
print_header "Deployment Complete! 🎉"

APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn \
    -o tsv)

print_success "Your application is deployed!"
echo ""
print_info "Application Details:"
echo "  🌐 Dashboard URL: https://$APP_URL"
echo "  🔌 WebSocket URL: wss://$APP_URL"
echo "  📡 UDP Receiver: $APP_URL:5005 (may need VNet for external access)"
echo ""

print_info "React Dashboard Features:"
echo "  ✅ Auto-connects to WebSocket"
echo "  ✅ Shows 3 YOLO feeds (detection, segmentation, pose)"
echo "  ✅ Real-time FPS counter"
echo "  ✅ Connection status indicator"
echo ""

print_info "Next Steps:"
echo "  1. Open browser: https://$APP_URL"
echo "  2. Update Raspberry Pi client to send to: $APP_URL:5005"
echo "  3. View logs:"
echo "     az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""

print_info "Resource Information:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Region: $LOCATION"
echo "  Container Registry: $ACR_NAME.azurecr.io"
echo "  Architecture: AMD64 (x86_64)"
echo ""

print_warning "Cost Management:"
echo "  • Current configuration: 2 CPU, 4GB RAM (~$150-200/month)"
echo "  • To stop when not in use:"
echo "    az containerapp scale --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --min-replicas 0"
echo "  • To delete everything:"
echo "    az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""

# Save deployment info to file
cat > deployment-info.txt << EOF
Edge QUIC CV - Deployment Information
Generated: $(date)
Architecture: AMD64 (x86_64)

Resource Group: $RESOURCE_GROUP
Location: $LOCATION
Container Registry: $ACR_NAME.azurecr.io
Container App: $CONTAINER_APP_NAME
Environment: $CONTAINER_APP_ENV

Application URLs:
  Dashboard: https://$APP_URL
  WebSocket: wss://$APP_URL
  UDP Receiver: $APP_URL:5005

Raspberry Pi Configuration:
  SERVER_HOST = '$APP_URL'
  UDP_PORT = 5005

React Dashboard:
  - Auto-connects to WebSocket
  - No manual configuration needed
  - Shows 3 YOLO model feeds

Useful Commands:
  View logs:
    az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow
  
  Update app (after code changes):
    az acr build --registry $ACR_NAME --image quic-yolo-server:latest --file Dockerfile.server --platform linux/amd64 .
    az containerapp update --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --image $ACR_NAME.azurecr.io/quic-yolo-server:latest
  
  Scale down (stop):
    az containerapp scale --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --min-replicas 0 --max-replicas 0
  
  Scale up (start):
    az containerapp scale --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --min-replicas 1 --max-replicas 2
  
  Delete all resources:
    az group delete --name $RESOURCE_GROUP --yes --no-wait
EOF

print_success "Deployment information saved to: deployment-info.txt"
echo ""
print_success "🚀 Deployment completed successfully!"