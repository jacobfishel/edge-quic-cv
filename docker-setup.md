# All-in-One Deployment Guide (React + Backend)

Your Dockerfile now includes **both the React frontend and Python backend** in a single container! This is simpler but makes the image larger.

## 🏗️ Architecture

```
Single Azure Container App
├── Flask (Port 8080) → Serves React Dashboard
├── WebSocket (Port 8081) → Streams YOLO feeds
├── UDP Receiver (Port 5005) → Receives frames from Pi
├── QUIC (Port 6000) → Legacy protocol
└── React Dashboard (built-in)
```

## 📦 What's Included in the Docker Image

### Stage 1 (Node.js):
- ✅ Builds React frontend with `npm run build`
- ✅ Creates optimized production build

### Stage 2 (Python):
- ✅ Copies built React app from Stage 1
- ✅ Installs PyTorch (standard version)
- ✅ Downloads 3 YOLO models
- ✅ Sets up Flask to serve React
- ✅ Configures WebSocket server

## 🚀 Quick Deployment

```

### Step 1: Deploy to Azure

```bash
cd edge-quic-cv

# Run the deployment script
chmod +x azure-deployment.sh
./azure-deployment.sh
```

Or manually:

```bash
# Variables
RESOURCE_GROUP="edge-quic-yolo-rg"
ACR_NAME="edgequicyolo$(date +%s)"
LOCATION="eastus"
CONTAINER_APP_NAME="edge-quic-yolo-server"
CONTAINER_APP_ENV="edge-quic-yolo-env"

# Login
az login

# Create resources
az group create --name $RESOURCE_GROUP --location $LOCATION

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# Build image (includes React build)
echo "Building Docker image (this will take 15-20 minutes)..."
az acr build \
  --registry $ACR_NAME \
  --image quic-yolo-server:latest \
  --file Dockerfile.server \
  --platform linux/amd64 \
  .

# Create Container Apps environment
az extension add --name containerapp --upgrade
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Get credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

# Deploy
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
  --memory 4.0Gi

# Get URL
APP_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  -o tsv)

echo "=========================="
echo "Deployment Complete!"
echo "=========================="
echo "Dashboard: https://$APP_URL"
echo "WebSocket: wss://$APP_URL"
echo "=========================="
```

### Step 2: Access Your Dashboard

Open your browser and go to:
```
https://your-app.azurecontainerapps.io
```

You should see:
- ✅ React Dashboard loaded
- ✅ WebSocket connected
- ✅ 3 YOLO feeds (detection, segmentation, pose)

## 📊 Build Time & Size

| Component | Time | Size |
|-----------|------|------|
| React build | ~3 min | ~2 MB |
| PyTorch install | ~10 min | ~2.5 GB |
| YOLO models | ~2 min | ~18 MB |
| **Total** | **~15-20 min** | **~4.5 GB** |

## 🔄 Updating Your App

### Update React Frontend Only

```bash
# Make changes to frontend/src/Dashboard.tsx
# Then rebuild and redeploy

az acr build \
  --registry $ACR_NAME \
  --image quic-yolo-server:latest \
  --file Dockerfile.server \
  --platform linux/amd64 \
  .

az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_NAME.azurecr.io/quic-yolo-server:latest
```

### Update Backend Only

Same process - the Dockerfile rebuilds everything.

## 📱 Connecting from Raspberry Pi

Update your Raspberry Pi client to send to Azure:

```python
import os
import socket

# Azure Container App URL
SERVER_HOST = os.getenv('SERVER_HOST', 'your-app.azurecontainerapps.io')
UDP_PORT = 5005

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send frames
sock.sendto(frame_data, (SERVER_HOST, UDP_PORT))
```

⚠️ **Important:** Azure Container Apps may not expose UDP ports externally by default. See troubleshooting below.

## ⚖️ Pros & Cons

### ✅ Pros (All-in-One):
- **Single deployment** - One command deploys everything
- **Same domain** - No CORS issues
- **Simpler networking** - WebSocket on same host
- **One URL** - Easy to share

### ❌ Cons (All-in-One):
- **Larger image** (~4.5GB vs ~4GB backend-only)
- **Longer builds** (~15-20 min vs ~12-15 min)
- **Can't scale frontend separately** - Frontend and backend scale together
- **React changes require full rebuild** - Can't update frontend independently

## 🔀 Alternative: Separate Deployment

If you prefer to deploy React separately (Vercel, Netlify, etc.):

1. **Remove Flask from `quic_server.py`**
2. **Use the backend-only Dockerfile** (previous version)
3. **Deploy frontend to Vercel/Netlify**
4. **Configure CORS** on backend

### When to Use Each:

| Use Case | Recommendation |
|----------|---------------|
| **Development/Testing** | All-in-One ⭐ |
| **Simple deployment** | All-in-One ⭐ |
| **High traffic** | Separate (scale independently) |
| **Frequent frontend updates** | Separate (faster deploys) |
| **Cost optimization** | Separate (cheaper frontend hosting) |

## 🐛 Troubleshooting

### Issue: React app loads but WebSocket won't connect

**Symptom:** Dashboard shows "Disconnected"

**Solution:** Check WebSocket URL in browser console:

```javascript
// In Dashboard.tsx, add logging:
console.log('Connecting to:', WEBSOCKET_URL);
```

Make sure it's using `wss://` (not `ws://`) in production.

### Issue: UDP frames not received from Raspberry Pi

**Symptom:** No video feeds showing

**Solution:** Azure Container Apps may not expose UDP externally. Options:

1. **Use Azure Virtual Network** (supports UDP)
2. **Modify Pi client to use WebSocket** instead of UDP
3. **Use TCP/QUIC fallback**

### Issue: Build takes too long

**Symptom:** `az acr build` times out or takes 30+ minutes

**Solution:**
- Use better cache: `--no-cache` flag might be forcing full rebuild
- Check internet speed: YOLO models are ~18MB download
- Consider pre-built base images

### Issue: Container crashes with "Out of Memory"

**Symptom:** Container restarts repeatedly

**Solution:** Increase memory:

```bash
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --cpu 4.0 \
  --memory 8.0Gi
```

## 💰 Cost Estimate

All-in-One deployment (2 CPU, 4GB RAM):

| Resource | Monthly Cost |
|----------|-------------|
| Container App (2 CPU, 4GB) | ~$150-200 |
| Container Registry (Basic) | ~$5 |
| **Total** | **~$155-205/month** |

## 🎯 Next Steps

1. ✅ Deploy to Azure
2. ✅ Test from browser (https://your-app.azurecontainerapps.io)
3. ✅ Configure Raspberry Pi client
4. ✅ Monitor logs and performance
5. ⚙️ Optimize based on usage

## 📝 Summary

You now have a **complete, self-contained deployment** with:
- ✅ React Dashboard (served by Flask on port 8080)
- ✅ WebSocket streaming (port 8081)
- ✅ 3 YOLO models (detection, segmentation, pose)
- ✅ UDP/QUIC frame reception
- ✅ Single Docker image
- ✅ One-command deployment

Everything is bundled together and deployed to Azure Container Apps!