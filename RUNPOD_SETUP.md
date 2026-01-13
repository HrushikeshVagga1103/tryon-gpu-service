# RunPod Environment Variables Setup

## Required Environment Variables

Set these in your RunPod pod's environment variables section:

### 1. `API_KEY` (Required)
Your secret API key for authenticating requests to the service.

**Example:**
```
API_KEY=your-super-secret-api-key-12345
```

**How to set in RunPod:**
- Go to your pod's configuration
- Add environment variable: `API_KEY`
- Set value to a strong, random string

---

### 2. `GCS_SERVICE_ACCOUNT_PATH` (Required)
Full path to your Google Cloud Storage service account JSON file.

**Example:**
```
GCS_SERVICE_ACCOUNT_PATH=/workspace/gcs-service-account.json
```

**Steps:**
1. Upload your GCS service account JSON file to RunPod (via web UI or SCP)
2. Note the full path where it's stored
3. Set this environment variable to that path

**Common locations on RunPod:**
- `/workspace/gcs-service-account.json` (if uploaded to workspace)
- `/root/gcs-service-account.json` (if uploaded to home directory)
- `/runpod-volume/gcs-service-account.json` (if using RunPod volumes)

---

### 3. `MODEL_PATH` (Required)
Hugging Face model ID or local path to your CatVTON/Stable Diffusion model.

**⚠️ IMPORTANT: This is now REQUIRED. No default model is provided.**

**Examples:**

**Option A: Use Stable Diffusion Inpainting (Recommended for testing):**
```
MODEL_PATH=runwayml/stable-diffusion-inpainting
```

**Option B: Use CompVis Stable Diffusion Inpainting:**
```
MODEL_PATH=CompVis/stable-diffusion-inpainting
```

**Option C: Use your local CatVTON model:**
```
MODEL_PATH=/workspace/models/catvton
```

**Option D: Use a custom Hugging Face model:**
```
MODEL_PATH=your-username/your-model-name
```

**Note:** Make sure the model is compatible with `StableDiffusionInpaintPipeline` from the `diffusers` library.

---

## Optional Environment Variables

### 4. `TEMP_DIR` (Optional)

**Examples:**
```
# Use Hugging Face model
MODEL_PATH=levihsu/OOTDiffusion

# Use local model path
MODEL_PATH=/workspace/models/catvton

# Leave empty to use default
MODEL_PATH=
```

---

### 4. `TEMP_DIR` (Optional)
Directory for temporary file processing. Defaults to `/tmp/cat-tryon` if not set.

**Example:**
```
TEMP_DIR=/tmp/cat-tryon
```

**Note:** On RunPod, `/tmp` is usually fine, but if you have a mounted volume with more space, you might want to use that:
```
TEMP_DIR=/workspace/tmp
```

---

## Complete RunPod Setup Checklist

### Step 1: Upload GCS Service Account File
1. Download your GCS service account JSON from Google Cloud Console
2. Upload it to your RunPod instance (via web UI or SCP)
3. Note the full path (e.g., `/workspace/gcs-service-account.json`)

### Step 2: Set Environment Variables in RunPod
In your RunPod pod configuration, add these environment variables:

```
API_KEY=your-secret-api-key-here
GCS_SERVICE_ACCOUNT_PATH=/workspace/gcs-service-account.json
MODEL_PATH=runwayml/stable-diffusion-inpainting
TEMP_DIR=/tmp/cat-tryon
```

**⚠️ Important:** `MODEL_PATH` is required. Use one of the following:
- `runwayml/stable-diffusion-inpainting` (recommended for testing)
- `CompVis/stable-diffusion-inpainting`
- Your local CatVTON model path
- Your custom Hugging Face model ID

### Step 3: Verify File Permissions
Make sure the GCS service account file is readable:
```bash
chmod 644 /workspace/gcs-service-account.json
```

### Step 4: Test the Service
Once the pod is running, test the health endpoint:
```bash
curl http://localhost:8000/health
```

---

## Quick Copy-Paste for RunPod Dashboard

When setting up in RunPod, you can use this format:

**Environment Variables:**
```
API_KEY=change-this-to-your-secret-key
GCS_SERVICE_ACCOUNT_PATH=/workspace/gcs-service-account.json
MODEL_PATH=runwayml/stable-diffusion-inpainting
```

**Docker Command (if running manually):**
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e API_KEY=your-secret-api-key \
  -e GCS_SERVICE_ACCOUNT_PATH=/workspace/gcs-service-account.json \
  -v /workspace/gcs-service-account.json:/workspace/gcs-service-account.json:ro \
  -v /workspace:/app \
  -w /app \
  your-image-name
```

---

## Troubleshooting

### Error: "GCS not configured"
- Check that `GCS_SERVICE_ACCOUNT_PATH` is set correctly
- Verify the file exists at that path
- Check file permissions (should be readable)

### Error: "Invalid API key"
- Verify `API_KEY` environment variable is set
- Make sure you're sending the same key in the `X-API-Key` header

### Error: "Model not loaded"
- Check GPU availability: `nvidia-smi`
- Verify CUDA is properly configured
- Check model download permissions if using Hugging Face

