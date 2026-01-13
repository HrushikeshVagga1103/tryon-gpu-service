# Quick Start - RunPod Environment Variables

## ⚠️ Fix for the 404 Error

The error you're seeing is because `MODEL_PATH` is not set. You **MUST** set it to a valid model.

## Required Environment Variables

Set these 3 environment variables in your RunPod pod:

```bash
API_KEY=your-secret-api-key-here
GCS_SERVICE_ACCOUNT_PATH=/workspace/gcs-service-account.json
MODEL_PATH=runwayml/stable-diffusion-inpainting
```

## Quick Setup

### 1. Set Environment Variables in RunPod

In your RunPod pod settings, add:

| Variable | Value | Example |
|----------|-------|---------|
| `API_KEY` | Your secret key | `my-secret-key-123` |
| `GCS_SERVICE_ACCOUNT_PATH` | Path to your GCS JSON file | `/workspace/gcs-service-account.json` |
| `MODEL_PATH` | Hugging Face model ID | `runwayml/stable-diffusion-inpainting` |

### 2. Recommended MODEL_PATH Options

**For testing (works immediately):**
```
MODEL_PATH=runwayml/stable-diffusion-inpainting
```

**Alternative:**
```
MODEL_PATH=CompVis/stable-diffusion-inpainting
```

**If you have a local CatVTON model:**
```
MODEL_PATH=/workspace/models/catvton
```

### 3. Restart Your Service

After setting the environment variables, restart your service:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Verify It Works

Check the health endpoint:
```bash
curl http://localhost:8000/health
```

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gcs_configured": true,
  "cuda_available": true
}
```

## Common Issues

### Error: "MODEL_PATH environment variable is required"
- **Solution:** Set the `MODEL_PATH` environment variable (see above)

### Error: "404 Client Error" or "Entry Not Found"
- **Solution:** The model ID is incorrect. Use `runwayml/stable-diffusion-inpainting` for testing

### Error: "GCS not configured"
- **Solution:** Make sure `GCS_SERVICE_ACCOUNT_PATH` points to a valid JSON file

