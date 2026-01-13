# Testing Guide

## Accessing Swagger UI

FastAPI automatically provides Swagger UI documentation. Once your service is running:

### On RunPod with Public URL

1. **Get your RunPod public URL:**
   - In RunPod dashboard, find your pod's public URL
   - It will look like: `https://xxxxx-8000.proxy.runpod.net`

2. **Access Swagger UI:**
   ```
   https://xxxxx-8000.proxy.runpod.net/docs
   ```

3. **Alternative - ReDoc:**
   ```
   https://xxxxx-8000.proxy.runpod.net/redoc
   ```

### Local Testing

If testing locally:
```
http://localhost:8000/docs
```

## Using Swagger UI

1. Open the Swagger UI page (`/docs`)
2. Click on the `/tryon` endpoint to expand it
3. Click "Try it out"
4. Fill in the request body:
   ```json
   {
     "person_image_uri": "gs://your-bucket/person.jpg",
     "garment_image_uri": "gs://your-bucket/garment.jpg"
   }
   ```
5. Add the API key in the "Authorize" button at the top, or manually add header:
   - Click "Authorize" button
   - Enter your API key
   - Or add `X-API-Key` header manually
6. Click "Execute"
7. View the response

## Testing Endpoints

### 1. Health Check (No Auth Required)

```bash
curl http://localhost:8000/health
```

Or via public URL:
```bash
curl https://xxxxx-8000.proxy.runpod.net/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gcs_configured": true,
  "cuda_available": true
}
```

### 2. Root Endpoint (No Auth Required)

```bash
curl http://localhost:8000/
```

### 3. Try-On Endpoint (Auth Required)

```bash
curl -X POST "http://localhost:8000/tryon" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "person_image_uri": "gs://your-bucket/path/to/person.jpg",
    "garment_image_uri": "gs://your-bucket/path/to/garment.jpg",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "strength": 0.8
  }'
```

Or via public URL:
```bash
curl -X POST "https://xxxxx-8000.proxy.runpod.net/tryon" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "person_image_uri": "gs://your-bucket/path/to/person.jpg",
    "garment_image_uri": "gs://your-bucket/path/to/garment.jpg"
  }'
```

Expected response:
```json
{
  "success": true,
  "output_image_uri": "gs://your-bucket/path/to/person_tryon.png",
  "message": "Try-on completed successfully"
}
```

## Test Script

See `test_api.py` for a Python test script.

## Troubleshooting

### Swagger UI Not Loading
- Check that the service is running: `curl http://localhost:8000/health`
- Verify the port is correct (default: 8000)
- Check RunPod firewall/network settings

### 401 Unauthorized
- Verify your `API_KEY` environment variable matches the header value
- Check that you're sending `X-API-Key` header (not `Authorization`)

### 404 Not Found
- Ensure GCS URIs are correct and files exist
- Verify GCS service account has read/write permissions

### 500 Internal Server Error
- Check service logs for detailed error messages
- Verify model is loaded: check `/health` endpoint
- Ensure GPU is available if using CUDA

