# CatVTON Virtual Try-On Service

A FastAPI-based service for virtual try-on using the CatVTON model, optimized for GPU-enabled RunPod instances.

## Features

- **GPU-Accelerated**: Optimized for NVIDIA CUDA with FP16 precision
- **GCS Integration**: Seamless download/upload from Google Cloud Storage
- **API Key Authentication**: Simple header-based authentication
- **Dockerized**: Ready-to-deploy Docker container for RunPod

## Prerequisites

- NVIDIA GPU with CUDA support
- Google Cloud Storage service account JSON file
- RunPod instance (Ubuntu/CUDA)

## Setup

### 1. Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `API_KEY`: Your secret API key for authentication
- `GCS_SERVICE_ACCOUNT_PATH`: Path to your GCS service account JSON file
- `MODEL_PATH`: (Optional) Custom model path or Hugging Face model ID

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Docker Deployment

```bash
# Build the image
docker build -t cat-tryon-service .

# Run the container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e API_KEY=your-api-key \
  -e GCS_SERVICE_ACCOUNT_PATH=/path/to/service-account.json \
  -v /path/to/service-account.json:/path/to/service-account.json:ro \
  cat-tryon-service
```

### 4. RunPod Deployment

1. Upload your code to RunPod
2. Set environment variables in RunPod dashboard
3. Mount your GCS service account file
4. Deploy with GPU enabled

## API Usage

### Authentication

Include the API key in the request header:
```
X-API-Key: your-secret-api-key-here
```

### Try-On Endpoint

**POST** `/tryon`

Request body:
```json
{
  "person_image_uri": "gs://bucket-name/path/to/person.jpg",
  "garment_image_uri": "gs://bucket-name/path/to/garment.jpg",
  "output_uri": "gs://bucket-name/path/to/output.png",  // Optional
  "num_inference_steps": 50,  // Optional, default: 50
  "guidance_scale": 7.5,  // Optional, default: 7.5
  "strength": 0.8  // Optional, default: 0.8
}
```

Response:
```json
{
  "success": true,
  "output_image_uri": "gs://bucket-name/path/to/output_tryon.png",
  "message": "Try-on completed successfully"
}
```

### Health Check

**GET** `/health`

Returns service health status including model and GPU availability.

## Example cURL Request

```bash
curl -X POST "http://localhost:8000/tryon" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "person_image_uri": "gs://my-bucket/person.jpg",
    "garment_image_uri": "gs://my-bucket/garment.jpg"
  }'
```

## Project Structure

```
cat-tryon-service/
├── main.py              # FastAPI application
├── model_wrapper.py     # CatVTON model wrapper
├── gcs_utils.py         # GCS download/upload utilities
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Notes

- The service automatically downloads images from GCS, processes them, and uploads results
- Temporary files are cleaned up after each request
- The model uses FP16 precision for efficiency on GPU
- Adjust `num_inference_steps`, `guidance_scale`, and `strength` based on your quality/speed requirements

## Troubleshooting

1. **Model loading errors**: Ensure the model path is correct and accessible
2. **GCS errors**: Verify service account JSON path and permissions
3. **CUDA errors**: Check GPU availability with `nvidia-smi`
4. **Memory issues**: Reduce batch size or use model CPU offloading

