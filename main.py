"""
FastAPI application for CatVTON Virtual Try-On service.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from gcs_utils import GCSUtils
from model_wrapper import CatVTONPredictor

# Load environment variables
load_dotenv()

app = FastAPI(
    title="CatVTON Virtual Try-On Service",
    description="API for virtual try-on using CatVTON model",
    version="1.0.0"
)

# Configuration
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
GCS_SERVICE_ACCOUNT_PATH = os.getenv("GCS_SERVICE_ACCOUNT_PATH")
MODEL_PATH = os.getenv("MODEL_PATH", None)
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/cat-tryon")

# Initialize components
gcs_utils = None
predictor = None


def get_api_key(x_api_key: str = Header(...)) -> str:
    """Dependency to validate API key."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


class TryOnRequest(BaseModel):
    """Request model for try-on endpoint."""
    person_image_uri: str  # GCS URI for person image
    garment_image_uri: str  # GCS URI for garment image
    output_uri: Optional[str] = None  # Optional output GCS URI
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    strength: Optional[float] = 0.8


class TryOnResponse(BaseModel):
    """Response model for try-on endpoint."""
    success: bool
    output_image_uri: str
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global gcs_utils, predictor
    
    print("Initializing CatVTON service...")
    
    # Initialize GCS utils
    gcs_utils = GCSUtils(service_account_path=GCS_SERVICE_ACCOUNT_PATH)
    print("GCS utils initialized")
    
    # Initialize model
    try:
        predictor = CatVTONPredictor(
            model_path=MODEL_PATH,
            use_fp16=True
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Create temp directory
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Temp directory created: {TEMP_DIR}")
    
    print("Service ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global predictor
    if predictor:
        del predictor
    print("Service shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CatVTON Virtual Try-On",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    import subprocess
    cuda_available = False
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        cuda_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "gcs_configured": gcs_utils is not None,
        "cuda_available": cuda_available
    }


@app.post("/tryon", response_model=TryOnResponse)
async def try_on(
    request: TryOnRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Process virtual try-on request.
    
    Args:
        request: TryOnRequest with person and garment image URIs
        api_key: API key for authentication
    
    Returns:
        TryOnResponse with output image GCS URI
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not gcs_utils:
        raise HTTPException(status_code=503, detail="GCS not configured")
    
    # Create temporary directory for this request
    request_temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    
    try:
        # Define local file paths
        person_local_path = os.path.join(request_temp_dir, "person.jpg")
        garment_local_path = os.path.join(request_temp_dir, "garment.jpg")
        output_local_path = os.path.join(request_temp_dir, "output.png")
        
        # Download images from GCS
        print(f"Downloading person image from: {request.person_image_uri}")
        gcs_utils.download_file(request.person_image_uri, person_local_path)
        
        print(f"Downloading garment image from: {request.garment_image_uri}")
        gcs_utils.download_file(request.garment_image_uri, garment_local_path)
        
        # Run inference
        print("Running inference...")
        result_image = predictor.inference(
            person_image_path=person_local_path,
            garment_image_path=garment_local_path,
            output_path=output_local_path,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength
        )
        
        # Determine output URI
        if request.output_uri:
            output_uri = request.output_uri
        else:
            # Generate output URI based on person image URI
            output_uri = gcs_utils.generate_output_uri(
                request.person_image_uri,
                suffix="_tryon"
            )
        
        # Upload result to GCS
        print(f"Uploading result to: {output_uri}")
        gcs_utils.upload_file(output_local_path, output_uri, content_type="image/png")
        
        return TryOnResponse(
            success=True,
            output_image_uri=output_uri,
            message="Try-on completed successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        print(f"Error during try-on processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if os.path.exists(request_temp_dir):
            shutil.rmtree(request_temp_dir)
            print(f"Cleaned up temp directory: {request_temp_dir}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

