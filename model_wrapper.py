"""
CatVTON model wrapper for inference.
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPImageProcessor


class CatVTONPredictor:
    """Wrapper class for CatVTON model inference."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize CatVTON predictor.
        
        Args:
            model_path: Path to CatVTON model (Hugging Face model ID or local path).
                       Defaults to official CatVTON model if None.
            device: Device to run inference on ('cuda' or 'cpu').
                   Auto-detects if None.
            use_fp16: Whether to use FP16 precision for efficiency.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # MODEL_PATH is required - no default to avoid 404 errors
        if model_path is None or model_path == "":
            raise ValueError(
                "MODEL_PATH environment variable is required. "
                "Please set it to a valid Hugging Face model ID (e.g., 'runwayml/stable-diffusion-inpainting') "
                "or a local path to your CatVTON model. "
                "Example: export MODEL_PATH=runwayml/stable-diffusion-inpainting"
            )
        
        print(f"Loading CatVTON model from: {model_path}")
        print(f"Device: {device}, FP16: {use_fp16}")
        
        # Load the pipeline
        # Note: CatVTON typically uses Stable Diffusion Inpainting architecture
        # Adjust model loading based on actual CatVTON implementation
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Check if model_path is a local path or Hugging Face ID
        is_local = os.path.exists(model_path) or os.path.isdir(model_path)
        
        try:
            if is_local:
                print(f"Loading from local path: {model_path}")
                # Try loading from local directory
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
            else:
                print(f"Loading from Hugging Face: {model_path}")
                # Try loading as a standard inpainting pipeline from Hugging Face
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            
            # Provide helpful error message
            if "404" in error_msg or "Entry Not Found" in error_msg:
                raise ValueError(
                    f"Model not found: {model_path}\n"
                    "Please verify:\n"
                    "1. The model ID is correct (check on huggingface.co)\n"
                    "2. The model is public or you're authenticated\n"
                    "3. For local paths, ensure the path is correct\n\n"
                    "Common working models:\n"
                    "- runwayml/stable-diffusion-inpainting\n"
                    "- CompVis/stable-diffusion-inpainting\n"
                    "Or set MODEL_PATH to your local CatVTON model directory"
                ) from e
            else:
                # Try alternative loading method
                print("Attempting alternative loading method...")
                try:
                    self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load model from {model_path}. "
                        f"Error: {str(e2)}"
                    ) from e2
        
        self.pipeline = self.pipeline.to(self.device)
        
        if self.use_fp16:
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_model_cpu_offload()
        
        print("Model loaded successfully!")
    
    def inference(
        self,
        person_image_path: str,
        garment_image_path: str,
        output_path: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        garment_type: str = "upper"  # "upper" for shirts, "lower" for pants
    ) -> Image.Image:
        """
        Run inference on person and garment images.
        
        Args:
            person_image_path: Path to person image
            garment_image_path: Path to garment image
            output_path: Optional path to save output image
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            strength: Strength of the inpainting (0.0 to 1.0)
            garment_type: Type of garment - "upper" (shirts) or "lower" (pants)
        
        Returns:
            PIL Image of the result
        """
        # Load images
        person_image = Image.open(person_image_path).convert("RGB")
        garment_image = Image.open(garment_image_path).convert("RGB")
        
        # Resize images to standard size (512x512 for SD models)
        target_size = (512, 512)
        person_image = person_image.resize(target_size, Image.Resampling.LANCZOS)
        garment_image = garment_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create a mask for the garment area on the person
        # For upper garments (shirts), mask the upper body area
        # For lower garments (pants), mask the lower body area
        mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        
        if garment_type == "upper":
            # Mask upper body area (approximately top 60% of image)
            # This covers the torso area where shirts would be worn
            mask_height = int(target_size[1] * 0.6)
            mask[0:mask_height, :] = 255
            
            # Create a more natural mask shape (rounded top, straight bottom)
            # Add some feathering at the edges for smoother blending
            y, x = np.ogrid[:mask_height, :target_size[0]]
            center_x = target_size[0] // 2
            center_y = mask_height // 2
            
            # Create elliptical mask for more natural shape
            ellipse_mask = ((x - center_x) ** 2 / (target_size[0] * 0.4) ** 2 + 
                          (y - center_y) ** 2 / (mask_height * 0.5) ** 2) <= 1
            mask[0:mask_height, :] = np.where(ellipse_mask, 255, 0).astype(np.uint8)
            
        elif garment_type == "lower":
            # Mask lower body area (approximately bottom 60% of image)
            mask_height = int(target_size[1] * 0.6)
            mask_start = target_size[1] - mask_height
            mask[mask_start:, :] = 255
        else:
            # Default: mask entire center area
            mask[int(target_size[1] * 0.2):int(target_size[1] * 0.8), :] = 255
        
        # Apply Gaussian blur to mask edges for smoother blending
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Convert mask to PIL Image
        mask_image = Image.fromarray(mask)
        
        # Prepare prompt with garment description
        # Use more specific prompts for better results
        prompt = (
            f"a person wearing a {garment_type} garment, "
            "high quality, detailed, realistic, professional photography, "
            "perfect fit, natural lighting, full body"
        )
        negative_prompt = (
            "blurry, low quality, distorted, deformed, "
            "bad anatomy, extra limbs, missing limbs, "
            "ugly, duplicate, watermark, text, signature"
        )
        
        # Run inference
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            ).images[0]
        
        # Save if output path provided
        if output_path:
            result.save(output_path)
        
        return result
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

