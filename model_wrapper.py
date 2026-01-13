"""
CatVTON model wrapper for inference.
"""
import os
import torch
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
        
        # Default to official CatVTON model if not specified
        if model_path is None:
            model_path = "levihsu/OOTDiffusion"  # Common CatVTON-style model
        
        print(f"Loading CatVTON model from: {model_path}")
        print(f"Device: {device}, FP16: {use_fp16}")
        
        # Load the pipeline
        # Note: CatVTON typically uses Stable Diffusion Inpainting architecture
        # Adjust model loading based on actual CatVTON implementation
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        try:
            # Try loading as a standard inpainting pipeline
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        except Exception as e:
            print(f"Error loading model as standard pipeline: {e}")
            print("Attempting alternative loading method...")
            # Fallback: try loading with different parameters
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )
        
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
        strength: float = 0.8
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
        
        # CatVTON typically uses concatenation approach:
        # Combine person and garment images as input
        # For inpainting-style models, we create a mask and combine images
        
        # Create a simple mask (can be adjusted based on actual CatVTON implementation)
        # In typical CatVTON, the garment area is masked out on the person image
        import numpy as np
        mask = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
        
        # For CatVTON, we typically concatenate person and garment
        # This is a simplified version - adjust based on actual model requirements
        combined_image = Image.new("RGB", (target_size[0] * 2, target_size[1]))
        combined_image.paste(person_image, (0, 0))
        combined_image.paste(garment_image, (target_size[0], 0))
        
        # Convert mask to PIL Image
        mask_image = Image.fromarray(mask)
        
        # Prepare prompt (CatVTON may use specific prompts)
        prompt = "a person wearing the garment, high quality, detailed"
        negative_prompt = "blurry, low quality, distorted, deformed"
        
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

