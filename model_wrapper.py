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
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}, FP16: {use_fp16}")
        
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Check if model_path is a local path or Hugging Face ID
        is_local = os.path.exists(model_path) or os.path.isdir(model_path)
        
        # Detect if this is CatVTON model
        self.is_catvton = False
        if is_local:
            # Check for CatVTON-specific files/directories
            catvton_indicators = [
                os.path.join(model_path, "CatVTON"),
                os.path.join(model_path, "catvton"),
                os.path.join(model_path, "inference.py"),
                os.path.join(model_path, "models"),
            ]
            if any(os.path.exists(indicator) for indicator in catvton_indicators):
                self.is_catvton = True
                print("⚠️  CatVTON model detected! Attempting to load...")
                print("   Note: Full CatVTON integration requires the CatVTON repository.")
                print("   See CATVTON_SETUP.md for setup instructions.")
        
        # Try to load CatVTON if detected
        if self.is_catvton:
            try:
                # Try importing CatVTON's inference module
                import sys
                catvton_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
                if catvton_dir not in sys.path:
                    sys.path.insert(0, catvton_dir)
                
                # Attempt to use CatVTON's actual implementation
                # This will work if CatVTON is properly installed
                try:
                    from inference import CatVTONInference  # CatVTON's inference class
                    self.catvton_model = CatVTONInference(model_path, device=device)
                    print("✅ CatVTON model loaded successfully!")
                    return
                except ImportError:
                    print("⚠️  CatVTON inference module not found. Using fallback method.")
                    self.is_catvton = False
            except Exception as e:
                print(f"⚠️  Failed to load CatVTON: {e}")
                print("   Falling back to inpainting model...")
                self.is_catvton = False
        
        # Fallback: Load as standard inpainting pipeline
        print("Loading as Stable Diffusion Inpainting model...")
        try:
            if is_local:
                print(f"Loading from local path: {model_path}")
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True
                )
            else:
                print(f"Loading from Hugging Face: {model_path}")
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            
            if "404" in error_msg or "Entry Not Found" in error_msg:
                raise ValueError(
                    f"Model not found: {model_path}\n\n"
                    "For CatVTON model:\n"
                    "1. Clone: git clone https://github.com/Zheng-Chong/CatVTON.git\n"
                    "2. Install dependencies and download weights\n"
                    "3. Set MODEL_PATH to CatVTON directory\n"
                    "See CATVTON_SETUP.md for details.\n\n"
                    "For testing (fallback):\n"
                    "- runwayml/stable-diffusion-inpainting\n"
                    "- CompVis/stable-diffusion-inpainting"
                ) from e
            else:
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
        # If using actual CatVTON model, use its inference method
        if hasattr(self, 'catvton_model') and self.catvton_model is not None:
            print("Using CatVTON concatenation-based inference...")
            return self._catvton_inference(
                person_image_path, garment_image_path, output_path,
                num_inference_steps, guidance_scale, garment_type
            )
        
        # Fallback: Use inpainting-based approach
        print("Using inpainting-based inference (fallback method)...")
        return self._inpainting_inference(
            person_image_path, garment_image_path, output_path,
            num_inference_steps, guidance_scale, strength, garment_type
        )
    
    def _catvton_inference(
        self,
        person_image_path: str,
        garment_image_path: str,
        output_path: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        garment_type: str
    ) -> Image.Image:
        """CatVTON concatenation-based inference."""
        # Load images
        person_image = Image.open(person_image_path).convert("RGB")
        garment_image = Image.open(garment_image_path).convert("RGB")
        
        # CatVTON uses concatenation: person + garment side by side
        # Resize to CatVTON's expected size (typically 768x1024 or similar)
        target_width = 768
        target_height = 1024
        
        person_image = person_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        garment_image = garment_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Concatenate horizontally: [person | garment]
        combined_image = Image.new("RGB", (target_width * 2, target_height))
        combined_image.paste(person_image, (0, 0))
        combined_image.paste(garment_image, (target_width, 0))
        
        # Use CatVTON's inference method
        result = self.catvton_model.inference(
            combined_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        if output_path:
            result.save(output_path)
        
        return result
    
    def _inpainting_inference(
        self,
        person_image_path: str,
        garment_image_path: str,
        output_path: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        garment_type: str
    ) -> Image.Image:
        """Fallback inpainting-based inference (for testing)."""
        # Load images
        person_image = Image.open(person_image_path).convert("RGB")
        garment_image = Image.open(garment_image_path).convert("RGB")
        
        # Resize images to standard size (512x512 for SD models)
        target_size = (512, 512)
        person_image = person_image.resize(target_size, Image.Resampling.LANCZOS)
        garment_image = garment_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Create a mask for the garment area on the person
        mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        
        if garment_type == "upper":
            # Mask upper body area (approximately top 60% of image)
            mask_height = int(target_size[1] * 0.6)
            mask[0:mask_height, :] = 255
            
            # Create elliptical mask for more natural shape
            y, x = np.ogrid[:mask_height, :target_size[0]]
            center_x = target_size[0] // 2
            center_y = mask_height // 2
            
            ellipse_mask = ((x - center_x) ** 2 / (target_size[0] * 0.4) ** 2 + 
                          (y - center_y) ** 2 / (mask_height * 0.5) ** 2) <= 1
            mask[0:mask_height, :] = np.where(ellipse_mask, 255, 0).astype(np.uint8)
            
        elif garment_type == "lower":
            # Mask lower body area
            mask_height = int(target_size[1] * 0.6)
            mask_start = target_size[1] - mask_height
            mask[mask_start:, :] = 255
        else:
            mask[int(target_size[1] * 0.2):int(target_size[1] * 0.8), :] = 255
        
        # Apply Gaussian blur to mask edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_image = Image.fromarray(mask)
        
        # Prepare prompts
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
        
        if output_path:
            result.save(output_path)
        
        return result
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

