# Setting Up CatVTON Model

## Why Use CatVTON?

The actual CatVTON model is specifically designed for virtual try-on tasks and will produce **much better results** than generic Stable Diffusion Inpainting models.

CatVTON:
- ✅ Specifically trained for virtual try-on
- ✅ Uses concatenation-based architecture (person + garment)
- ✅ Lightweight (~899M parameters, <8GB VRAM)
- ✅ Produces realistic garment transfer results

## Installation Options

### Option 1: Clone CatVTON Repository (Recommended)

1. **Clone the official CatVTON repository:**
   ```bash
   cd /workspace
   git clone https://github.com/Zheng-Chong/CatVTON.git
   cd CatVTON
   ```

2. **Install CatVTON dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained weights:**
   - Follow the instructions in the CatVTON repository
   - Download the model checkpoints
   - Place them in the appropriate directory

4. **Set MODEL_PATH environment variable:**
   ```bash
   export MODEL_PATH=/workspace/CatVTON
   ```
   Or point to the specific model checkpoint directory.

### Option 2: Use Hugging Face (If Available)

If CatVTON is available on Hugging Face:
```bash
export MODEL_PATH=Zheng-Chong/CatVTON
```

### Option 3: Local Model Directory

If you have CatVTON model files locally:
```bash
export MODEL_PATH=/path/to/catvton/model
```

## Model Architecture

CatVTON uses a **concatenation-based approach**:
- Person image and garment image are concatenated along spatial dimensions
- No need for complex preprocessing or encoding modules
- Direct concatenation → Diffusion model → Output

This is different from inpainting models which require masks.

## Integration with FastAPI Service

The updated `model_wrapper.py` now supports:
1. **CatVTON models** (if properly installed)
2. **Fallback to inpainting models** (for testing)

To use CatVTON:
1. Install CatVTON following Option 1 above
2. Set `MODEL_PATH` to point to CatVTON
3. Restart your FastAPI service

## Verification

After setup, check that CatVTON is loaded:
```bash
# Check service logs - should show:
# "Loading CatVTON model from: /workspace/CatVTON"
# "Model loaded successfully!"
```

## Troubleshooting

### Model Not Found
- Verify the path in `MODEL_PATH` is correct
- Check that CatVTON repository is cloned and dependencies installed
- Ensure model checkpoints are downloaded

### Import Errors
- Install all CatVTON dependencies: `pip install -r requirements.txt` (in CatVTON directory)
- Check Python path includes CatVTON directory

### Memory Issues
- CatVTON requires <8GB VRAM for high-res images
- If issues, reduce image resolution in preprocessing

## References

- **GitHub Repository**: https://github.com/Zheng-Chong/CatVTON
- **Paper**: https://arxiv.org/abs/2407.15886
- **Model Weights**: Check the repository for download links

