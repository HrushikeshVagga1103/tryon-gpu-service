# Image Requirements for Virtual Try-On

To get the best results from the virtual try-on service, follow these guidelines for preparing your images.

## Person Image Requirements

### ✅ Best Practices

1. **Full Body View**
   - Person should be standing upright
   - Full body visible from head to at least mid-thigh
   - Front-facing pose (facing the camera directly)

2. **Pose**
   - Arms at sides or slightly away from body
   - Standing straight, not leaning
   - Feet visible or at least lower legs visible
   - Avoid crossed arms or hands covering the torso

3. **Background**
   - Plain, solid color background (white, gray, or black works best)
   - Minimal distractions
   - No complex patterns or objects behind the person

4. **Lighting**
   - Even, natural lighting
   - Avoid harsh shadows on the person
   - No overexposure or underexposure

5. **Image Quality**
   - High resolution (minimum 512x512, preferably 1024x1024 or higher)
   - Clear, sharp image
   - Good contrast between person and background

### ❌ Avoid

- Side profiles or angled poses
- Person sitting or in unusual poses
- Busy backgrounds with patterns or objects
- Poor lighting with heavy shadows
- Blurry or low-resolution images
- Person wearing loose/baggy clothing that obscures body shape

## Garment Image Requirements

### ✅ Best Practices

1. **Flat Lay Presentation**
   - Garment laid flat on a surface
   - Full garment visible (all parts of the shirt/pants)
   - No folds or wrinkles that obscure the design
   - Garment should be spread out, not bunched up

2. **Background**
   - Plain, solid color background (white, gray, or black)
   - High contrast with garment color
   - No patterns or textures in background

3. **Orientation**
   - Garment should be oriented correctly (right-side up)
   - Front of garment facing up
   - For shirts: collar at top, hem at bottom
   - For pants: waistband at top, cuffs at bottom

4. **Lighting**
   - Even lighting across the entire garment
   - No shadows on the garment
   - Colors should be accurate and vibrant

5. **Image Quality**
   - High resolution (minimum 512x512, preferably 1024x1024 or higher)
   - Sharp, clear image
   - All details visible (patterns, logos, textures)

### ❌ Avoid

- Garment hanging on a hanger
- Garment being worn by a model
- Wrinkled or folded garment
- Busy backgrounds
- Poor lighting with shadows
- Low resolution or blurry images
- Only partial garment visible

## Example Good Images

### Person Image Example:
```
✓ Full body, front-facing
✓ Standing upright
✓ Plain white background
✓ Even lighting
✓ Arms at sides
✓ High resolution
```

### Garment Image Example:
```
✓ Flat lay on plain background
✓ Full garment visible
✓ Right-side up orientation
✓ Even lighting
✓ High resolution
✓ No wrinkles or folds
```

## API Usage

When calling the API, specify the garment type:

```json
{
  "person_image_uri": "gs://bucket/person.jpg",
  "garment_image_uri": "gs://bucket/garment.jpg",
  "garment_type": "upper"  // or "lower" for pants
}
```

- `"upper"`: For shirts, t-shirts, tops, jackets (default)
- `"lower"`: For pants, jeans, shorts, skirts

## Tips for Better Results

1. **Consistent Image Sizes**: Both images should be similar in dimensions
2. **Color Contrast**: Ensure good contrast between garment and background
3. **Garment Type**: Always specify correct `garment_type` parameter
4. **Inference Parameters**: 
   - For better quality: `num_inference_steps: 75-100`, `strength: 0.7-0.9`
   - For faster processing: `num_inference_steps: 30-50`, `strength: 0.6-0.8`

## Troubleshooting Poor Results

If results are not good, check:

1. ✅ Person image is full-body and front-facing
2. ✅ Garment image is flat-laid on plain background
3. ✅ Both images are high resolution and clear
4. ✅ Backgrounds are plain and uncluttered
5. ✅ Lighting is even in both images
6. ✅ Correct `garment_type` is specified
7. ✅ Try adjusting `strength` parameter (0.6-0.9 range)
8. ✅ Try increasing `num_inference_steps` (50-100)

## Model Limitations

**Note**: The current implementation uses Stable Diffusion Inpainting, which is a general-purpose model. For best results, consider:

1. Using a dedicated CatVTON model if available
2. Pre-processing images to ensure they meet requirements
3. Post-processing results if needed
4. Fine-tuning inference parameters for your specific use case

