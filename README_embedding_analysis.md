# Embedding Analysis for Spatial Invariance

This analysis suite tests whether your trained ViT models have achieved spatial invariance in their patch embeddings. Specifically, it compares the `base` (standard patch embedding) vs `base_patch_conv` (convolutional patch embedding) architectures.

## Quick Start

### 1. Test Locally First
```bash
# Test data generation only
python run_analysis_test.py --test data

# Test model loading only  
python run_analysis_test.py --test models

# Run full analysis (small scale)
python run_analysis_test.py --test analysis
```

### 2. Submit Full Analysis
```bash
sbatch run_embedding_analysis.sbatch
```

## What It Tests

### 1. Pairwise Similarity
- **Question**: Do patches from the same semantic class (e.g., two different sphere patches) have higher cosine similarity than random patches?
- **Expected**: `base_patch_conv` should show higher sphere-sphere similarity than `base`

### 2. Linear Probes  
- **Semantic Probe**: Can a linear classifier predict patch content (background/membrane/sphere/cube) from embeddings?
- **Coordinate Probe**: Can a linear classifier predict patch spatial location from embeddings?
- **Expected**: High semantic accuracy, low coordinate accuracy = good spatial invariance

### 3. Translation Sensitivity
- **Question**: When we translate (shift) the volume, how much do embeddings of the "same" patch change?
- **Expected**: `base_patch_conv` should be less sensitive to translations

## Understanding Results

### Good Spatial Invariance Signs:
- **Higher sphere-sphere similarity** vs random pairs
- **High semantic accuracy**, low coordinate accuracy in probes  
- **Low translation sensitivity** scores
- **Similar results** with and without positional encoding

### Poor Spatial Invariance Signs:
- **No difference** between semantic and random similarities
- **High coordinate accuracy** (location still encoded)
- **High translation sensitivity** 
- **Big drop** in semantic performance when position encoding added

## File Outputs

- `analysis_results/analysis_results.json` - Raw numerical results
- `analysis_results/pairwise_similarity_{model}.png` - Similarity histograms
- `analysis_results/linear_probes_{model}.png` - Probe accuracy comparison
- `analysis_results/test_data.pkl` - Generated test volumes (can reuse)

## Model Loading Details

The analysis automatically:
- Loads models from `checkpoints/` directory
- Applies `channels_last_3d` memory format (matching training)
- Disables `torch.compile` to access internal embeddings
- Uses fp32 precision (models saved in fp32 despite bf16 training)
- Looks for patterns like `best_model_*base*.pt`, `best_model_*base_patch_conv*.pt`

## Troubleshooting

### "No checkpoint found"
- Check that model files exist in `checkpoints/`
- Ensure filenames contain `base` or `base_patch_conv`
- Try running `ls checkpoints/*.pt` to see available files

### "CUDA out of memory"  
- Reduce `--num_volumes` or `--num_pairs` in the sbatch script
- Models are loaded sequentially to save memory

### "No sphere patches found"
- The generator creates 8 combinations of structures
- Some volumes may have no spheres - this is normal
- Analysis skips volumes without the required structures

## Expected Runtime
- **Local test**: ~2-5 minutes  
- **Full analysis**: ~30-60 minutes on GPU
- **Memory usage**: ~4-8GB GPU, ~16GB RAM

## Next Steps

If `base_patch_conv` doesn't show clear spatial invariance:

1. **Check positional encoding**: Try relative vs absolute position
2. **Examine attention patterns**: See if heads focus on location vs content  
3. **Try deeper analysis**: Use CKA or mutual information methods
4. **Consider training changes**: More translation augmentation during pretraining 