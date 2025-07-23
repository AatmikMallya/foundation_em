#!/usr/bin/env python3
"""
Quick test runner for embedding analysis
========================================
Run a smaller version of the analysis locally to test the code
before submitting the full job to SLURM.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def test_analysis():
    """Run a quick test of the embedding analysis."""
    from analyze_embeddings import main, generate_test_data, load_models
    
    print("=== Testing Embedding Analysis ===")
    
    # Create a minimal test args object
    class TestArgs:
        def __init__(self):
            self.output_dir = "test_analysis_results"
            self.checkpoint_dir = "checkpoints"
            self.data_path = None
            self.save_data = True
            self.num_volumes = 10  # Small number for testing
            self.volume_size = 96
            self.test_similarity = True
            self.test_probes = True
            self.test_translation = True
            self.num_pairs = 100  # Smaller for testing
            self.num_translations = 20  # Smaller for testing
    
    args = TestArgs()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    try:
        print("Starting test analysis...")
        main(args)
        print("✅ Test completed successfully!")
        
        # Check results
        results_path = Path(args.output_dir) / 'analysis_results.json'
        if results_path.exists():
            print(f"✅ Results saved to {results_path}")
            
            # Print file sizes
            import json
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("\n=== Quick Results Summary ===")
            for model_name in results.keys():
                print(f"\n{model_name.upper()} MODEL:")
                model_results = results[model_name]
                
                if 'linear_probes' in model_results:
                    probes = model_results['linear_probes']
                    if 'raw' in probes:
                        print(f"  Semantic accuracy (raw): {probes['raw']['semantic_accuracy']:.3f}")
                        print(f"  Coordinate accuracy (raw): {probes['raw']['coordinate_accuracy']:.3f}")
                
                if 'pairwise_similarity' in model_results:
                    sim = model_results['pairwise_similarity']
                    if 'raw' in sim:
                        import numpy as np
                        sphere_mean = np.mean(sim['raw']['sphere_sphere'])
                        random_mean = np.mean(sim['raw']['random_random'])
                        print(f"  Sphere-sphere similarity: {sphere_mean:.3f}")
                        print(f"  Random-random similarity: {random_mean:.3f}")
                        print(f"  Similarity boost: {sphere_mean - random_mean:.3f}")
        else:
            print("❌ Results file not found")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_data_generation():
    """Test just the data generation part."""
    print("=== Testing Data Generation ===")
    
    try:
        from analyze_embeddings import generate_test_data
        
        print("Generating 3 test volumes...")
        volumes, masks = generate_test_data(num_volumes=3, volume_size=96)
        
        print(f"✅ Generated {len(volumes)} volumes")
        print(f"Volume shape: {volumes[0].shape}")
        print(f"Mask shape: {masks[0].shape}")
        print(f"Volume range: [{volumes[0].min():.3f}, {volumes[0].max():.3f}]")
        print(f"Mask labels: {set(masks[0].flatten())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading."""
    print("=== Testing Model Loading ===")
    
    try:
        from analyze_embeddings import load_models
        from vit_3d import get_device
        
        device = get_device()
        print(f"Using device: {device}")
        
        models = load_models("checkpoints", device)
        
        for name, model in models.items():
            print(f"✅ Loaded {name} model")
            print(f"   Device: {next(model.parameters()).device}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test embedding analysis")
    parser.add_argument("--test", choices=['all', 'data', 'models', 'analysis'], 
                        default='all', help="What to test")
    
    args = parser.parse_args()
    
    success = True
    
    if args.test in ['all', 'data']:
        success &= test_data_generation()
        print()
    
    if args.test in ['all', 'models']:
        success &= test_model_loading()
        print()
    
    if args.test in ['all', 'analysis']:
        success &= test_analysis()
        print()
    
    if success:
        print("🎉 All tests passed! Ready to submit to SLURM.")
        exit(0)
    else:
        print("💥 Some tests failed. Check the errors above.")
        exit(1) 