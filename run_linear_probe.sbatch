#!/bin/bash
#SBATCH --job-name=mae_layer_probe
#SBATCH --output=logs/layer_probe_%j.out
#SBATCH --error=logs/layer_probe_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h200:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@120

module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3

source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB

export PYTHONFAULTHANDLER=1          # Python will dump a traceback on segfault
export PYTHONUNBUFFERED=1
VOLUME_DIR=/gpfs/radev/home/am3833/scratch/volumes_96
MASK_DIR=/gpfs/radev/home/am3833/scratch/volumes_96/masks
CKPT=checkpoints/best_model_base_patch_conv_v4_85_stable_eps1e-5.pt

python3 -u linear_probe.py \
    --shard_dir "$VOLUME_DIR" \
    --mask_dir  "$MASK_DIR" \
    --ckpt "$CKPT" \
    --arch base_patch_conv \
    --layers -1 -2 0 2 4 6 8 10 11 \
    --bs 16 \
    --steps 2000 \
    --amp \
    --project mae-layer-probes \
    --run probes_v2_linear \
    --probe linear \
    --coord_weights 10 1 0     \
    --calibrate                \
    --noposition  

echo "=== Done ==="
