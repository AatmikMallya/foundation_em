# `train_helper.py`
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from itertools import product
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
import random
import glob
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import importlib
import pickle
import os
import torch
import torch.nn as nn
import pandas as pd
from time import time
from dataset import MicrotubuleDataset
import platform
import datetime
from monai.networks.nets import UNet, SwinUNETR, SegResNet, DynUNet, AttentionUnet, BasicUNet, VNet, SegResNetDS, SegResNetVAE, HighResNet, BasicUNetPlusPlus
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    OneOf,
)

hyperparameters = {
    'experiment_name': '',
    # Data parameters
    'image_dir': 'training/subvols/image',
    'label_dir': 'training/labeled_binary_large',
    'checkpoint_dir': 'checkpoints',
    # Training parameters
    'batch_size': 1,
    'num_epochs': 500,
    'learning_rate': 5e-4,
    'weight_decay': 2e-4,
    'num_workers': 0,
    'random_seed': 42,
    'patience': 50,
    'pos_weight': 1.0,  # Weight for microtubule pixels in loss function
    'device': 'cpu',
    # Model parameters
    'spatial_dims': 3,
    'in_channels': 1,
    'out_channels': 1,
    'kernel_size': 3,
    'network_channels': (32, 64, 128, 256), #(16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256, 512)
    'strides': (2, 2, 2),
    'num_res_units': 6,
    'dropout': 0.2,
    'freeze_encoder': False,
    # System parameters
    'torch_version': torch.__version__,
    'numpy_version': np.__version__,
    'python_version': platform.python_version(),
    'system_platform': platform.system(),
    # Time info
    'training_start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

EXPERIMENT_NAME = hyperparameters['experiment_name']
IMAGE_DIR = hyperparameters['image_dir']
LABEL_DIR = hyperparameters['label_dir']
CHECKPOINT_DIR = hyperparameters['checkpoint_dir']

# Training parameters
BATCH_SIZE = hyperparameters['batch_size']
NUM_EPOCHS = hyperparameters['num_epochs']
LEARNING_RATE = hyperparameters['learning_rate']
WEIGHT_DECAY = hyperparameters['weight_decay']
NUM_WORKERS = hyperparameters['num_workers']
RANDOM_SEED = hyperparameters['random_seed']
PATIENCE = hyperparameters['patience']
POS_WEIGHT = hyperparameters['pos_weight']

# Model parameters
SPATIAL_DIMS = hyperparameters['spatial_dims']
IN_CHANNELS = hyperparameters['in_channels']
OUT_CHANNELS = hyperparameters['out_channels']
NETWORK_CHANNELS = hyperparameters['network_channels']
KERNEL_SIZE = hyperparameters['kernel_size']
NUM_RES_UNITS = hyperparameters['num_res_units']
STRIDES = hyperparameters['strides']
DROPOUT = hyperparameters['dropout']
FREEZE_ENCODER = hyperparameters['freeze_encoder']

# Device configuration
DEVICE = hyperparameters['device']

class TanhWrapper(nn.Module):
    def __init__(self, model_class, *args, **kwargs):
        super().__init__()
        self.model = model_class(*args, **kwargs)
    
    def forward(self, x):
        x = self.model(x)
        x = torch.tanh(x)
        return x

from math import ceil

def bilinear_kernel_3d(kernel_size):
    # Create a 1D kernel
    def upsample_filt(size):
        factor = (size + 1) // 2
        center = (2 * factor - 1 - factor % 2) / (2.0 * factor)
        og = torch.arange(size).float()
        return (1 - torch.abs(og / factor - center)).clamp_(0)
    
    # Create the 1D kernels
    filt_1d = upsample_filt(kernel_size)
    # Outer product for 3D (e.g., using torch.einsum or manually expanding)
    filt_3d = filt_1d[:, None, None] * filt_1d[None, :, None] * filt_1d[None, None, :]
    return filt_3d / filt_3d.sum()  # Normalize

def init_transposed_conv_as_bilinear(conv):
    # conv is a nn.ConvTranspose3d layer
    kernel_size = conv.kernel_size[0]  # assuming cubic kernel
    bilinear_kernel = bilinear_kernel_3d(kernel_size).unsqueeze(0).unsqueeze(0)
    # Repeat this for all in/out channels if needed
    out_channels = conv.out_channels
    in_channels = conv.in_channels
    weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
    for i in range(out_channels):
        for j in range(in_channels):
            weight[i, j] = bilinear_kernel
    with torch.no_grad():
        conv.weight.copy_(weight)


def initialize_model(model_type="unet"):
    model_configs = {
        "unet": {
            "class": UNet,
            "params": {
                "spatial_dims": SPATIAL_DIMS,
                "in_channels": IN_CHANNELS,
                "out_channels": OUT_CHANNELS,
                "channels": NETWORK_CHANNELS,
                "kernel_size": KERNEL_SIZE,
                "num_res_units": NUM_RES_UNITS,
                "strides": STRIDES,
                "dropout": DROPOUT,
                "norm": ("INSTANCE", {"affine": True}),
                # "norm": "BATCH"
            }
        },
       "vnet": {
            "class": VNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "dropout_prob_down": 0.3,
                "dropout_prob_up": (0.3, 0.3),
                "bias": False,
                "act": ("elu", {"inplace": True})  # VNet originally uses ELU
            }
        },
        "segresnet": {
            "class": SegResNet,
            "params": {
                "spatial_dims": 3,
                "init_filters": 16,
                "in_channels": 1,
                "out_channels": 1,
                "dropout_prob": 0.3,
                "use_conv_final": True,
                "blocks_down": (1, 2, 2, 4),
                "blocks_up": (1, 1, 1),
                "act": ("RELU", {"inplace": True}),  # Changed to RELU
                "norm": ("GROUP", {"num_groups": 8})  # Group norm often works well
            }
        },
        "attentionunet": {
            "class": AttentionUnet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "channels": (16, 32, 64, 128),  # Reduced channels
                "strides": (2, 2, 2),           # Reduced depth
                "dropout": 0.3,
                "kernel_size": 3,
                "up_kernel_size": 3,  # Explicit kernel sizes
                "attention_levels": [False, True, True, False],
                "attention_dim": 4  # Reduce attention bottleneck dimension
            }
        },
        "dynunet": {
            "class": DynUNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                # Adjusted for 96x96x96 input
                "kernel_size": ((3,3,3), (3,3,3), (3,3,3)),
                "strides": ((1,1,1), (2,2,2), (2,2,2)),
                "upsample_kernel_size": ((2,2,2), (2,2,2)),
                "norm_name": ('INSTANCE', {'affine': True}),  # Instance norm
                "deep_supervision": True,
                "deep_supr_num": 1,  # Reduced from 2
                "res_block": True,    # Added residual blocks
                "dropout": 0.3        # Added dropout
            }
        },
        "basicunet": {
            "class": BasicUNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "features": (16, 32, 64, 128, 256, 16),  # Adjusted for memory
                "act": ("LeakyReLU", {"inplace": True, "negative_slope": 0.1}),
                "norm": ("instance", {"affine": True}),
                "dropout": 0.3,
                "upsample": "deconv"  # Better than interpolate for precise features
            }
        },
        "segresnetds": {  # SegResNet with deep supervision
            "class": SegResNetDS,
            "params": {
                "spatial_dims": 3,
                "init_filters": 32,
                "in_channels": 1,
                "out_channels": 1,
                "act": "relu",
                "norm": "batch",
                "blocks_down": (1, 2, 2, 4),
                "dsdepth": 2,  # Number of deep supervision outputs
                "resolution": None  # For isotropic kernels
            }
        },
        "segresnetvae": {  # SegResNet with VAE regularization
            "class": SegResNetVAE,
            "params": {
                "input_image_size": (96, 96, 96),
                "vae_estimate_std": False,
                "vae_default_std": 0.3,
                "vae_nz": 64,
                "spatial_dims": 3,
                "init_filters": 8,
                "in_channels": 1,
                "out_channels": 1,
                "dropout_prob": 0.3,
                "norm": ("GROUP", {"num_groups": 8})
            }
        },
        "highresnet": {
            "class": HighResNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "dropout_prob": 0.3,
                "norm_type": ("batch", {"affine": True}),
                "channel_matching": "pad"
            }
        },
        "basicunetplusplus": {  # UNet++ architecture
            "class": BasicUNetPlusPlus,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "features": (32, 32, 64, 128, 256, 32),
                "act": ("LeakyReLU", {"inplace": True, "negative_slope": 0.1}),
                "norm": ("instance", {"affine": True}),
                "dropout": 0.3,
                "upsample": "deconv",
                "deep_supervision": True
            }
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Model type {model_type} not supported. Choose from: {list(model_configs.keys())}")
    
    config = model_configs[model_type]
    model = config["class"](**config["params"])
    #################################################################### model = TanhWrapper(config["class"], **config["params"])

    # final_layer = model.model[-1]
    # new_final = nn.Sequential(
    #     final_layer,  # Keep the ConvTranspose3d
    #     nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)  # Add 1x1x1 conv
    # )
    # model.model[-1] = new_final
    
    # # Initialize transposed conv layers
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.ConvTranspose3d):
    #         init_transposed_conv_as_bilinear(module)
    
    print(f"Initialized modified {model_type.upper()}")
    return model





def dice_loss(preds, targets, smooth=1.0):
    """
    Computes the Dice loss between predictions and targets.
    """
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def balanced_l1_loss(outputs, labels, pos_weight=1.0, alpha=0.8):
    """
    Simple weighted L1 loss.
    
    Args:
        outputs: model predictions in [-1, 1]
        labels: ground truth SDTs in [-1, 1]
        pos_weight: weight for microtubule pixels (where label < 0)
    """
    l1_loss = nn.L1Loss(reduction='none')(outputs, labels).mean()
    
    # # Multiply the loss by pos_weight where labels < 0
    # weights = torch.ones_like(labels)
    # weights[labels < 0] = pos_weight
    # loss = (l1_loss * weights).mean()

    # outputs_bin = (outputs < 0.1).float()
    # labels_bin = (labels < 0.1).float()

    # # Compute Dice loss on binarized masks
    # dice = dice_loss(outputs_bin, labels_bin)

    # loss = alpha * loss + (1 - alpha) * dice
    
    return l1_loss

def bce_dice_loss(outputs, labels, pos_weight):
    # outputs: raw logits (N,1,D,H,W)
    # Apply sigmoid inside the loss
    prob = torch.sigmoid(outputs)
    
    # BCE loss
    # bce = F.binary_cross_entropy_with_logits(outputs, labels)

    # Handle empty volumes specially
    if not torch.any(labels):
        return prob.mean()  # Penalize any predictions in empty volumes
    
    # Dice loss (simple version)
    eps = 1e-6
    intersection = (prob * labels).sum()
    union = prob.sum() + labels.sum()
    dice = 1 - (2. * intersection + eps) / (union + eps)

    return dice
    
    # return 0 * bce + 1 * dice


def format_metrics(metrics, prefix=''):
    """Format metrics for pretty printing"""
    base_metrics = ['loss', 'dice', 'precision', 'recall', 'accuracy', 'balanced_accuracy']
    strict_metrics = ['dice_strict', 'precision_strict', 'recall_strict']
    
    metric_strings = []
    # Add base metrics
    for key in base_metrics:
        if key in metrics:
            metric_strings.append(f"{key.capitalize()}: {metrics[key]:.4f}")
    
    # Add strict metrics
    for key in strict_metrics:
        if key in metrics:
            metric_strings.append(f"{key.capitalize()}: {metrics[key]:.4f}")
            
    return f"{prefix} - " + ", ".join(metric_strings)


def compute_segmentation_metrics(outputs, labels, threshold=0.5):
    """
    Compute comprehensive segmentation metrics from SDT predictions.
    
    Args:
        outputs: model predictions in SDT space (N, 1, D, H, W)
        labels: ground truth in SDT space (N, 1, D, H, W)
        threshold: threshold for converting SDT to binary (default 0.0)
    """

    prob = torch.sigmoid(outputs)
    pred_masks = (prob > threshold).float()
    true_masks = labels.float()  # already 0/1
    
    ##################################################################### # Convert SDTs to binary masks
    ##################################################################### pred_masks = (outputs <= threshold).float()
    ##################################################################### true_masks = (labels <= threshold).float()
    
    # Compute basic counts for metrics
    true_positives = (pred_masks * true_masks).sum()
    true_negatives = ((1 - pred_masks) * (1 - true_masks)).sum()
    false_positives = (pred_masks * (1 - true_masks)).sum()
    false_negatives = ((1 - pred_masks) * true_masks).sum()
    
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    
    # Compute precision and recall
    precision = (true_positives + eps) / (true_positives + false_positives + eps)
    recall = (true_positives + eps) / (true_positives + false_negatives + eps)
    
    # Compute accuracy
    total_pixels = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total_pixels
    
    # Compute balanced accuracy
    sensitivity = recall  # Same as recall
    specificity = (true_negatives + eps) / (true_negatives + false_positives + eps)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Compute dice (equivalent to F1 for binary case)
    dice = (2 * true_positives + eps) / (2 * true_positives + false_positives + false_negatives + eps)
    
    return {
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'balanced_accuracy': balanced_accuracy.item(),
        'true_positives': true_positives.item(),
        'true_negatives': true_negatives.item(),
        'false_positives': false_positives.item(),
        'false_negatives': false_negatives.item()
    }


def get_uniform_transforms() -> Compose:
    """
    Generates all possible combinations of rotations and flips for a 3D object.
    Returns a MONAI OneOf transform that randomly selects from all possible orientations.
    
    Total configurations = 48 (24 rotations × 2 for no flip/flip)
    """
    # Generate all possible flip combinations (excluding redundant ones due to rotational symmetry)
    # We only need one flip axis since other flips can be achieved through rotations
    flips = [
        [],  # No flip
        [(0,)],  # Flip along x-axis
    ]
    
    # Generate all 24 rotations using composition of elementary rotations
    # We use the following convention:
    # - First rotate around z (xy-plane)
    # - Then rotate around y (xz-plane)
    # - Finally rotate around x (yz-plane)
    rotations = []
    
    # For each face that can be on top (6 possibilities)
    for face_up in range(6):
        # For each possible rotation of that face (4 possibilities)
        for rotation in range(4):
            if face_up == 0:  # Original orientation
                rots = [(0, 1)] * rotation
            elif face_up == 1:  # Rotate 180° around y
                rots = [(0, 2), (0, 2)] + [(0, 1)] * rotation
            elif face_up == 2:  # Rotate 90° around y
                rots = [(0, 2)] + [(0, 1)] * rotation
            elif face_up == 3:  # Rotate -90° around y
                rots = [(0, 2), (0, 2), (0, 2)] + [(0, 1)] * rotation
            elif face_up == 4:  # Rotate 90° around x
                rots = [(1, 2)] + [(1, 0)] * rotation
            else:  # face_up == 5, Rotate -90° around x
                rots = [(1, 2), (1, 2), (1, 2)] + [(1, 0)] * rotation
            
            rotations.append(rots)

    # Generate all combinations of flips and rotations
    all_transforms = []
    keys = ['image', 'label']
    
    for flip_axes, rotation_sequence in product(flips, rotations):
        transform_list = []
        
        # Add flips
        for axis in flip_axes:
            transform_list.append(
                RandFlipd(keys=keys, prob=1.0, spatial_axis=axis)
            )
        
        # Add rotations
        for rot_axes in rotation_sequence:
            transform_list.append(
                RandRotate90d(keys=keys, prob=1.0, max_k=1, spatial_axes=rot_axes)
            )
        
        all_transforms.append(Compose(transform_list))
    
    # Create a OneOf transform that randomly selects one transform
    # Each transform has equal probability
    return OneOf(all_transforms, weights=[1/len(all_transforms)] * len(all_transforms))

def verify_transform_count(transforms: List[Compose]) -> Dict:
    """
    Verifies the number of transforms generated.
    
    Returns:
        Dict containing counts and verification results
    """
    count = len(transforms)
    expected = 48  # 24 rotations × 2 for flips
    
    return {
        "total_transforms": count,
        "expected_transforms": expected,
        "is_correct": count == expected
    }

def save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, checkpoint_dir, filename):
    """Helper function to save model checkpoints"""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'hyperparameters': hyperparameters
    }
    torch.save(checkpoint, f"{checkpoint_dir}/{filename}")
    wandb.save(f"{checkpoint_dir}/{filename}")

def log_final_metrics(training_duration, epoch_times, completed_epochs, best_val_loss, best_val_dice, checkpoint_dir):
    """Helper function to log final training metrics"""
    final_metrics = {
        'training_duration_seconds': training_duration.total_seconds(),
        'training_duration_formatted': str(training_duration),
        'average_epoch_time_seconds': np.mean(epoch_times),
        'fastest_epoch_seconds': np.min(epoch_times),
        'slowest_epoch_seconds': np.max(epoch_times),
        'epoch_time_std_seconds': np.std(epoch_times),
        'completed_epochs': completed_epochs,
        'best_val_loss': best_val_loss,
        'best_val_dice': best_val_dice
    }





