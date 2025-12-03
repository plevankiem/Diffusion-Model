#!/usr/bin/env python3
"""
Sampling script for generating images from a trained diffusion model.
"""

import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image

from src.diffusion_model.models.unet import UNetModel, ModelConfig
from src.diffusion_model.schedulers.ddpm_scheduler import DDPMScheduler, DDPMSchedulerConfig
from src.diffusion_model.utils.checkpoint import load_checkpoint


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image from [-1, 1] to [0, 1] range.
    
    Args:
        image: Image tensor in range [-1, 1]
    
    Returns:
        Image tensor in range [0, 1]
    """
    return (image + 1.0) / 2.0


def save_samples(
    samples: torch.Tensor,
    output_dir: Path,
    prefix: str = "sample",
    grid: bool = True,
    nrow: int = 8
):
    """
    Save generated samples as images.
    
    Args:
        samples: Tensor of shape [N, C, H, W] in range [-1, 1]
        output_dir: Directory to save images
        prefix: Prefix for filenames
        grid: If True, also save as a grid
        nrow: Number of images per row in grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize to [0, 1]
    samples = denormalize_image(samples)
    
    # Clamp to valid range
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # Save individual images
    for i, sample in enumerate(samples):
        filename = output_dir / f"{prefix}_{i:04d}.png"
        save_image(sample, filename)
    
    print(f"Saved {len(samples)} individual images to {output_dir}")
    
    # Save grid if requested
    if grid:
        grid_image = make_grid(samples, nrow=nrow, padding=2, pad_value=1.0)
        grid_filename = output_dir / f"{prefix}_grid.png"
        save_image(grid_image, grid_filename)
        print(f"Saved grid image to {grid_filename}")


def generate_samples(
    model: UNetModel,
    scheduler: DDPMScheduler,
    num_samples: int,
    shape: tuple,
    device: torch.device,
    conditional: bool = False,
    labels: torch.Tensor = None,
    verbose: bool = True
) -> torch.Tensor:
    """
    Generate samples from the diffusion model.
    
    Args:
        model: Trained UNet model
        scheduler: DDPM scheduler
        num_samples: Number of samples to generate
        shape: Shape tuple (C, H, W)
        device: Device to generate on
        conditional: Whether to use conditional generation
        labels: Optional labels for conditional generation
        verbose: Whether to print progress
    
    Returns:
        Generated samples tensor
    """
    model.eval()
    
    # Create full shape
    full_shape = (num_samples,) + shape
    
    # Generate labels if conditional but not provided
    y = None
    if conditional:
        if labels is None:
            # Sample random labels
            num_classes = model.config.num_classes if hasattr(model.config, 'num_classes') and model.config.num_classes else 10
            y = torch.randint(0, num_classes, (num_samples,), device=device)
        else:
            y = labels.to(device)
            if len(y) != num_samples:
                raise ValueError(f"Number of labels ({len(y)}) must match num_samples ({num_samples})")
    
    if verbose:
        print(f"Generating {num_samples} samples with shape {shape}...")
    
    with torch.no_grad():
        samples = scheduler.sample(
            model=model,
            shape=full_shape,
            y=y,
            device=device
        )
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a trained diffusion model")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    
    # Generation arguments
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate (default: 16)")
    parser.add_argument("--image-size", type=int, default=None, help="Image size (if not in checkpoint)")
    parser.add_argument("--output-dir", type=str, default="samples", help="Output directory for samples (default: samples)")
    parser.add_argument("--prefix", type=str, default="sample", help="Prefix for output filenames (default: sample)")
    parser.add_argument("--no-grid", action="store_true", help="Don't save grid image")
    parser.add_argument("--grid-nrow", type=int, default=8, help="Number of images per row in grid (default: 8)")
    
    # Conditional generation
    parser.add_argument("--conditional", action="store_true", help="Use conditional generation")
    parser.add_argument("--labels", type=int, nargs="+", default=None, help="Specific class labels for conditional generation")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (if not in checkpoint)")
    
    # Scheduler arguments (for override)
    parser.add_argument("--num-timesteps", type=int, default=None, help="Override number of timesteps for sampling")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    
    # First, we need to create a dummy model to load the checkpoint
    # We'll get config from checkpoint if available
    # Use weights_only=False to allow loading custom dataclasses (safe for our own checkpoints)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Extract configs from checkpoint
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        print("Loaded model config from checkpoint")
    else:
        # Create default config (will be overridden by state dict)
        image_size = args.image_size if args.image_size else 32
        num_classes = args.num_classes if args.num_classes else (10 if args.conditional else None)
        model_config = ModelConfig(
            image_size=image_size,
            in_channels=3,
            base_channels=128,
            channel_mult=(1, 2, 2, 2),
            num_res_blocks=2,
            num_heads=4,
            attention_resolutions=(16, 8),
            num_classes=num_classes,
            dropout=0.0,
            use_scale_shift_norm=True,
        )
        print("Using default model config (may not match checkpoint)")
    
    if "scheduler_config" in checkpoint:
        scheduler_config = checkpoint["scheduler_config"]
        if args.num_timesteps:
            scheduler_config.num_timesteps = args.num_timesteps
        print(f"Loaded scheduler config from checkpoint (timesteps: {scheduler_config.num_timesteps})")
    else:
        # Create default scheduler config
        num_timesteps = args.num_timesteps if args.num_timesteps else 1000
        scheduler_config = DDPMSchedulerConfig(
            beta_start=1e-4,
            beta_end=0.02,
            num_timesteps=num_timesteps,
            method="basic",
        )
        print(f"Using default scheduler config (timesteps: {num_timesteps})")
    
    # Create model
    print("Creating model...")
    model = UNetModel(model_config)
    model = model.to(device)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model state loaded successfully")
    else:
        raise ValueError("Checkpoint does not contain model_state_dict")
    
    # Create scheduler
    from src.diffusion_model.schedulers.ddpm_scheduler import DDPMScheduler
    scheduler = DDPMScheduler(scheduler_config)
    
    # Determine image shape
    image_size = model_config.image_size
    in_channels = model_config.in_channels
    shape = (in_channels, image_size, image_size)
    
    print(f"Image shape: {shape}")
    
    # Determine if conditional
    conditional = args.conditional or (model_config.num_classes is not None)
    
    # Prepare labels if conditional
    labels = None
    if conditional:
        if args.labels:
            if len(args.labels) != args.num_samples:
                raise ValueError(f"Number of labels ({len(args.labels)}) must match --num-samples ({args.num_samples})")
            labels = torch.tensor(args.labels, device=device)
        elif args.conditional:
            # Will be generated randomly in generate_samples
            pass
    
    # Generate samples
    samples = generate_samples(
        model=model,
        scheduler=scheduler,
        num_samples=args.num_samples,
        shape=shape,
        device=device,
        conditional=conditional,
        labels=labels,
        verbose=True
    )
    
    # Save samples
    print(f"\nSaving samples to {args.output_dir}...")
    save_samples(
        samples=samples,
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        grid=not args.no_grid,
        nrow=args.grid_nrow
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

