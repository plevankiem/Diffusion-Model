#!/usr/bin/env python3
"""
Training script for the diffusion model.

Supports quick test mode (few samples, few epochs) and full training mode.
"""

import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.diffusion_model.models.unet import UNetModel, ModelConfig
from src.diffusion_model.schedulers.ddpm_scheduler import DDPMScheduler, DDPMSchedulerConfig
from src.diffusion_model.training.train_pipeline import TrainingPipieline, TrainingConfig
from src.diffusion_model.data.dataset import create_train_val_test_datasets
from src.diffusion_model.data.transforms import build_transform_for_split
from src.diffusion_model.utils.checkpoint import save_checkpoint


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_quick_test_datasets(datasets_dict: Dict, num_train_samples: int = 100, num_test_samples: int = 50):
    """
    Create quick test datasets by limiting the number of samples.
    
    Args:
        datasets_dict: Dictionary of datasets
        num_train_samples: Number of training samples for quick test
        num_test_samples: Number of test samples for quick test
    
    Returns:
        Dictionary with limited datasets
    """
    quick_datasets = {}
    
    if "train" in datasets_dict:
        train_dataset = datasets_dict["train"]
        indices = list(range(min(num_train_samples, len(train_dataset))))
        quick_datasets["train"] = Subset(train_dataset, indices)
        print(f"Quick test mode: Using {len(indices)} training samples")
    
    if "test" in datasets_dict:
        test_dataset = datasets_dict["test"]
        indices = list(range(min(num_test_samples, len(test_dataset))))
        quick_datasets["test"] = Subset(test_dataset, indices)
        print(f"Quick test mode: Using {len(indices)} test samples")
    
    return quick_datasets


def create_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    image_size: Optional[int] = None,
    quick_test: bool = False,
    num_train_samples: int = 100,
    num_test_samples: int = 50,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training and testing.
    
    Args:
        dataset_name: Name of the dataset (e.g., "cifar10")
        data_root: Root directory for data
        batch_size: Batch size for dataloaders
        image_size: Optional image size (None uses dataset default)
        quick_test: If True, use limited samples for quick testing
        num_train_samples: Number of training samples for quick test
        num_test_samples: Number of test samples for quick test
    
    Returns:
        Dictionary of dataloaders
    """
    # Build transforms
    transforms_by_split = {
        "train": build_transform_for_split(dataset_name, image_size=image_size),
        "test": build_transform_for_split(dataset_name, image_size=image_size),
    }
    
    # Create datasets
    datasets_dict = create_train_val_test_datasets(
        name=dataset_name,
        root=data_root,
        transforms_by_split=transforms_by_split,
        download=True,
        image_size=image_size,
    )
    
    # Limit datasets for quick test if requested
    if quick_test:
        datasets_dict = create_quick_test_datasets(datasets_dict, num_train_samples, num_test_samples)
    
    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets_dict.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    
    return dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (default: cifar10)")
    parser.add_argument("--data-root", type=str, default="src/diffusion_model/data", help="Data root directory")
    parser.add_argument("--image-size", type=int, default=None, help="Image size (None for dataset default)")
    
    # Model arguments
    parser.add_argument("--base-channels", type=int, default=128, help="Base channels for UNet (default: 128)")
    parser.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 2, 2], help="Channel multipliers (default: 1 2 2 2)")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Number of residual blocks per level (default: 2)")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--attention-resolutions", type=int, nargs="+", default=[16, 8], help="Resolutions for attention (default: 16 8)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0)")
    parser.add_argument("--conditional", action="store_true", help="Use conditional generation")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes for conditional generation")
    
    # Scheduler arguments
    parser.add_argument("--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps (default: 1000)")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Beta start value (default: 1e-4)")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Beta end value (default: 0.02)")
    parser.add_argument("--scheduler-method", type=str, default="basic", choices=["basic", "vlb"], help="Scheduler method (default: basic)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Quick test mode
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode (few samples, few epochs)")
    parser.add_argument("--num-train-samples", type=int, default=100, help="Number of training samples for quick test (default: 100)")
    parser.add_argument("--num-test-samples", type=int, default=50, help="Number of test samples for quick test (default: 50)")
    parser.add_argument("--quick-epochs", type=int, default=2, help="Number of epochs for quick test (default: 2)")
    parser.add_argument("--quick-timesteps", type=int, default=100, help="Number of timesteps for quick test (default: 100)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Adjust parameters for quick test mode
    if args.quick_test:
        print("=" * 60)
        print("QUICK TEST MODE")
        print("=" * 60)
        args.epochs = args.quick_epochs
        args.num_timesteps = args.quick_timesteps
        args.batch_size = min(args.batch_size, 32)  # Limit batch size for quick test
        print(f"Quick test: {args.epochs} epochs, {args.num_timesteps} timesteps")
    
    # Create output directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine image size (CIFAR-10 is 32x32)
    if args.image_size is None:
        if args.dataset.lower() == "cifar10":
            args.image_size = 32
        else:
            args.image_size = 32  # Default
    
    # Create dataloaders
    print(f"Creating dataloaders for dataset: {args.dataset}")
    dataloaders = create_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        quick_test=args.quick_test,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
    )
    
    # Get sample batch to determine image shape
    sample_batch = next(iter(dataloaders["train"]))
    if isinstance(sample_batch, (list, tuple)):
        sample_img = sample_batch[0]
    else:
        sample_img = sample_batch
    
    _, channels, height, width = sample_img.shape
    print(f"Image shape: {channels}x{height}x{width}")
    
    # Determine number of classes
    num_classes = args.num_classes
    if num_classes is None:
        if args.dataset.lower() == "cifar10":
            num_classes = 10
        elif args.dataset.lower() == "celeba":
            num_classes = None  # CelebA doesn't have class labels by default
        else:
            num_classes = None
    
    # Create model config
    model_config = ModelConfig(
        image_size=args.image_size,
        in_channels=channels,
        base_channels=args.base_channels,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        num_heads=args.num_heads,
        attention_resolutions=tuple(args.attention_resolutions),
        num_classes=num_classes if args.conditional else None,
        dropout=args.dropout,
        use_scale_shift_norm=True,
    )
    
    # Create model
    print("Creating UNet model...")
    model = UNetModel(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create scheduler
    scheduler_config = DDPMSchedulerConfig(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_timesteps=args.num_timesteps,
        method=args.scheduler_method,
    )
    scheduler = DDPMScheduler(scheduler_config)
    print(f"Scheduler created with {scheduler.num_timesteps} timesteps")
    
    # Create training config
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        conditional=args.conditional,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        from src.diffusion_model.utils.checkpoint import load_checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(args.resume, model, device=device)
        if "training_config" in checkpoint:
            training_config = checkpoint["training_config"]
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    
    # Create training pipeline
    print("Creating training pipeline...")
    pipeline = TrainingPipieline(
        config=training_config,
        model=model,
        dataloader=dataloaders,
        scheduler=scheduler,
    )
    
    # Start training (the train() method handles all epochs internally)
    print("Starting training...")
    print(f"Training for {training_config.num_epochs} epochs")
    pipeline.train()
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{training_config.num_epochs}.pth"
    save_checkpoint(
        model=model,
        optimizer=pipeline.optimizer,
        scheduler=scheduler,
        training_config=training_config,
        epoch=training_config.num_epochs,
        filepath=str(final_checkpoint_path),
        model_config=model_config,
        scheduler_config=scheduler_config,
    )
    
    print("Training completed!")
    print(f"Final checkpoint saved in: {final_checkpoint_path}")


if __name__ == "__main__":
    main()