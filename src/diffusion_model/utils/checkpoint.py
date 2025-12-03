"""
Checkpoint utilities for saving and loading model states.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from src.diffusion_model.models.unet import UNetModel
from src.diffusion_model.training.train_pipeline import TrainingConfig


def save_checkpoint(
    model: UNetModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    training_config: TrainingConfig,
    epoch: int,
    loss: Optional[float] = None,
    filepath: str = "checkpoint.pth",
    **kwargs
) -> None:
    """
    Save a checkpoint containing model, optimizer, and training state.
    
    Args:
        model: The UNet model to save
        optimizer: The optimizer state
        scheduler: The diffusion scheduler
        training_config: Training configuration
        epoch: Current epoch number
        loss: Optional loss value
        filepath: Path to save the checkpoint
        **kwargs: Additional items to save (e.g., fid_score, etc.)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_config": training_config,
        "loss": loss,
        **kwargs
    }
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: UNetModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model and optionally optimizer state.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
    
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use weights_only=False to allow loading custom dataclasses (safe for our own checkpoints)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint

