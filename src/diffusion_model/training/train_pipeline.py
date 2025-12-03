from src.diffusion_model.schedulers.ddpm_scheduler import DDPMScheduler
from src.diffusion_model.models.unet import UNetModel
from src.diffusion_model.utils.fid import compute_fid, InceptionV3FeatureExtractor
from dataclasses import dataclass
import torch.nn as nn
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 128
    conditional: bool = False

class TrainingPipieline:
    def __init__(self, config: TrainingConfig, model: UNetModel, dataloader: Dict[str, DataLoader], scheduler: DDPMScheduler):
        self.config = config
        self.model = model
        self.scheduler = scheduler
        self.dataloader = dataloader

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
    
    def generate_samples(self, num_samples: int, shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            shape: Optional shape tuple (channels, height, width). 
                   If None, infers from dataloader.
        
        Returns:
            Generated images tensor of shape [num_samples, C, H, W]
        """
        self.model.eval()
        
        # Infer shape from dataloader if not provided
        if shape is None:
            # Try to get shape from test dataloader, fallback to train
            dataloader_key = "test" if "test" in self.dataloader else "train"
            sample_batch = next(iter(self.dataloader[dataloader_key]))
            if isinstance(sample_batch, (list, tuple)):
                sample_img = sample_batch[0]
            else:
                sample_img = sample_batch
            shape = (num_samples, sample_img.shape[1], sample_img.shape[2], sample_img.shape[3])
        else:
            shape = (num_samples,) + shape
        
        with torch.no_grad():
            # Handle conditional generation
            y = None
            if self.config.conditional:
                # Sample random labels if conditional
                # Assuming labels are integers from 0 to num_classes-1
                # You may need to adjust this based on your dataset
                num_classes = self.model.config.num_classes if hasattr(self.model.config, 'num_classes') and self.model.config.num_classes else 10
                y = torch.randint(0, num_classes, (num_samples,), device=self.device)
            
            generated = self.scheduler.sample(self.model, shape, y=y, device=self.device)
        
        return generated
    
    def compute_fid_score(self, num_samples: Optional[int] = None) -> float:
        """
        Compute FID score between generated samples and test dataset.
        
        Args:
            num_samples: Number of samples to use for FID computation.
                        If None, uses all samples in test dataloader.
        
        Returns:
            FID score
        """
        if "test" not in self.dataloader:
            raise ValueError("Test dataloader not found. Cannot compute FID score.")
        
        self.model.eval()
        
        # Determine number of samples
        if num_samples is None:
            # Use the size of the test dataset
            test_dataset = self.dataloader["test"].dataset
            num_samples = len(test_dataset)
        
        # Generate samples
        print(f"Generating {num_samples} samples for FID computation...")
        generated_samples = self.generate_samples(num_samples)
        
        # Compute FID
        print("Computing FID score...")
        feature_extractor = InceptionV3FeatureExtractor(device=self.device)
        fid_score = compute_fid(
            real_dataloader=self.dataloader["test"],
            generated_images=generated_samples,
            feature_extractor=feature_extractor,
            device=self.device,
            max_samples=num_samples
        )
        
        return fid_score
    
    def train(self):

        print(f"Training on {self.device}...")
        self.model.train()
        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(self.dataloader["train"], desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            epoch_loss = 0.0
            for i, (x, y) in enumerate(progress_bar):
                x = x.to(self.device)
                if self.config.conditional:
                    y = y.to(self.device)
                else:
                    y = None
                timesteps = torch.randint(0, self.scheduler.num_timesteps, (x.shape[0],), device=self.device)
                x_t, noise = self.scheduler.forward(x, timesteps)
                noise_pred = self.model(x_t, timesteps, y)
                loss = self.loss_fn(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / (i + 1))
            
            # Compute FID score at the end of each epoch (if test dataloader is available)
            if "test" in self.dataloader:
                self.model.eval()
                with torch.no_grad():
                    try:
                        num_test_samples = len(self.dataloader["test"].dataset)
                        fid_score = self.compute_fid_score(num_samples=min(5000, num_test_samples))
                        print(f"Epoch {epoch+1}/{self.config.num_epochs} - FID: {fid_score:.4f}")
                    except Exception as e:
                        print(f"Error computing FID: {e}")
                        # Continue training even if FID computation fails
            
            # Set model back to training mode
            self.model.train()

