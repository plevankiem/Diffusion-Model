from src.diffusion_model.schedulers.ddpm_scheduler import DDPMScheduler
from src.diffusion_model.models.unet import UNetModel
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
    
    def train(self):

        print(f"Training on {self.device}...")
        print("--------------------------------")
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
                    val_loss = 0.0
                    progress_bar = tqdm(self.dataloader["test"], desc=f"Validation Epoch {epoch+1}/{self.config.num_epochs}")
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
                        val_loss += loss.item()
                        progress_bar.set_postfix(loss=val_loss / (i + 1))
            # Set model back to training mode
            self.model.train()

            print(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {epoch_loss / len(self.dataloader['train'])} - Val Loss: {val_loss / len(self.dataloader['test'])}")
            print("--------------------------------")