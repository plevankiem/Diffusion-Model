from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class DDPMSchedulerConfig:
    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_timesteps: int = 1000
    method: str = "basic"

class DDPMScheduler:
    def __init__(self, config: DDPMSchedulerConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.betas = torch.linspace(config.beta_start, config.beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.bar_alphas = torch.cumprod(self.alphas, dim=0)
        self.sqrt_bar_alphas = torch.sqrt(self.bar_alphas)
        self.bar_alphas_prev = torch.cat([torch.ones(1), self.bar_alphas[:-1]])
        self.sqrt_one_minus_bar_alphas = torch.sqrt(1 - self.bar_alphas)
        if config.method == "basic":
            self.sigmas = torch.sqrt(self.betas)
        elif config.method == "vlb":
            self.sigmas = torch.sqrt(self.betas * (1 - self.bar_alphas_prev) / (1 - self.bar_alphas))
        else:
            raise ValueError(f"Invalid method: {config.method}")

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the noisy sample at time t.
        Args:
            x_0: [B, C, H, W]
            t: [B]
        
        Output:
            x_t: [B, C, H, W]
        """
        device = x_0.device
        # Move scheduler parameters to the same device as x_0
        sqrt_bar_alphas = self.sqrt_bar_alphas.to(device)
        sqrt_one_minus_bar_alphas = self.sqrt_one_minus_bar_alphas.to(device)
        
        noise = torch.randn_like(x_0)
        
        # Index and reshape for broadcasting: [B] -> [B, 1, 1, 1]
        sqrt_bar_alphas_t = sqrt_bar_alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_bar_alphas_t = sqrt_one_minus_bar_alphas[t].view(-1, 1, 1, 1)
        
        return sqrt_bar_alphas_t * x_0 + sqrt_one_minus_bar_alphas_t * noise, noise
    
    def backward(self, x_t: torch.Tensor, t: torch.Tensor, model: nn.Module, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the denoised sample at time t.
        Args:
            x_t: [B, C, H, W]
            t: [B] - tensor of timesteps
            model: nn.Module that predicts the noise
            y: [B] - optional conditioning labels
        
        Output:
            x_{t-1}: [B, C, H, W]
        """
        device = x_t.device
        # Move scheduler parameters to the same device as x_t
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        sqrt_one_minus_bar_alphas = self.sqrt_one_minus_bar_alphas.to(device)
        sigmas = self.sigmas.to(device)
        
        noise_pred = model(x_t, t, y)
        
        # Index and reshape for broadcasting: [B] -> [B, 1, 1, 1]
        betas_t = betas[t].view(-1, 1, 1, 1)
        alphas_t = alphas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_bar_alphas_t = sqrt_one_minus_bar_alphas[t].view(-1, 1, 1, 1)
        sigmas_t = sigmas[t].view(-1, 1, 1, 1)
        
        return (x_t - (betas_t / sqrt_one_minus_bar_alphas_t) * noise_pred) / torch.sqrt(alphas_t) + sigmas_t * torch.randn_like(x_t)
    
    def sample(self, model: nn.Module, shape: Tuple[int, ...], y: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from the diffusion model.
        Args:
            model: nn.Module that predicts the noise
            shape: Tuple of (batch_size, channels, height, width)
            y: [B] - optional conditioning labels
            device: Device to use for sampling
        
        Output:
            x_0: [B, C, H, W]
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Denoise step by step
        batch_size = shape[0]
        for t_val in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            x_t = self.backward(x_t, t, model, y)
        
        return x_t