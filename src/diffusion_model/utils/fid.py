"""
FID (Fréchet Inception Distance) computation utilities.

This module provides functions to compute FID scores between real and generated images.
FID measures the similarity between two sets of images by comparing their statistics
in the feature space of Inception v3.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from typing import Optional
import numpy as np
from scipy import linalg
from tqdm import tqdm


class InceptionV3FeatureExtractor(nn.Module):
    """
    Extracts features from images using Inception v3 model.
    Features are extracted from the last average pooling layer before classification.
    """
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        # Load Inception v3 pretrained model
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()  # Remove the classification head
        inception.aux_logits = False  # Disable auxiliary outputs
        inception.eval()
        
        # Move to device and set to evaluation mode
        inception = inception.to(device)
        
        self.model = inception
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input images of shape [B, C, H, W] in range [-1, 1]
        
        Returns:
            Features of shape [B, 2048]
        """
        # Convert from [-1, 1] to [0, 1] for Inception v3
        x = (x + 1.0) / 2.0
        
        # Resize to 299x299 if needed (Inception v3 input size)
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.model(x)
        return features


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """
    Calculate the Fréchet distance between two multivariate Gaussians.
    
    Args:
        mu1: Mean of the first distribution
        sigma1: Covariance matrix of the first distribution
        mu2: Mean of the second distribution
        sigma2: Covariance matrix of the second distribution
        eps: Small constant for numerical stability
    
    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance matrix of features.
    
    Args:
        features: Array of shape [N, feature_dim]
    
    Returns:
        Mean vector and covariance matrix
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def extract_features_from_dataloader(
    dataloader: DataLoader,
    feature_extractor: InceptionV3FeatureExtractor,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Extract features from all images in a DataLoader.
    
    Args:
        dataloader: DataLoader containing images
        feature_extractor: InceptionV3FeatureExtractor instance
        max_samples: Maximum number of samples to process (None for all)
        device: Device to use for computation
    
    Returns:
        Array of features of shape [N, 2048]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    features_list = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]  # Assume first element is images
            else:
                images = batch
            
            images = images.to(device)
            
            # Extract features
            batch_features = feature_extractor(images)
            features_list.append(batch_features.cpu().numpy())
            
            total_samples += images.shape[0]
            if max_samples is not None and total_samples >= max_samples:
                break
    
    features = np.concatenate(features_list, axis=0)
    
    # Limit to max_samples if specified
    if max_samples is not None:
        features = features[:max_samples]
    
    return features


def compute_fid(
    real_dataloader: DataLoader,
    generated_images: torch.Tensor,
    feature_extractor: Optional[InceptionV3FeatureExtractor] = None,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None
) -> float:
    """
    Compute FID score between real images (from dataloader) and generated images.
    
    Args:
        real_dataloader: DataLoader containing real images
        generated_images: Tensor of generated images of shape [N, C, H, W] in range [-1, 1]
        feature_extractor: Optional InceptionV3FeatureExtractor (will be created if None)
        device: Device to use for computation
        max_samples: Maximum number of samples to use from each set (None for all)
    
    Returns:
        FID score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if feature_extractor is None:
        feature_extractor = InceptionV3FeatureExtractor(device=device)
    else:
        feature_extractor = feature_extractor.to(device)
    
    # Extract features from real images
    real_features = extract_features_from_dataloader(
        real_dataloader, feature_extractor, max_samples=max_samples, device=device
    )
    
    # Extract features from generated images
    generated_images = generated_images.to(device)
    if max_samples is not None and generated_images.shape[0] > max_samples:
        generated_images = generated_images[:max_samples]
    
    with torch.no_grad():
        generated_features = feature_extractor(generated_images).cpu().numpy()
    
    # Compute statistics
    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(generated_features)
    
    # Compute FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return float(fid)

