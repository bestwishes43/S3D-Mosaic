import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from scipy.interpolate import interp1d
import pandas as pd

MAX_VAL_8_BIT = 2**8 - 1
MAX_VAL_12_BIT = 2**12 - 1
MAX_VAL_16_BIT = 2**16 - 1



def repeat_to(x: torch.Tensor, target_shape: tuple[int]|list[int]) -> torch.Tensor:
    """
    Repeat a tensor x to match the target shape.
    If a dimension can't be evenly divided, repeat to the smallest size ≥ target dimension first,
    then slice to the exact target shape.
    
    Args:
        x: Input tensor to be repeated
        target_shape: Target shape to match (tuple of integers)
    
    Returns:
        Tensor with shape equal to target_shape, created by repeating and slicing x
        
    Raises:
        ValueError: If target_shape has fewer dimensions than x (cannot reduce dimensions)
    """
    current_shape = x.shape
    num_current_dims = len(current_shape)
    num_target_dims = len(target_shape)
    
    if num_target_dims < num_current_dims:
        raise ValueError(
            f"Target shape ({target_shape}) has fewer dimensions than input tensor ({current_shape})"
        )
    
    if num_target_dims > num_current_dims:
        pad_dims = num_target_dims - num_current_dims
        x = x.view((1,) * pad_dims + current_shape)
        current_shape = x.shape
    
    repeat_times = [(target_dim + curr_dim - 1) // curr_dim 
                    for curr_dim, target_dim in zip(current_shape, target_shape)]
    
    x_repeated = x.repeat(*repeat_times)
    
    slices = [slice(0, target_dim) for target_dim in target_shape]
    x_final = x_repeated[tuple(slices)]
    
    return x_final

def im2single(img:torch.Tensor, bit_depth: int = 0):
    if img.dtype in [torch.uint16, torch.uint8]:
        max_val = torch.iinfo(img.dtype).max
        if bit_depth > 0:
            max_val = 2**bit_depth - 1
    else:
        raise ValueError('Input image dtype must be uint16 or uint8')
    img = img.float() / max_val
    return img

def srgb2linear(img:torch.Tensor):
    return img ** (1/2.2)

def im2uint8(x:np.ndarray):
    # assert x.max() <= 255 and x.min() >= 0, "input should be limited in [0, 255] for int and [0, 1] for float."
    if np.issubdtype(x.dtype, np.floating):
        x = x.clip(0.0, 1.0)*255
    return x.astype(np.uint8)

def calculate_psnr(
    ground_truth: torch.Tensor, 
    predicted: torch.Tensor, 
    value_range: float = 1., 
    per_band: bool = False
) -> torch.Tensor:
    """Calculates Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        ground_truth: Ground truth image tensor [H, W, C]
        predicted: Reconstructed image tensor [H, W, C]
        value_range: Maximum possible pixel value
        per_band: Return PSNR per band if True
        
    Returns:
        PSNR value(s) in dB
    """
    
    mse_per_band = torch.mean((predicted - ground_truth)**2, dim=(0, 1))
    psnr_per_band = 10 * torch.log10(value_range**2 / mse_per_band.clip(torch.finfo(torch.float32).eps))
    
    return psnr_per_band if per_band else torch.mean(psnr_per_band)

def calculate_sam(
    ground_truth: torch.Tensor, 
    predicted: torch.Tensor, 
) -> torch.Tensor:
    """Calculates Spectral Angle Mapper (SAM) in degrees.
    
    Args:
        ground_truth: Ground truth image tensor [H, W, C]
        predicted: Reconstructed image tensor [H, W, C]
        
    Returns:
        Mean SAM value in degrees
    """
    eps = 1e-12
    mask = ground_truth.sum(dim=-1) != 0
    dot_product = torch.sum(ground_truth * predicted, dim=-1)
    norm_product = torch.sqrt(
        torch.sum(ground_truth**2, dim=-1) * 
        torch.sum(predicted**2, dim=-1)
    ).clip(eps)
    sam_per_pixel = torch.acos((dot_product / norm_product).clip(-1.0, 1.0))
    return torch.mean(sam_per_pixel[mask] * 180 / torch.pi)

def calculate_ssim(
    ground_truth: torch.Tensor, 
    predicted: torch.Tensor, 
    value_range: float = 1., 
) -> float:
    """Calculates Spectral Angle Mapper (SAM) in degrees.
    
    Args:
        ground_truth: Ground truth image tensor [H, W, C]
        predicted: Reconstructed image tensor [H, W, C]
        
    Returns:
        Mean SAM value in degrees
    """
    ssim_value = ssim(ground_truth.cpu().numpy(), 
                      predicted.cpu().numpy(), 
                      data_range=value_range, 
                      multichannel=True)
    return ssim_value # type: ignore