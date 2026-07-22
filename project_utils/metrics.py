import torch
from typing import Literal

__all__ = ["calculate_psnr", "calculate_sam", "SSIM"]

def check_shape_equality(*images):
    """Check that all images have the same shape"""
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')
    return

def im2uint(x:torch.Tensor, data_range:int=255):
    """Converts floating-point tensor to uint representation.
    
    Args:
        x: Input tensor (float in [0,1])
        data_range: Maximum value for scaling (typically 255 for 8-bit)
    
    Returns:
        Converted uint tensor
    """
    if not torch.is_floating_point(x):
        return x
    
    # Scale, round, and clip floating-point inputs
    x = torch.round(x * data_range)
    x = torch.clamp(x, 0, data_range)
    return x.to(torch.uint32)

def generate_gaussian_kernel(
    kernel_size: int,
    sigma: float
) -> torch.Tensor:
    """Generate 2D Gaussian convolution kernel.
    
    Args:
        kernel_size: Size of kernel (must be odd)
        sigma: Standard deviation of Gaussian distribution
    
    Returns:
        Normalized kernel tensor [1, 1, kernel_size, kernel_size]
    
    Example:
        >>> kernel = generate_gaussian_kernel(5, 1.5)
        >>> blurred = F.conv2d(image, kernel, padding=2)
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Create 1D Gaussian vector
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Outer product for 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)

class GaussianConv2d(torch.nn.Module):
    def __init__(
        self, 
        kernel_size: int, 
        sigma: float, 
        padding: str | tuple[int, int] | int = 0,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "reflect"
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False, padding_mode=padding_mode)
        self.conv.weight.data = generate_gaussian_kernel(kernel_size, sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B*C, 1, H, W)
        feat = self.conv(x).squeeze(1)
        feat = torch.unflatten(feat, dim=0, sizes=(B, C))
        return feat

def calculate_psnr(
    im1: torch.Tensor, 
    im2: torch.Tensor, 
    bit_depth: int = 8, 
    per_channel: bool = False
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    Args:
        im1, im2 : Images with same shape [B, C, H, W].
        bit_depth: Bit depth of image (used to quantification)
        per_channel: Return per-channel PSNR if True
    
    Returns:
        PSNR values [B] or [B, C] depending on per_channel flag
    
    Formula:
        PSNR = 10 * log10(MAX² / MSE)
        where MAX = (2^bit_depth - 1) for int or 1. for float.
    """
    check_shape_equality(im1, im2)
    
    if bit_depth > 0:
        data_range = 2**bit_depth - 1
        im1 = im2uint(im1, data_range).float()
        im2 = im2uint(im2, data_range).float()
    else:
        data_range = 1.0
    
    # Compute MSE per channel
    mse_per_channel = torch.mean(
        (im1 - im2) ** 2, dim=(-2, -1)  
    )

    # Calculate PSNR with numerical stability
    epsilon = torch.finfo(torch.float32).eps
    psnr_per_channel = 10 * torch.log10(
        (data_range ** 2) / torch.clamp(mse_per_channel, min=epsilon)
    )
    
    if per_channel:
        return psnr_per_channel
    return psnr_per_channel.mean(dim=-1)

def calculate_sam(
    im1: torch.Tensor,
    im2: torch.Tensor,
    bit_depth: int = 8,
    epsilon = 1e-12
) -> torch.Tensor:
    """Compute Spectral Angle Mapper (SAM) in degrees.
    
    Args:
        im1, im2 : Images with same shape [B, C, H, W].
        bit_depth: Bit depth for normalization
        epsilon: Small value to avoid division by zero
    Returns:
        Mean SAM angle in degrees [B]
    
    Note:
        - Handles zero-magnitude pixels safely
    """
    check_shape_equality(im1, im2)

    if bit_depth > 0:
        data_range = 2**bit_depth - 1
        im1 = im2uint(im1, data_range).float()
        im2 = im2uint(im2, data_range).float()

    # Compute vector magnitudes
    mag_im1 = torch.norm(im1, p=2, dim=1)
    mag_im2 = torch.norm(im2, p=2, dim=1)
    mag_product = mag_im1 * mag_im2

    # Find valid mask: both vectors must have non-zero magnitude
    valid_mask = mag_product > epsilon
    
    # Handle edge case: all pixels invalid
    if not torch.any(valid_mask):
        print("\033[93mWarning: No valid pixels. SAM is undefined. Returning 90.0° (theoretical max).\033[0m")
        return torch.full((im1.shape[0],), 90.0, device=im1.device)
    
    # Compute dot product
    dot_product = torch.sum(im1 * im2, dim=1)

    # Compute cosine similarity with numerical stability
    cos_similarity = torch.clamp(
        dot_product / mag_product.clamp(min=epsilon),
        -1.0, 1.0
    )
    cos_similarity[~valid_mask] = torch.nan

    # Compute mean SAM in degrees
    sam_radians = torch.acos(cos_similarity).nanmean(dim=[-2, -1])
    return torch.rad2deg(sam_radians)

class SSIM(torch.nn.Module):
    def __init__(
        self,
        K1: float = 0.01,
        K2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5,
        per_channel: bool = False
    ):
        """Create mean Structural Similarity Index (M-SSIM) module.
    
        Args:
            K1, K2: Constants for SSIM formula.
            window_size: Size of sliding window
            sigma: Sigma for Gaussian weighting
            per_channel: Return per-channel SSIM if True
        
        Reference:
        ----------
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing,
        13, 600-612.
        """
        super().__init__()
        self.conv = GaussianConv2d(window_size, sigma).requires_grad_(False)
        self.K1 = K1
        self.K2 = K2
        self.per_channel = per_channel
    
    def forward(
        self,
        im1: torch.Tensor,
        im2: torch.Tensor,
        bit_depth: int = 8
    ) -> torch.Tensor:
        """Compute mean Structural Similarity Index (M-SSIM).
    
        Args:
            im1, im2 : Images with same shape [B, C, H, W].
            bit_depth: Image bit depth.
        Returns:
            SSIM values [B] or [B, C] depending on per_band
        """
        check_shape_equality(im1, im2)
    
        if bit_depth > 0:
            data_range = 2**bit_depth - 1
            im1 = im2uint(im1, data_range).float()
            im2 = im2uint(im2, data_range).float()
        else:
            data_range = 1.
        
        # Local statistics
        mu1 = self.conv(im1)
        mu2 = self.conv(im2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = self.conv(im1 ** 2) - mu1_sq
        sigma2_sq = self.conv(im2 ** 2) - mu2_sq
        sigma12 = self.conv(im1 * im2) - mu1_mu2
        
        # M-SSIM computation for each channel
        C1 = (self.K1 * data_range) ** 2
        C2 = (self.K2 * data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        mssim_per_channel = ssim_map.mean(dim=(-2, -1))
        
        if self.per_channel:
            return mssim_per_channel
        return mssim_per_channel.mean(dim=-1)


if __name__ == "__main__":
    # Validation Code
    C, H, W = 28, 256, 256
    im1 = torch.rand(1, C, H, W)
    im2 = torch.rand(1, C, H, W)

    
    # Our implementation
    calculate_ssim = SSIM()
    
    psnr_val_our = calculate_psnr(im1, im2, bit_depth=-1).numpy()
    ssim_val_our = calculate_ssim(im1, im2, bit_depth=-1).numpy()
    sam_val_our = calculate_sam(im1, im2, bit_depth=-1).numpy()

    # skimage implementation
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import numpy as np

    im1_np = im1[0].numpy()
    im2_np = im2[0].numpy()

    psnr_sum = np.zeros(1)
    ssim_sum = np.zeros(1)
    for c in range(C):
        psnr_c = psnr(
            im1_np[c], im2_np[c], data_range=1.
        )
        ssim_c = ssim(
            im1_np[c], im2_np[c], data_range=1., 
            gaussian_weights=True, use_sample_covariance=False
        )
        psnr_sum += psnr_c
        ssim_sum += ssim_c
    psnr_val_skimage = psnr_sum / C
    ssim_val_skimage = ssim_sum / C

    # Compare results
    print(f"PSNR: {np.abs(psnr_val_our-psnr_val_skimage)}")
    print(f"SSIM: {np.abs(ssim_val_our-ssim_val_skimage)}")
