import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBilinear2D(nn.Module):
    """ Weighted Bilinear 2D convolution with separable convolution.
    
    Methods:
        WeightedBilinear2D(pattern_size): list[int] -> WeightedBilinear2D object

        forward(x, mask): [B, 1, H, W] x [1, C, H, W] -> [B, C, H, W]
    """
    def __init__(self, pattern_size):
        super().__init__()
        self.pattern_size = pattern_size
        
        increasing = torch.linspace(1, pattern_size[0], pattern_size[0]) / pattern_size[0]
        kernel_h = torch.concat([increasing, increasing[:-1].flip(0)]) 

        increasing = torch.linspace(1, pattern_size[1], pattern_size[1]) / pattern_size[1]
        kernel_w = torch.concat([increasing, increasing[:-1].flip(0)]) 

        self.kernel_h = nn.Parameter(kernel_h.reshape(1, 1, -1))
        self.kernel_w = nn.Parameter(kernel_w.reshape(1, 1, -1))
        

    def forward(self, x, mask):
        """
        x : [B, 1, H, W]
        mask : [1, C, H, W]

        return : [B, C, H, W]
        """
        x = x * mask
        B, C, H, W = x.shape
        

        x = x.reshape(B*C*H, 1, W)
        x = F.conv1d(x, self.kernel_w, stride=1, padding="same")
        x = x.reshape(B*C, H, W).mT.reshape(B*C*W, 1, H)
        x = F.conv1d(x, self.kernel_h, stride=1, padding="same")
        x = x.reshape(B, C, W, H).mT
        return x


class ItSD(nn.Module):
    def __init__(self, pattern_size):
        super().__init__()
        self.wb = WeightedBilinear2D(pattern_size)
    
    def _forward_parallel(self, x, mask, C_a_bar):
        result = torch.zeros(x.shape[0], *mask.shape[1:], device=x.device, dtype=x.dtype)
        B, C, H, W = result.shape

        Kab_hat = (C_a_bar - x).reshape(B*C, 1, H, W)

        # %% Considering space complexity ( O(BHWC^2) once time ), no further parallelization here. 
        
        # Kab_bar = self.wb(Kab_hat, mask).reshape(B, C, C, H, W) 
        # result = torch.sum((x.unsqueeze(1) - Kab_bar) * mask.unsqueeze(2), dim=1)
        
        for b in range(C): 
            Kab_bar = self.wb(Kab_hat, mask[:, [b]]).reshape(B, C, H, W)
            result[:, b] = torch.sum((x - Kab_bar) * mask, dim=1)
        return result

    # def _forward_serial(self, x, mask, C_a_bar):
    #     """ Just for better understanding the parallel version. """
    #     result = torch.zeros(x.shape[0], *mask.shape[1:], device=x.device, dtype=x.dtype)
    #     B, C, H, W = result.shape

    #     for b in range(C):
    #         for a in range(C):
    #             if a==b:
    #                 continue
    #             Kab_bar = self.wb(C_a_bar[:, [a]] - x, mask[:, [b]])
    #             result[:, b][mask[:, a]==1.] = (x - Kab_bar)[:, 0][mask[:, a]==1.]
    #     return result

    def forward(self, x, mask, max_iter=5):
        """
        x : [B, 1, H, W]
        mask : [1, C, H, W]
        """
        # SD for initial guess.
        C_a_bar = self.wb(x, mask)
        result = self._forward_parallel(x, mask, C_a_bar)

        # %% If max_iter is not 0, run ItSD.
        for _ in range(max_iter):
            result = self._forward_parallel(x, mask, result)
        return result


class PPID(nn.Module):
    def __init__(self, pattern_size):
        super().__init__()
        m, n = pattern_size
        
        if m % 2 == 0:
            kernel_h = 2*torch.ones((m+1, 1))
            kernel_h[0] = 1.
            kernel_h[-1] = 1.
        else:
            kernel_h = torch.ones((m, 1))

        if n % 2 == 0:
            kernel_w = 2*torch.ones((1, n+1))
            kernel_w[:, 0] = 1.
            kernel_w[:, -1] = 1.
        else:
            kernel_w = torch.ones((1, n))
        norm_factor = (kernel_h * kernel_w).sum().sqrt()
        self.kernel_h = nn.Parameter(kernel_h.reshape(1, 1, -1) / norm_factor)
        self.kernel_w = nn.Parameter(kernel_w.reshape(1, 1, -1) / norm_factor)

        self.wb = WeightedBilinear2D(pattern_size)

    def forward(self, x, mask):
        """
        x : [B, 1, H, W]
        mask : [1, C, H, W]
        """
        x_hs = x * mask
        B, C, H, W = x_hs.shape

        # Raw Value Scale Adjustment
        max_value_per_band = x_hs.reshape((B, C, -1)).max(dim=-1, keepdim=True).values.unsqueeze(-1)
        adjust_scale = max_value_per_band.max(dim=1, keepdim=True).values / max_value_per_band
        I_apo_MSFA = (x_hs * adjust_scale).sum(dim=1, keepdim=True)

        # PPI Estimation
        I_M = I_apo_MSFA.reshape(B*H, 1, W)
        I_M = F.conv1d(I_M, self.kernel_w, stride=1, padding="same")
        I_M = I_M.reshape(B, H, W).mT.reshape(B*W, 1, H)
        I_M = F.conv1d(I_M, self.kernel_h, stride=1, padding="same")
        I_M = I_M.reshape(B, W, H).mT

        # computes the sparse difference
        Delta_M = (I_apo_MSFA - I_M) * mask
        I_apo = I_M + self.wb(Delta_M, mask)
        
        I_hat = I_apo / adjust_scale

        return I_hat
