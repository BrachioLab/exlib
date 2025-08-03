import torch
import numpy as np
from typing import Union

def torch_img_to_np(x: torch.Tensor) -> np.ndarray:
    """Convert torch image tensor to numpy array.
    
    Args:
        x: Tensor of shape (C, H, W) or (N, C, H, W)
        
    Returns:
        Array of shape (H, W, C) or (N, H, W, C)
    """
    if x.dim() == 4:
        return x.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif x.dim() == 3:
        return x.detach().cpu().permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

def np_to_torch_img(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch image tensor.
    
    Args:
        x: Array of shape (H, W, C) or (N, H, W, C)
        
    Returns:
        Tensor of shape (C, H, W) or (N, C, H, W)
    """
    x = torch.from_numpy(x)
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        return x.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {x.dim()}D")