import torch
import torch.nn.functional as F
from typing import Tuple


def patch_segment_image(image: torch.Tensor, patch_size: int = 1) -> torch.Tensor:
    """Segment an image into square patches.

    Args:
        image: Tensor of shape (C, H, W) or (B, C, H, W)
        patch_size: Size of each square patch

    Returns:
        Tensor of shape (H, W) or (B, H, W) with patch (segment) indices
    """
    # Always work with 4D input for simplicity
    orig_dim = image.dim()
    if orig_dim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
    b, c, h, w = image.shape
    segments = torch.zeros((b, h, w), dtype=torch.long, device=image.device)
    for b_idx in range(b):
        idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                segments[b_idx, i:i+patch_size, j:j+patch_size] = idx
                idx += 1
    if orig_dim == 3:
        segments = segments[0]
    return segments


def convert_idx_masks_to_bool(masks: torch.Tensor) -> torch.Tensor:
    """Convert index masks to boolean masks.
    
    Args:
        masks: Tensor of shape (1, H, W) with segment indices
        
    Returns:
        Boolean tensor of shape (num_segments, H, W)
    """
    unique_idxs = torch.unique(masks).sort().values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool

def create_patch_mask(
    height: int, 
    width: int, 
    patch_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    """Create a grid mask for patches.
    
    Args:
        height: Image height
        width: Image width
        patch_size: Size of each patch (h, w)
        
    Returns:
        Tensor of shape (1, 1, height, width) with patch indices
    """
    ph, pw = patch_size
    n_patches_h = height // ph
    n_patches_w = width // pw
    
    idx = torch.arange(n_patches_h * n_patches_w).float()
    idx = idx.view(1, 1, n_patches_h, n_patches_w)
    
    # Resize to full image size
    mask = F.interpolate(idx, size=(height, width), mode='nearest')
    return mask.long()