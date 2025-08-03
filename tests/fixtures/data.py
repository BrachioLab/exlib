import torch
import numpy as np
from typing import Tuple

def get_test_image(size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Get a test image tensor.
    
    Args:
        size: Image size (H, W)
        
    Returns:
        Tensor of shape (3, H, W) with values in [0, 1]
    """
    # Create a simple gradient image
    h, w = size
    x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
    y = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
    
    r = x
    g = y
    b = 1 - (x + y) / 2
    
    img = torch.stack([r, g, b], dim=0)
    return img.clamp(0, 1)

def get_test_text_inputs(seq_len: int = 20, vocab_size: int = 1000) -> torch.Tensor:
    """Get test text input IDs.
    
    Args:
        seq_len: Sequence length
        vocab_size: Vocabulary size
        
    Returns:
        Tensor of shape (seq_len,) with token IDs
    """
    # Avoid special tokens (0, 1, 2 typically)
    return torch.randint(3, vocab_size, (seq_len,))