# Feature Attribution Methods Refactoring Plan

## Overview

This document provides implementation instructions for refactoring the feature attribution methods from `src/exlib/explainers/` into a cleaner implementation in `src/exlib/new_explainers/`. The design prioritizes simplicity and low abstraction.

## Implementation Instructions

This plan is designed to be executed by a coding agent. Follow the steps in order, creating files and tests as specified. Each section contains complete code templates and specific instructions.

## Goals

1. **Minimize External Dependencies**: Preserve the approach of vendoring external libraries
2. **Low Abstraction**: Simple, direct API without complex inheritance hierarchies
3. **Reduce Code Duplication**: Share utilities where it makes sense
4. **Improve Type Safety**: Add type hints for clarity
5. **Preserve Functionality**: Ensure all existing features work

## Directory Structure

Create the following directory structure:

```
src/exlib/new_explainers/
├── __init__.py
├── lime.py              # LimeImage and LimeText classes
├── shap.py              # ShapImage and ShapText classes
├── intgrad.py           # IntGradImage and IntGradText classes
├── gradcam.py           # GradCAMImage and GradCAMText classes
├── mfaba.py             # MfabaImage and MfabaText classes
├── rise.py              # RiseImage and RiseText classes
├── archipelago.py       # ArchipelagoImage class
├── utils/
│   ├── __init__.py
│   ├── common.py        # Shared utilities (tensor conversions, etc.)
│   ├── batch.py         # Batch processing utilities
│   └── masking.py       # Mask utilities
└── vendor/              # Vendored external dependencies
    ├── __init__.py
    ├── archipelago/
    ├── craft/
    ├── fullgrad/
    ├── lime/
    └── saliency/

tests/
├── __init__.py
├── test_intgrad.py
├── test_lime.py
├── test_shap.py
├── test_gradcam.py
├── test_mfaba.py
├── test_rise.py
├── test_archipelago.py
├── test_utils.py
└── fixtures/          # Test data and models
    ├── __init__.py
    ├── models.py      # Simple test models
    └── data.py        # Test inputs
```

### API Design

#### Simple Explainer Classes

Each explainer will be a simple class with an `explain` method:

```python
# lime.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LimeExplanation:
    """LIME-specific explanation results"""
    attributions: torch.Tensor
    metadata: Dict[str, Any] = None
    segments: Optional[np.ndarray] = None  # LIME segmentation mask
    local_exp: Optional[Dict] = None  # LIME's local explanation dict

class LimeImage:
    """LIME explainer for images"""
    
    def __init__(self, num_samples: int = 500, **kwargs):
        self.num_samples = num_samples
        self.kwargs = kwargs
    
    def explain(
        self, 
        model: nn.Module, 
        input: torch.Tensor,
        target: Optional[int] = None,
        **kwargs
    ) -> LimeExplanation:
        """
        Explain model prediction on input
        
        Args:
            model: PyTorch model to explain
            input: Input tensor of shape (C, H, W)
            target: Target class to explain (if None, uses predicted class)
            **kwargs: Additional method-specific parameters
            
        Returns:
            LimeExplanation object with attributions and metadata
        """
        # Implementation
        pass

class LimeText:
    """LIME explainer for text"""
    
    def __init__(self, tokenizer, num_samples: int = 500, **kwargs):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.kwargs = kwargs
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,  # Can be token IDs or embeddings
        target: Optional[int] = None,
        **kwargs
    ) -> LimeExplanation:
        """
        Explain model prediction on text input
        
        Args:
            model: PyTorch model to explain
            input: Input tensor (token IDs or embeddings)
            target: Target class to explain
            **kwargs: Additional parameters
            
        Returns:
            LimeExplanation object with attributions and metadata
        """
        # Implementation
        pass
```

#### Usage Examples

```python
import exlib.new_explainers as explainers

# Image example
model = resnet50.from_pretrained(...)
input = dataset[0]  # (C, H, W) tensor

lime_explainer = explainers.LimeImage(num_samples=1000)
explanation = lime_explainer.explain(model, input, target=5)

# Access results
attributions = explanation.attributions  # (H, W) tensor
metadata = explanation.metadata  # Dict with LIME-specific info

# Text example
model = BertForSequenceClassification.from_pretrained(...)
tokenizer = BertTokenizer.from_pretrained(...)
input_ids = tokenizer.encode("Some text")

lime_explainer = explainers.LimeText(tokenizer, num_samples=500)
explanation = lime_explainer.explain(model, input_ids, target=1)

# Access results
attributions = explanation.attributions  # (seq_len,) tensor
```

## Implementation Details

### Common Utilities

The `utils/` directory will contain minimal shared functionality:

```python
# utils/common.py
import torch
import numpy as np

def torch_img_to_np(x: torch.Tensor) -> np.ndarray:
    """Convert torch image tensor to numpy array"""
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1).numpy()
    elif x.dim() == 3:
        return x.permute(1, 2, 0).numpy()
    else:
        raise ValueError("Image tensor must have 3 or 4 dimensions")

def np_to_torch_img(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch image tensor"""
    x = torch.from_numpy(x)
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        return x.permute(2, 0, 1)
    else:
        raise ValueError("Image array must have 3 or 4 dimensions")

# utils/batch.py
def process_in_batches(
    items: list,
    process_fn: callable,
    batch_size: int = 16,
    show_progress: bool = False
):
    """Simple batch processing utility"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results.extend(process_fn(batch))
    return results
```

### Method Implementation Pattern

Each explainer follows a simple pattern with method-specific explanation types:

```python
# intgrad.py
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from .utils.common import *

@dataclass
class IntGradExplanation:
    attributions: torch.Tensor
    metadata: dict = None
    convergence_delta: Optional[float] = None  # IntGrad-specific field

class IntGradImage:
    """Integrated Gradients for images"""
    
    def __init__(self, n_steps: int = 50, baseline: str = "zero"):
        self.n_steps = n_steps
        self.baseline = baseline
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        **kwargs
    ) -> IntGradExplanation:
        """Generate integrated gradients explanation"""
        # If no target specified, use model prediction
        if target is None:
            with torch.no_grad():
                pred = model(input.unsqueeze(0))
                target = pred.argmax(dim=1).item()
        
        # Create baseline
        if self.baseline == "zero":
            baseline = torch.zeros_like(input)
        else:
            baseline = self.baseline
        
        # Compute integrated gradients
        input = input.requires_grad_()
        attributions = self._compute_gradients(
            model, input, baseline, target
        )
        
        return IntGradExplanation(
            attributions=attributions,
            metadata={"target": target, "n_steps": self.n_steps}
        )
    
    def _compute_gradients(self, model, input, baseline, target):
        # Implementation details
        pass
```

### Vendor Management

Vendored code will be minimal and documented:

```python
# vendor/lime/__init__.py
"""
Vendored from lime v0.2.0.1
Only includes core functionality needed for image/text explanation
Modified files:
- lime_base.py: Removed matplotlib dependencies
- lime_image.py: Simplified segmentation interface
"""
```

## Step-by-Step Implementation Guide

### Step 1: Create Directory Structure and Base Files

1. Create all directories as shown above
2. Create empty `__init__.py` files in each directory
3. Create the main `__init__.py` file:

```python
# src/exlib/new_explainers/__init__.py
from .intgrad import IntGradImage, IntGradText, IntGradExplanation
from .lime import LimeImage, LimeText, LimeExplanation
from .shap import ShapImage, ShapText, ShapExplanation
from .gradcam import GradCAMImage, GradCAMText, GradCAMExplanation
from .mfaba import MfabaImage, MfabaText, MfabaExplanation
from .rise import RiseImage, RiseText, RiseExplanation
from .archipelago import ArchipelagoImage, ArchipelagoExplanation

__all__ = [
    # Integrated Gradients
    'IntGradImage', 'IntGradText', 'IntGradExplanation',
    # LIME
    'LimeImage', 'LimeText', 'LimeExplanation',
    # SHAP
    'ShapImage', 'ShapText', 'ShapExplanation',
    # GradCAM
    'GradCAMImage', 'GradCAMText', 'GradCAMExplanation',
    # MFABA
    'MfabaImage', 'MfabaText', 'MfabaExplanation',
    # RISE
    'RiseImage', 'RiseText', 'RiseExplanation',
    # Archipelago
    'ArchipelagoImage', 'ArchipelagoExplanation',
]
```

### Step 2: Implement Utilities

Create the utility files with the exact code below:

```python
# src/exlib/new_explainers/utils/__init__.py
from .common import torch_img_to_np, np_to_torch_img
from .batch import process_in_batches
from .masking import convert_idx_masks_to_bool, create_patch_mask

__all__ = [
    'torch_img_to_np', 'np_to_torch_img',
    'process_in_batches',
    'convert_idx_masks_to_bool', 'create_patch_mask'
]
```

```python
# src/exlib/new_explainers/utils/common.py
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
```

```python
# src/exlib/new_explainers/utils/batch.py
from typing import List, Callable, Any, Optional
from tqdm import tqdm
import torch

def process_in_batches(
    items: List[Any],
    process_fn: Callable,
    batch_size: int = 16,
    show_progress: bool = False,
    desc: Optional[str] = None
) -> List[Any]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        process_fn: Function that processes a batch of items
        batch_size: Size of each batch
        show_progress: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of processed results
    """
    results = []
    
    # Create progress bar if requested
    if show_progress:
        pbar = tqdm(range(0, len(items), batch_size), desc=desc)
    else:
        pbar = range(0, len(items), batch_size)
    
    for i in pbar:
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results
```

```python
# src/exlib/new_explainers/utils/masking.py
import torch
import torch.nn.functional as F
from typing import Tuple

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
```

### Step 3: Create Test Fixtures

```python
# tests/fixtures/__init__.py
from .models import get_vision_model, get_text_model
from .data import get_test_image, get_test_text_inputs

__all__ = ['get_vision_model', 'get_text_model', 'get_test_image', 'get_test_text_inputs']
```

```python
# tests/fixtures/models.py
import torch
import torch.nn as nn
from typing import Tuple, Union

def get_vision_model(model_name: str = "resnet50") -> Tuple[nn.Module, dict]:
    """Get a vision model for testing.
    
    Args:
        model_name: Either "resnet50" or "vit"
        
    Returns:
        Tuple of (model, config) where config contains model-specific info
    """
    if model_name == "resnet50":
        from torchvision import models
        model = models.resnet50(pretrained=False)
        model.eval()
        config = {
            "input_size": (224, 224),
            "num_classes": 1000,
            "output_type": "tensor"  # Direct logits
        }
        return model, config
        
    elif model_name == "vit":
        from transformers import ViTForImageClassification, ViTConfig
        # Use a smaller ViT for faster testing
        config_vit = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_labels=1000
        )
        model = ViTForImageClassification(config_vit)
        model.eval()
        config = {
            "input_size": (224, 224),
            "num_classes": 1000,
            "output_type": "dataclass"  # Returns ModelOutput dataclass
        }
        return model, config
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_text_model(model_name: str = "gpt2") -> Tuple[nn.Module, dict]:
    """Get a text model for testing.
    
    Args:
        model_name: Currently only "gpt2" supported
        
    Returns:
        Tuple of (model, config) where config contains model-specific info
    """
    if model_name == "gpt2":
        from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer
        
        # Create a small GPT2 for testing
        config_gpt2 = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            num_labels=2,  # Binary classification
            pad_token_id=50256
        )
        model = GPT2ForSequenceClassification(config_gpt2)
        model.eval()
        
        # Also get tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        config = {
            "tokenizer": tokenizer,
            "max_length": 512,
            "num_classes": 2,
            "embedding_layer": model.transformer.wte,  # Word token embeddings
            "output_type": "dataclass"
        }
        return model, config
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Helper function to extract logits from model outputs
def get_model_logits(output: Union[torch.Tensor, object]) -> torch.Tensor:
    """Extract logits from model output.
    
    Args:
        output: Either a tensor (ResNet) or dataclass (ViT/GPT2)
        
    Returns:
        Logits tensor
    """
    if isinstance(output, torch.Tensor):
        return output
    else:
        # Handle HuggingFace model outputs
        return output.logits
```

```python
# tests/fixtures/data.py
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
```

### Step 4: Implement Integrated Gradients (First Method)

This is the complete implementation for Integrated Gradients. Copy this exactly:

```python
# src/exlib/new_explainers/intgrad.py
import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class IntGradExplanation:
    """Integrated Gradients explanation results."""
    attributions: torch.Tensor
    metadata: dict = None
    convergence_delta: Optional[float] = None

class IntGradImage:
    """Integrated Gradients explainer for images."""
    
    def __init__(
        self, 
        n_steps: int = 50,
        baseline: Union[str, torch.Tensor, Callable] = "zero"
    ):
        """
        Args:
            n_steps: Number of integration steps
            baseline: Baseline for integration. Can be:
                - "zero": Black image
                - "mean": Dataset mean (will use 0.5 for each channel)
                - torch.Tensor: Custom baseline image
                - Callable: Function that takes input and returns baseline
        """
        self.n_steps = n_steps
        self.baseline = baseline
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_convergence_delta: bool = False
    ) -> IntGradExplanation:
        """Generate integrated gradients explanation.
        
        Args:
            model: PyTorch model to explain
            input: Input tensor of shape (C, H, W)
            target: Target class (if None, uses predicted class)
            return_convergence_delta: Whether to compute convergence check
            
        Returns:
            IntGradExplanation with attributions
        """
        # Ensure input requires grad
        input = input.detach().requires_grad_(True)
        
        # Add batch dimension if needed
        if input.dim() == 3:
            input_batch = input.unsqueeze(0)
        else:
            input_batch = input
            
        # Get target class if not provided
        if target is None:
            with torch.no_grad():
                output = model(input_batch)
                # Handle both tensor and dataclass outputs
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                target = logits.argmax(dim=1).item()
        
        # Create baseline
        baseline = self._get_baseline(input)
        
        # Compute integrated gradients
        attributions = self._integrate_gradients(
            model, input, baseline, target
        )
        
        # Compute convergence delta if requested
        convergence_delta = None
        if return_convergence_delta:
            convergence_delta = self._compute_convergence_delta(
                model, input, baseline, attributions, target
            )
        
        return IntGradExplanation(
            attributions=attributions,
            metadata={
                "target": target,
                "n_steps": self.n_steps,
                "baseline_type": self.baseline if isinstance(self.baseline, str) else "custom"
            },
            convergence_delta=convergence_delta
        )
    
    def _get_baseline(self, input: torch.Tensor) -> torch.Tensor:
        """Get baseline for given input."""
        if isinstance(self.baseline, str):
            if self.baseline == "zero":
                return torch.zeros_like(input)
            elif self.baseline == "mean":
                return torch.ones_like(input) * 0.5
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline}")
        elif isinstance(self.baseline, torch.Tensor):
            return self.baseline
        elif callable(self.baseline):
            return self.baseline(input)
        else:
            raise ValueError(f"Invalid baseline type: {type(self.baseline)}")
    
    def _integrate_gradients(
        self,
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        target: int
    ) -> torch.Tensor:
        """Compute integrated gradients."""
        # Initialize accumulator
        integrated_grads = torch.zeros_like(input)
        
        # Compute gradients for each step
        for step in range(self.n_steps):
            # Interpolate between baseline and input
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (input - baseline)
            interpolated = interpolated.detach().requires_grad_(True)
            
            # Forward pass
            if interpolated.dim() == 3:
                output = model(interpolated.unsqueeze(0))
            else:
                output = model(interpolated)
            
            # Handle both tensor and dataclass outputs
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Get gradient for target class
            model.zero_grad()
            logits[0, target].backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad / self.n_steps
        
        # Multiply by input - baseline
        integrated_grads *= (input - baseline)
        
        return integrated_grads
    
    def _compute_convergence_delta(
        self,
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        attributions: torch.Tensor,
        target: int
    ) -> float:
        """Compute convergence delta for sanity check."""
        # Get model outputs
        with torch.no_grad():
            if input.dim() == 3:
                out_input = model(input.unsqueeze(0))
                out_baseline = model(baseline.unsqueeze(0))
            else:
                out_input = model(input)
                out_baseline = model(baseline)
            
            # Handle dataclass outputs
            if hasattr(out_input, 'logits'):
                input_output = out_input.logits[0, target]
                baseline_output = out_baseline.logits[0, target]
            else:
                input_output = out_input[0, target]
                baseline_output = out_baseline[0, target]
        
        # Compute sum of attributions
        attr_sum = attributions.sum()
        
        # Compute expected difference
        expected_diff = input_output - baseline_output
        
        # Return absolute difference
        return abs(attr_sum - expected_diff).item()


class IntGradText:
    """Integrated Gradients explainer for text."""
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        n_steps: int = 50,
        baseline: Union[str, int] = "pad"
    ):
        """
        Args:
            embedding_layer: The embedding layer of the model
            n_steps: Number of integration steps
            baseline: Baseline token ID or "pad" for padding token (usually 0)
        """
        self.embedding_layer = embedding_layer
        self.n_steps = n_steps
        self.baseline = baseline
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_convergence_delta: bool = False
    ) -> IntGradExplanation:
        """Generate integrated gradients explanation for text.
        
        Args:
            model: PyTorch model to explain
            input: Input tensor of token IDs, shape (seq_len,)
            target: Target class (if None, uses predicted class)
            return_convergence_delta: Whether to compute convergence check
            
        Returns:
            IntGradExplanation with attributions
        """
        # Ensure we have token IDs
        if input.dtype not in [torch.long, torch.int]:
            raise ValueError("Input must be token IDs (long or int tensor)")
        
        # Add batch dimension if needed
        if input.dim() == 1:
            input_batch = input.unsqueeze(0)
        else:
            input_batch = input
        
        # Get embeddings
        input_embeds = self.embedding_layer(input_batch)
        
        # Get target class if not provided
        if target is None:
            with torch.no_grad():
                output = model(input_batch)
                # Handle both tensor and dataclass outputs
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                target = logits.argmax(dim=1).item()
        
        # Create baseline embeddings
        baseline_embeds = self._get_baseline_embeds(input_batch)
        
        # Compute integrated gradients on embeddings
        embeds_grads = self._integrate_gradients_embeds(
            model, input_batch, input_embeds, baseline_embeds, target
        )
        
        # Average across embedding dimension to get token-level attributions
        attributions = embeds_grads.norm(dim=-1).squeeze(0)
        
        # Compute convergence delta if requested
        convergence_delta = None
        if return_convergence_delta:
            convergence_delta = self._compute_convergence_delta(
                model, input_batch, baseline_embeds, embeds_grads, target
            )
        
        return IntGradExplanation(
            attributions=attributions,
            metadata={
                "target": target,
                "n_steps": self.n_steps,
                "baseline": self.baseline
            },
            convergence_delta=convergence_delta
        )
    
    def _get_baseline_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get baseline embeddings."""
        if self.baseline == "pad":
            baseline_ids = torch.zeros_like(input_ids)
        elif isinstance(self.baseline, int):
            baseline_ids = torch.full_like(input_ids, self.baseline)
        else:
            raise ValueError(f"Invalid baseline: {self.baseline}")
        
        return self.embedding_layer(baseline_ids)
    
    def _integrate_gradients_embeds(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        target: int
    ) -> torch.Tensor:
        """Compute integrated gradients on embeddings."""
        integrated_grads = torch.zeros_like(input_embeds)
        
        for step in range(self.n_steps):
            # Interpolate embeddings
            alpha = step / self.n_steps
            interpolated_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interpolated_embeds = interpolated_embeds.detach().requires_grad_(True)
            
            # Forward pass with interpolated embeddings
            output = model.forward(inputs_embeds=interpolated_embeds)
            
            # Handle dataclass outputs
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Get gradient
            model.zero_grad()
            logits[0, target].backward()
            
            # Accumulate
            integrated_grads += interpolated_embeds.grad / self.n_steps
        
        # Multiply by difference
        integrated_grads *= (input_embeds - baseline_embeds)
        
        return integrated_grads
    
    def _compute_convergence_delta(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        baseline_embeds: torch.Tensor,
        embeds_grads: torch.Tensor,
        target: int
    ) -> float:
        """Compute convergence delta."""
        with torch.no_grad():
            out_input = model(input_ids)
            out_baseline = model.forward(inputs_embeds=baseline_embeds)
            
            # Handle dataclass outputs
            if hasattr(out_input, 'logits'):
                input_output = out_input.logits[0, target]
                baseline_output = out_baseline.logits[0, target]
            else:
                input_output = out_input[0, target]
                baseline_output = out_baseline[0, target]
        
        attr_sum = embeds_grads.sum()
        expected_diff = input_output - baseline_output
        
        return abs(attr_sum - expected_diff).item()
```

### Step 5: Create Tests for Integrated Gradients

```python
# tests/test_intgrad.py
import pytest
import torch
import torch.nn as nn
from src.exlib.new_explainers import IntGradImage, IntGradText, IntGradExplanation
from tests.fixtures import get_vision_model, get_text_model, get_test_image, get_test_text_inputs

class TestIntGradImage:
    """Test IntGradImage explainer."""
    
    @pytest.mark.parametrize("model_name", ["resnet50", "vit"])
    def test_basic_explanation(self, model_name):
        """Test basic explanation generation with real models."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = IntGradImage(n_steps=10)
        
        explanation = explainer.explain(model, input)
        
        # Check output type
        assert isinstance(explanation, IntGradExplanation)
        
        # Check attribution shape
        assert explanation.attributions.shape == input.shape
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_steps"] == 10
    
    @pytest.mark.parametrize("model_name", ["resnet50", "vit"])
    def test_with_target(self, model_name):
        """Test explanation with specific target."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = IntGradImage(n_steps=10)
        
        # Explain for specific target
        target = 2
        explanation = explainer.explain(model, input, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_different_baselines(self):
        """Test different baseline types."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        # Test zero baseline
        explainer_zero = IntGradImage(n_steps=10, baseline="zero")
        exp_zero = explainer_zero.explain(model, input)
        
        # Test mean baseline
        explainer_mean = IntGradImage(n_steps=10, baseline="mean")
        exp_mean = explainer_mean.explain(model, input)
        
        # Attributions should be different
        assert not torch.allclose(exp_zero.attributions, exp_mean.attributions)
    
    @pytest.mark.parametrize("model_name", ["resnet50", "vit"])
    def test_convergence_delta(self, model_name):
        """Test convergence delta computation."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = IntGradImage(n_steps=50)
        
        explanation = explainer.explain(
            model, input, return_convergence_delta=True
        )
        
        assert explanation.convergence_delta is not None
        assert isinstance(explanation.convergence_delta, float)
        # Should be small for sufficient steps
        assert explanation.convergence_delta < 10.0  # More lenient for larger models
    
    def test_custom_baseline(self):
        """Test with custom baseline tensor."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        baseline = torch.ones_like(input) * 0.3
        
        explainer = IntGradImage(n_steps=10, baseline=baseline)
        explanation = explainer.explain(model, input)
        
        assert explanation.attributions.shape == input.shape


class TestIntGradText:
    """Test IntGradText explainer."""
    
    def test_basic_explanation(self):
        """Test basic text explanation with GPT2."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "This is a test sentence for explanation"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        
        explainer = IntGradText(
            embedding_layer=config["embedding_layer"],
            n_steps=10
        )
        
        explanation = explainer.explain(model, input_ids)
        
        # Check output
        assert isinstance(explanation, IntGradExplanation)
        assert explanation.attributions.shape == (input_ids.shape[0],)  # seq_len
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_steps"] == 10
    
    def test_with_target(self):
        """Test with specific target class."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "This is another test"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = IntGradText(
            embedding_layer=config["embedding_layer"],
            n_steps=10
        )
        
        target = 1
        explanation = explainer.explain(model, input_ids, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_convergence_delta(self):
        """Test convergence for text."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing convergence"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = IntGradText(
            embedding_layer=config["embedding_layer"],
            n_steps=50
        )
        
        explanation = explainer.explain(
            model, input_ids, return_convergence_delta=True
        )
        
        assert explanation.convergence_delta is not None
        assert isinstance(explanation.convergence_delta, float)


if __name__ == "__main__":
    pytest.main([__file__])
```

### Step 6: Vendor External Dependencies

For each method that requires external libraries, copy only the necessary files:

```python
# src/exlib/new_explainers/vendor/README.md
# Vendored Dependencies

This directory contains minimal vendored code from external libraries to avoid
requiring additional pip installations.

## Structure

- `lime/`: Core LIME functionality (from lime 0.2.0.1)
- `shap/`: Core SHAP functionality (from shap 0.41.0)
- `archipelago/`: Archipelago explainer (from custom implementation)
- `saliency/`: Saliency methods including MFABA variants

## Modifications

Each vendored module includes a README describing:
1. Source version
2. Files included
3. Modifications made
4. Original license

## Adding New Vendored Code

When vendoring new code:
1. Copy only essential files
2. Remove unnecessary dependencies
3. Document all changes
4. Include original license
5. Test thoroughly
```

### Step 7: Implementation Order for Remaining Methods

Implement the remaining methods in this order:

1. **RISE** (Step 8) - Simple, minimal dependencies
2. **LIME** (Step 9) - Requires vendoring lime library
3. **SHAP** (Step 10) - Requires vendoring shap components
4. **GradCAM** (Step 11) - Requires layer hooks
5. **MFABA** (Step 12) - Requires saliency library vendoring
6. **Archipelago** (Step 13) - Most complex vendoring

Each method should follow the same pattern as IntGrad above.

## Final Steps

### Step 14: Integration Testing

Create integration tests that compare new implementations with original:

```python
# tests/test_integration.py
import torch
from src.exlib.explainers import IntGradImageCls as OldIntGrad
from src.exlib.new_explainers import IntGradImage as NewIntGrad

def test_intgrad_compatibility():
    """Test that new IntGrad produces similar results to old."""
    # Implementation comparing old vs new
    pass
```

### Step 15: Performance Benchmarking

```python
# tests/test_performance.py
import time
import torch
from src.exlib.new_explainers import IntGradImage

def benchmark_intgrad():
    """Benchmark IntGrad performance."""
    # Time various input sizes
    pass
```

## Success Verification

After implementation, verify:

1. All tests pass: `pytest tests/`
2. No import errors: `python -c "import exlib.new_explainers"`
3. Attributions are reasonable (non-zero, correct shape)
4. Performance is comparable to original
5. Memory usage is not excessive

## Notes for Implementation

- Start with IntGrad as shown above
- Test each method thoroughly before moving to the next
- Keep vendored code minimal
- Document any deviations from original behavior
- Ensure all methods follow the same API pattern

## Important: Handling Different Model Types

The implementation above handles both standard PyTorch models (like ResNet50) and HuggingFace models (like ViT and GPT2) which return dataclass outputs:

1. **Vision Models**:
   - ResNet50: Returns tensor directly
   - ViT: Returns ModelOutput with `.logits` attribute

2. **Text Models**:
   - GPT2: Returns ModelOutput with `.logits` attribute
   - Requires `inputs_embeds` parameter for embedding interpolation

3. **Key Pattern**:
   ```python
   # Always check for dataclass output
   if hasattr(output, 'logits'):
       logits = output.logits
   else:
       logits = output
   ```

Apply this pattern consistently across all explainer implementations.

### 1. Simple API
- Each explainer is a standalone class
- Single `explain` method as entry point
- Model passed to `explain`, not constructor
- Minimal abstraction layers

### 2. Method-Specific Outputs
- Each method has its own explanation dataclass (e.g., `LimeExplanation`, `IntGradExplanation`)
- All explanation classes have `attributions` tensor
- `metadata` dict contains method-specific information
- No inheritance between explanation types
- Allows method-specific fields without bloating a common interface

### 3. Minimal Dependencies
- Vendor only essential external code
- Prefer reimplementation for simple algorithms
- Document all vendored code origins
- Keep vendor code isolated

### 4. Practical Defaults
- Sensible default parameters
- Auto-detect target class when not specified
- Handle both single inputs and batches
- Clear error messages

## Testing Strategy

### Simple Test Pattern

```python
# tests/test_intgrad.py
import torch
import pytest
from exlib.new_explainers import IntGradImage

def test_intgrad_basic():
    # Simple CNN model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 10, 3),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(10, 2)
    )
    
    # Random input
    input = torch.randn(3, 32, 32)
    
    # Create explainer and explain
    explainer = IntGradImage(n_steps=10)
    explanation = explainer.explain(model, input)
    
    # Check output
    assert explanation.attributions.shape == (3, 32, 32)
    assert "target" in explanation.metadata
```

### Integration Tests

Compare outputs with original implementation:

```python
def test_compare_with_original():
    # Load same model and input
    model, input = load_test_case()
    
    # Original implementation
    from exlib.explainers import IntGradImageCls
    original = IntGradImageCls(model, num_steps=50)
    orig_result = original(input.unsqueeze(0), torch.tensor([0]))
    
    # New implementation
    from exlib.new_explainers import IntGradImage
    new = IntGradImage(n_steps=50)
    new_result = new.explain(model, input, target=0)
    
    # Compare attributions
    torch.testing.assert_close(
        orig_result.attributions,
        new_result.attributions.unsqueeze(0),
        rtol=1e-4
    )
```

## Migration Guide

### For Users

Old API:
```python
from exlib.explainers import LimeImageCls
explainer = LimeImageCls(model, num_samples=500)
result = explainer(x, t, return_groups=False)
```

New API:
```python
from exlib.new_explainers import LimeImage
explainer = LimeImage(num_samples=500)
result = explainer.explain(model, x[0], target=t[0])
```

Key differences:
1. Model passed to `explain`, not constructor
2. Single input instead of batch
3. Cleaner parameter names
4. Consistent return type

### Compatibility Wrapper

For gradual migration:

```python
# compat.py
class LimeImageClsCompat:
    """Compatibility wrapper for old API"""
    
    def __init__(self, model, **kwargs):
        self.model = model
        self.explainer = LimeImage(**kwargs)
    
    def __call__(self, x, t, **kwargs):
        # Handle batch inputs
        results = []
        for i in range(x.shape[0]):
            exp = self.explainer.explain(
                self.model, 
                x[i], 
                target=t[i].item()
            )
            results.append(exp.attributions)
        
        # Return in old format
        attrs = torch.stack(results)
        return FeatureAttrOutput(attrs, {})
```

## Success Criteria

1. **API Simplicity**: Can explain in 3 lines of code
2. **Performance**: No regression vs original
3. **Maintainability**: Each file < 500 lines
4. **Testing**: > 90% code coverage
5. **Documentation**: Every public method documented

## Timeline

- **Week 1**: Setup, utilities, and IntGrad
- **Week 2**: LIME and SHAP
- **Week 3**: GradCAM, RISE, and others
- **Week 4**: Testing and documentation
- **Week 5**: Performance optimization
- **Week 6**: Migration guide and release
