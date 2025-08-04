"""SHAP (SHapley Additive exPlanations) implementation.

Low-abstraction implementation following the pattern:
    explainer = ShapImage(configs)
    explanation = explainer.explain(model, input, target)
"""

from dataclasses import dataclass
from typing import Optional, Union, Callable, List
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations


@dataclass
class ShapExplanation:
    """SHAP-specific explanation output."""
    attributions: torch.Tensor
    segments: torch.Tensor  # Same shape as input, values in [0, num_segments-1]
    expected_value: float  # Expected output value
    base_value: Optional[float] = None  # Actual baseline output
    coalitions: Optional[dict] = None  # Coalition information for debugging
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ShapImage:
    """SHAP explainer for image models using KernelSHAP approach.
    
    Usage:
        explainer = ShapImage(n_samples=100, patch_size=16)
        explanation = explainer.explain(model, image, target=5)
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        baseline: Union[str, torch.Tensor] = "zero",
        segmentation_fn: Optional[Callable] = None,
        patch_size: int = 16,
        random_seed: int = 42
    ):
        self.n_samples = n_samples
        self.baseline = baseline
        self.segmentation_fn = segmentation_fn
        self.patch_size = patch_size
        self.random_seed = random_seed
    
    def _default_segmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Use patch-based segmentation by default."""
        from .utils.masking import patch_segment_image
        return patch_segment_image(image, self.patch_size)
    
    def _get_baseline_image(self, image: torch.Tensor) -> torch.Tensor:
        """Get baseline image based on configuration."""
        if isinstance(self.baseline, torch.Tensor):
            return self.baseline
        elif self.baseline == "zero":
            return torch.zeros_like(image)
        elif self.baseline == "mean":
            return torch.ones_like(image) * image.mean()
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline}")
    
    def _kernel_shap_weights(self, n_features: int, coalitions: np.ndarray) -> np.ndarray:
        """Compute KernelSHAP weights for coalitions."""
        weights = []
        for coalition in coalitions:
            k = coalition.sum()
            if k == 0 or k == n_features:
                # Special case for empty and full coalitions
                weights.append(1000.0)  # Large weight
            else:
                # SHAP kernel weight
                weight = (n_features - 1) / (k * (n_features - k))
                weights.append(weight)
        return np.array(weights)
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_coalitions: bool = False,
        batch_size: int = 16
    ) -> ShapExplanation:
        """Generate SHAP explanation for an image.
        
        Args:
            model: PyTorch model to explain
            input: Input image tensor (C, H, W) or (B, C, H, W)
            target: Target class index. If None, uses predicted class
            return_coalitions: Whether to return coalition information
            
        Returns:
            ShapExplanation with attributions and expected value
        """
        # Handle batch dimension
        if input.dim() == 4:
            input = input[0]
        
        original_mode = model.training
        model.eval()
        
        # Get segmentation
        if self.segmentation_fn is not None:
            segments = self.segmentation_fn(input)
        else:
            segments = self._default_segmentation(input)
        n_features = segments.max().item() + 1
        
        # Get baseline
        baseline_img = self._get_baseline_image(input)
        
        # Get original and baseline predictions
        with torch.no_grad():
            # Original prediction
            output = model(input.unsqueeze(0))
            logits = getattr(output, 'logits', output)
            
            if target is None:
                target = logits.argmax(dim=1).item()
            
            original_prob = torch.softmax(logits, dim=1)[0, target].item()
            
            # Baseline prediction
            baseline_output = model(baseline_img.unsqueeze(0))
            if hasattr(baseline_output, 'logits'):
                baseline_logits = baseline_output.logits
            else:
                baseline_logits = baseline_output
            baseline_prob = torch.softmax(baseline_logits, dim=1)[0, target].item()
        
        # Generate random coalitions
        np.random.seed(self.random_seed)
        coalitions = np.random.binomial(1, 0.5, size=(self.n_samples, n_features))
        
        # Always include empty and full coalitions
        coalitions[0, :] = 0  # Empty coalition
        coalitions[1, :] = 1  # Full coalition
        
        # Get predictions for coalitions
        masked_images = []

        for i in range(self.n_samples):
            coalition = coalitions[i]
            masked = baseline_img.clone()
            for seg_id in range(n_features):
                if coalition[seg_id] == 1:
                    masked[:, segments == seg_id] = input[:, segments == seg_id]
            masked_images.append(masked)

        # Stack all masked images into a batch
        masked_images_tensor = torch.stack(masked_images, dim=0).to(input.device)

        predictions = []
        with torch.no_grad():
            for batch in torch.split(masked_images_tensor, batch_size, dim=0):
                output = model(batch)
                logits = getattr(output, 'logits', output)
                probs = torch.softmax(logits, dim=1)[:, target]
                predictions.extend(probs.cpu().numpy())

        predictions = np.array(predictions)
        
        # Compute KernelSHAP weights
        weights = self._kernel_shap_weights(n_features, coalitions)
        
        # Fit weighted linear regression
        # Add intercept term
        X = np.column_stack([np.ones(self.n_samples), coalitions])
        
        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ predictions
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            coeffs = np.linalg.pinv(XtWX) @ XtWy
        
        # Extract SHAP values (exclude intercept)
        shap_values = coeffs[1:]
        expected_value = coeffs[0]
        
        # Create attribution map
        attribution_map = torch.zeros_like(input[0])  # Single channel
        for seg_id in range(n_features):
            attribution_map[segments == seg_id] = shap_values[seg_id]
        
        # Expand to match input channels
        attributions = attribution_map.unsqueeze(0).expand_as(input)
        
        model.train(original_mode)
        
        # Expand segments to match input shape
        # segments is (H, W), need to expand to (C, H, W)
        segments_expanded = segments.unsqueeze(0).expand(input.shape[0], -1, -1)
        
        # Build explanation
        explanation = ShapExplanation(
            attributions=attributions,
            segments=segments_expanded,
            expected_value=expected_value,
            base_value=baseline_prob,
            coalitions={'coalitions': coalitions, 'predictions': predictions} if return_coalitions else None,
            metadata={
                'target': target,
                'original_probability': original_prob,
                'n_samples': self.n_samples,
                'baseline_type': self.baseline if isinstance(self.baseline, str) else 'custom',
                'num_segments': n_features,
                'patch_size': self.patch_size
            }
        )
        
        return explanation


class ShapText:
    """SHAP explainer for text models using sampling approach.
    
    Usage:
        explainer = ShapText(embedding_layer, n_samples=100)
        explanation = explainer.explain(model, input_ids, target=1)
    """
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        n_samples: int = 500,
        mask_token_id: int = 0,
        random_seed: int = 42
    ):
        self.embedding_layer = embedding_layer
        self.n_samples = n_samples
        self.mask_token_id = mask_token_id
        self.random_seed = random_seed
    
    def _shapley_kernel_weight(self, M: int, S: int) -> float:
        """Compute Shapley kernel weight."""
        if S == 0 or S == M:
            return 1000.0  # Large weight for empty/full coalitions
        return (M - 1) / (S * (M - S))
    
    def explain(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        target: Optional[int] = None,
        return_coalitions: bool = False,
        batch_size: int = 16
    ) -> ShapExplanation:
        """Generate SHAP explanation for text.
        
        Args:
            model: PyTorch model to explain
            input_ids: Input token IDs (seq_len,) or (batch_size, seq_len)
            target: Target class index. If None, uses predicted class
            return_coalitions: Whether to return coalition information
            
        Returns:
            ShapExplanation with per-token attributions
        """
        # Handle batch dimension
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        
        device = input_ids.device
        seq_len = input_ids.shape[0]
        original_mode = model.training
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            output = model(input_ids.unsqueeze(0))
            logits = getattr(output, 'logits', output)
            
            if target is None:
                target = logits.argmax(dim=1).item()
            
            original_prob = torch.softmax(logits, dim=1)[0, target].item()
            
            # Baseline prediction (all masked)
            baseline_ids = torch.full_like(input_ids, self.mask_token_id)
            baseline_output = model(baseline_ids.unsqueeze(0))
            if hasattr(baseline_output, 'logits'):
                baseline_logits = baseline_output.logits
            else:
                baseline_logits = baseline_output
            baseline_prob = torch.softmax(baseline_logits, dim=1)[0, target].item()
        
        # For text, we'll use a sampling approach
        np.random.seed(self.random_seed)
        
        # Generate random masks
        masks = np.random.binomial(1, 0.5, size=(self.n_samples, seq_len))
        
        # Always include empty and full masks
        masks[0, :] = 0  # All masked
        masks[1, :] = 1  # None masked
        # Batched evaluation for efficiency
        # Prepare all masked inputs in a batch
        masked_inputs = []
        weights = []

        for i in range(self.n_samples):
            mask = masks[i]
            masked_ids = input_ids.clone()
            mask_indices = torch.tensor(mask == 0, device=device)
            masked_ids[mask_indices] = self.mask_token_id
            masked_inputs.append(masked_ids)
            S = mask.sum()
            weight = self._shapley_kernel_weight(seq_len, S)
            weights.append(weight)

        # Stack all masked inputs into a batch
        masked_inputs_tensor = torch.stack(masked_inputs, dim=0)  # [n_samples, seq_len]

        # Batched model evaluation
        predictions = []
        for batch in torch.split(masked_inputs_tensor, batch_size, dim=0):
            with torch.no_grad():
                output = model(batch)
                logits = getattr(output, 'logits', output)
                probs = torch.softmax(logits, dim=1)[:, target]
                predictions.extend(probs.cpu().numpy())
        predictions = np.array(predictions)
        weights = np.array(weights)
        # Fit weighted linear regression
        X = np.column_stack([np.ones(self.n_samples), masks])
        
        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ predictions
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            coeffs = np.linalg.pinv(XtWX) @ XtWy
        
        # Extract SHAP values (exclude intercept)
        shap_values = coeffs[1:]
        expected_value = coeffs[0]
        
        # Convert to tensor
        attributions = torch.tensor(shap_values, device=device, dtype=torch.float32)
        
        model.train(original_mode)
        
        # For text, each token is its own segment
        segments = torch.arange(len(attributions), device=device)
        
        # Build explanation
        explanation = ShapExplanation(
            attributions=attributions,
            segments=segments,
            expected_value=expected_value,
            base_value=baseline_prob,
            coalitions={'masks': masks, 'predictions': predictions} if return_coalitions else None,
            metadata={
                'target': target,
                'original_probability': original_prob,
                'n_samples': self.n_samples,
                'mask_token_id': self.mask_token_id,
                'num_segments': len(attributions)
            }
        )
        
        return explanation