"""LIME (Local Interpretable Model-agnostic Explanations) implementation.

Low-abstraction implementation following the pattern:
    explainer = LimeImage(configs)
    explanation = explainer.explain(model, input, target)
"""

from dataclasses import dataclass
from typing import Optional, Union, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class LimeExplanation:
    """LIME-specific explanation output."""
    attributions: torch.Tensor
    segments: Optional[torch.Tensor] = None  # For image segmentation
    local_model: Optional[object] = None    # The fitted Ridge model
    r2_score: Optional[float] = None        # Model fit quality
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LimeImage:
    """LIME explainer for image models.

    Usage:
        explainer = LimeImage(n_samples=100, n_segments=10)
        explanation = explainer.explain(model, image, target=5)
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_segments: int = 10,
        kernel_width: float = 25.0,  # Appropriate for L2 distance in pixel space
        random_seed: int = 42,
        segmentation_fn: Optional[Callable] = None
    ):
        self.n_samples = n_samples
        self.n_segments = n_segments
        self.kernel_width = kernel_width
        self.random_seed = random_seed
        self.segmentation_fn = segmentation_fn or self._default_segmentation

    def _default_segmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Simple grid-based segmentation."""
        _, h, w = image.shape
        grid_size = int(np.sqrt(self.n_segments))
        seg_h, seg_w = h // grid_size, w // grid_size
        segment_ids = torch.arange(seg_h * seg_w, dtype=torch.long).view(seg_h, seg_w)
        segments = F.interpolate(
            segment_ids[None, None].float(),
            size=(h, w),
            mode='nearest'
        ).view(h, w).long()
        return segments

    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_segments: bool = False,
        return_local_model: bool = False
    ) -> LimeExplanation:
        """Generate LIME explanation for an image."""
        device = input.device
        original_mode = model.training
        model.eval()

        # Segmentation and feature count
        segments = self.segmentation_fn(input)
        n_features = segments.max().item() + 1

        # Get original prediction and target
        with torch.no_grad():
            output = model(input.unsqueeze(0))
            logits = getattr(output, 'logits', output)
            if target is None:
                target = logits.argmax(dim=1).item()
            base_prob = torch.softmax(logits, dim=1)[0, target].item()

        # Generate binary masks for perturbations
        np.random.seed(self.random_seed)
        masks = np.random.binomial(1, 0.5, size=(self.n_samples, n_features))
        masks[0] = 1  # Ensure original image is included

        # Perturb images and collect predictions
        predictions = []
        perturbed_images = []
        for mask in masks:
            perturbed = input.clone()
            for seg_id in range(n_features):
                if not mask[seg_id]:
                    perturbed[:, segments == seg_id] = 0
            perturbed_images.append(perturbed)
            with torch.no_grad():
                out = model(perturbed.unsqueeze(0))
                logits = getattr(out, 'logits', out)
                prob = torch.softmax(logits, dim=1)[0, target].item()
                predictions.append(prob)
        predictions = np.array(predictions)

        # Compute distances and kernel weights
        original_flat = input.cpu().numpy().flatten()
        distances = np.array([
            np.linalg.norm(original_flat - p.cpu().numpy().flatten())
            for p in perturbed_images
        ])
        weights = np.sqrt(np.exp(-(distances ** 2) / (self.kernel_width ** 2)))

        # Fit weighted linear model
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(masks, predictions, sample_weight=weights)
        importance = ridge.coef_

        # Attribution map
        attribution_map = torch.zeros_like(input[0])
        for seg_id in range(n_features):
            attribution_map[segments == seg_id] = importance[seg_id]
        attributions = attribution_map.unsqueeze(0).expand_as(input)

        # R² score
        r2 = ridge.score(masks, predictions, sample_weight=weights)

        model.train(original_mode)

        return LimeExplanation(
            attributions=attributions,
            segments=segments if return_segments else None,
            local_model=ridge if return_local_model else None,
            r2_score=r2,
            metadata={
                'target': target,
                'base_probability': base_prob,
                'n_samples': self.n_samples,
                'n_segments': n_features,
                'kernel_width': self.kernel_width
            }
        )

class LimeText:
    """LIME explainer for text models.

    Usage:
        explainer = LimeText(n_samples=100)
        explanation = explainer.explain(model, input_ids, target=1)
    """

    def __init__(
        self,
        embedding_layer: nn.Module,
        n_samples: int = 1000,
        mask_token_id: int = 0,  # Usually [MASK] or padding token
        kernel_width: float = 0.25,
        random_seed: int = 42
    ):
        self.embedding_layer = embedding_layer
        self.n_samples = n_samples
        self.mask_token_id = mask_token_id
        self.kernel_width = kernel_width
        self.random_seed = random_seed

    def explain(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        target: Optional[int] = None,
        return_local_model: bool = False
    ) -> LimeExplanation:
        """Generate LIME explanation for text.

        Args:
            model: PyTorch model to explain
            input_ids: Input token IDs (seq_len,) or (batch_size, seq_len)
            target: Target class index. If None, uses predicted class
            return_local_model: Whether to return the fitted Ridge model

        Returns:
            LimeExplanation with per-token attributions
        """
        # Ensure input is 1D (single sequence)
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
            base_prob = torch.softmax(logits, dim=1)[0, target].item()

        # Generate perturbed samples
        np.random.seed(self.random_seed)
        masks = np.random.binomial(1, 0.5, size=(self.n_samples, seq_len))
        masks[0, :] = 1  # Ensure original is included

        # Get predictions for perturbed samples
        predictions = []
        perturbed_inputs = []
        for i in range(self.n_samples):
            mask = masks[i]
            perturbed_ids = input_ids.clone()
            mask_indices = torch.tensor(mask == 0, device=device)
            perturbed_ids[mask_indices] = self.mask_token_id
            perturbed_inputs.append(perturbed_ids)
            with torch.no_grad():
                out = model(perturbed_ids.unsqueeze(0))
                logits = getattr(out, 'logits', out)
                prob = torch.softmax(logits, dim=1)[0, target].item()
                predictions.append(prob)
        predictions = np.array(predictions)

        # Compute distances and kernel weights
        original_mask = np.ones(seq_len)
        # Use cosine distance (1 - cosine similarity)
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(masks, original_mask.reshape(1, -1)).ravel()
        weights = np.sqrt(np.exp(-(distances ** 2) / (self.kernel_width ** 2)))

        # Fit weighted linear model
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(masks, predictions, sample_weight=weights)

        # Get token importance
        attributions = torch.tensor(ridge.coef_, device=device, dtype=torch.float32)

        # Calculate R² score
        r2 = ridge.score(masks, predictions, sample_weight=weights)

        model.train(original_mode)

        # Build explanation
        explanation = LimeExplanation(
            attributions=attributions,
            local_model=ridge if return_local_model else None,
            r2_score=r2,
            metadata={
                'target': target,
                'base_probability': base_prob,
                'n_samples': self.n_samples,
                'kernel_width': self.kernel_width,
                'mask_token_id': self.mask_token_id
            }
        )

        return explanation
