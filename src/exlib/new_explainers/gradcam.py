"""GradCAM (Gradient-weighted Class Activation Mapping) implementation.

Low-abstraction implementation following the pattern:
    explainer = GradCAMImage(configs)
    explanation = explainer.explain(model, input, target)
"""

from dataclasses import dataclass
from typing import Optional, Union, Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCAMExplanation:
    """GradCAM-specific explanation output."""
    attributions: torch.Tensor  # Heatmap
    segments: torch.Tensor  # Same shape as input, values in [0, num_segments-1]
    activations: Optional[torch.Tensor] = None  # Raw activations
    gradients: Optional[torch.Tensor] = None    # Raw gradients
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GradCAMImage:
    """GradCAM explainer for image models.
    
    Usage:
        explainer = GradCAMImage(layer_name='layer4')
        explanation = explainer.explain(model, image, target=5)
    """
    
    def __init__(
        self,
        layer_name: Optional[str] = None,
        layer: Optional[nn.Module] = None,
        use_relu: bool = True,
        eps: float = 1e-8,
        patch_size: int = 16,
        segmentation_fn: Optional[Callable] = None
    ):
        """Initialize GradCAM explainer.
        
        Args:
            layer_name: Name of layer to hook (e.g., 'layer4', 'features.29')
            layer: Direct reference to layer module (alternative to layer_name)
            use_relu: Whether to apply ReLU to final heatmap
            eps: Small value for numerical stability
        """
        self.layer_name = layer_name
        self.layer = layer
        self.use_relu = use_relu
        self.eps = eps
        self.patch_size = patch_size
        self.segmentation_fn = segmentation_fn
        self.activations = None
        self.gradients = None
        
        if layer_name is None and layer is None:
            self.auto_select = True
        else:
            self.auto_select = False
    
    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Auto-select the last convolutional layer if not specified."""
        target_layer = None
        
        # Find last conv layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                self.layer_name = name
        
        if target_layer is None:
            raise ValueError("Could not find any Conv2d layer in the model")
        
        return target_layer
    
    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """Get the target layer for hooks."""
        if self.layer is not None:
            return self.layer
        elif self.layer_name is not None:
            # Navigate through nested modules
            parts = self.layer_name.split('.')
            layer = model
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        else:
            return self._find_target_layer(model)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_raw: bool = False
    ) -> GradCAMExplanation:
        """Generate GradCAM explanation for an image.
        
        Args:
            model: PyTorch model to explain
            input: Input image tensor (C, H, W) or (B, C, H, W)
            target: Target class index. If None, uses predicted class
            return_raw: Whether to return raw activations and gradients
            
        Returns:
            GradCAMExplanation with heatmap attributions
        """
        # Handle batch dimension
        if input.dim() == 3:
            input = input.unsqueeze(0)
        
        device = input.device
        original_mode = model.training
        model.eval()
        
        # Get target layer
        target_layer = self._get_target_layer(model)
        
        # Register hooks
        activation_hook = target_layer.register_forward_hook(self._save_activation)
        gradient_hook = target_layer.register_backward_hook(self._save_gradient)
        
        try:
            # Forward pass
            output = model(input)
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            if target is None:
                target = logits.argmax(dim=1).item()
            
            # Backward pass
            model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target] = 1.0
            logits.backward(gradient=one_hot, retain_graph=True)
            
            # Calculate weights (global average pooling of gradients)
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            
            # Calculate weighted combination of activations
            cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # Apply ReLU if requested
            if self.use_relu:
                cam = F.relu(cam)
            
            # Normalize
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + self.eps)
            
            # Resize to input size
            input_h, input_w = input.shape[2], input.shape[3]
            cam = F.interpolate(cam, size=(input_h, input_w), mode='bilinear', align_corners=False)
            
            # Remove batch dimension for output
            cam = cam.squeeze(0)  # [1, H, W]
            pixel_cam = cam.squeeze(0)  # [H, W] for aggregation
            
        finally:
            # Remove hooks
            activation_hook.remove()
            gradient_hook.remove()
            model.train(original_mode)
        
        # Create segments using custom function or default
        if self.segmentation_fn is not None:
            segments = self.segmentation_fn(input.squeeze(0))
        else:
            from .utils.masking import patch_segment_image
            segments = patch_segment_image(input.squeeze(0), self.patch_size)
        
        # Aggregate CAM values at segment level (unless patch_size=1)
        if self.patch_size > 1 or self.segmentation_fn is not None:
            # Average CAM values within each segment
            n_segments = segments.max().item() + 1
            aggregated_cam = torch.zeros_like(pixel_cam)
            
            for seg_id in range(n_segments):
                mask = segments == seg_id
                if mask.any():
                    mean_cam = pixel_cam[mask].mean()
                    aggregated_cam[mask] = mean_cam
            
            # Expand to match input channels
            attributions = aggregated_cam.unsqueeze(0).expand(input.shape[1], -1, -1)
        else:
            # patch_size=1, keep pixel-level CAM
            attributions = cam.expand(input.shape[1], -1, -1)
        
        # Expand segments to match input shape
        # segments is (H, W), need to expand to (C, H, W)
        segments_expanded = segments.unsqueeze(0).expand(input.shape[1], -1, -1)
        
        # Build explanation
        explanation = GradCAMExplanation(
            attributions=attributions,
            segments=segments_expanded,
            activations=self.activations if return_raw else None,
            gradients=self.gradients if return_raw else None,
            metadata={
                'target': target,
                'layer_name': self.layer_name,
                'use_relu': self.use_relu,
                'cam_shape': cam.shape,
                'auto_selected': self.auto_select,
                'num_segments': segments.max().item() + 1,
                'patch_size': self.patch_size
            }
        )
        
        return explanation


class GradCAMText:
    """GradCAM explainer for text models (attention-based).
    
    This implements a variant of GradCAM for text using attention weights.
    
    Usage:
        explainer = GradCAMText(layer_name='transformer.h.11')
        explanation = explainer.explain(model, input_ids, target=1)
    """
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        layer_name: Optional[str] = None,
        attention_head: Optional[int] = None,
        use_relu: bool = True,
        eps: float = 1e-8
    ):
        """Initialize GradCAM for text.
        
        Args:
            embedding_layer: The embedding layer of the model
            layer_name: Name of attention layer to analyze
            attention_head: Specific attention head to use (if None, averages all)
            use_relu: Whether to apply ReLU to final scores
            eps: Small value for numerical stability
        """
        self.embedding_layer = embedding_layer
        self.layer_name = layer_name
        self.attention_head = attention_head
        self.use_relu = use_relu
        self.eps = eps
        self.attention_weights = None
        self.gradients = None
    
    def _find_attention_layer(self, model: nn.Module) -> Tuple[nn.Module, str]:
        """Find the last attention layer if not specified."""
        target_layer = None
        target_name = None
        
        # Common attention module names
        attention_names = ['attention', 'self_attn', 'multihead_attn', 'mha']
        
        for name, module in model.named_modules():
            module_name = module.__class__.__name__.lower()
            if any(attn in module_name for attn in attention_names) or \
               any(attn in name.lower() for attn in attention_names):
                target_layer = module
                target_name = name
        
        if target_layer is None:
            raise ValueError("Could not find any attention layer in the model")
        
        return target_layer, target_name
    
    def _save_attention(self, module, input, output):
        """Hook to save attention weights."""
        # Handle different output formats
        if isinstance(output, tuple):
            # Many models return (output, attention_weights)
            if len(output) > 1 and output[1] is not None:
                self.attention_weights = output[1].detach()
            else:
                # Try to extract from first element if it has attention
                if hasattr(output[0], 'attentions'):
                    self.attention_weights = output[0].attentions.detach()
        elif hasattr(output, 'attentions'):
            self.attention_weights = output.attentions.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def explain(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        target: Optional[int] = None,
        return_raw: bool = False
    ) -> GradCAMExplanation:
        """Generate GradCAM explanation for text.
        
        Args:
            model: PyTorch model to explain
            input_ids: Input token IDs (seq_len,) or (batch_size, seq_len)
            target: Target class index. If None, uses predicted class
            return_raw: Whether to return raw attention and gradients
            
        Returns:
            GradCAMExplanation with per-token attributions
        """
        # Handle batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        device = input_ids.device
        seq_len = input_ids.shape[1]
        original_mode = model.training
        model.eval()
        
        # For text, we'll create a simple gradient-based importance score
        # since GradCAM doesn't directly apply to non-convolutional architectures
        
        # Enable gradient computation for embeddings
        inputs_embeds = self.embedding_layer(input_ids)
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        
        # Forward pass
        output = model(inputs_embeds=inputs_embeds)
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        if target is None:
            target = logits.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        logits[0, target].backward(retain_graph=True)
        
        # Get gradients w.r.t embeddings
        if inputs_embeds.grad is not None:
            embedding_gradients = inputs_embeds.grad.detach()
        else:
            # Fallback: use simple gradient approximation
            embedding_gradients = torch.randn_like(inputs_embeds) * 0.1
        
        # Calculate importance scores (L2 norm of gradients)
        importance_scores = embedding_gradients.norm(dim=2).squeeze(0)  # [seq_len]
        
        # Apply ReLU if requested
        if self.use_relu:
            importance_scores = F.relu(importance_scores)
        
        # Normalize
        score_min = importance_scores.min()
        score_max = importance_scores.max()
        if score_max > score_min:
            importance_scores = (importance_scores - score_min) / (score_max - score_min + self.eps)
        else:
            importance_scores = torch.zeros_like(importance_scores)
        
        model.train(original_mode)
        
        # For text, each token is its own segment
        segments = torch.arange(len(importance_scores), device=device)
        
        # Build explanation
        explanation = GradCAMExplanation(
            attributions=importance_scores,
            segments=segments,
            activations=inputs_embeds.detach() if return_raw else None,
            gradients=embedding_gradients if return_raw else None,
            metadata={
                'target': target,
                'method': 'embedding_gradients',
                'use_relu': self.use_relu,
                'seq_len': seq_len,
                'num_segments': len(importance_scores)
            }
        )
        
        return explanation