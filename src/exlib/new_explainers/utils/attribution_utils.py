"""Utility functions for working with attribution explanations and segments."""

import torch
from typing import Literal


def aggregate_attributions_by_segments(
    attributions: torch.Tensor,
    segments: torch.Tensor,
    aggregation: Literal["mean", "sum", "max"] = "mean"
) -> torch.Tensor:
    """Aggregate attributions by segments.
    
    Args:
        attributions: Attribution tensor of shape (C, H, W) or (H, W)
        segments: Segments tensor of same shape with values [0, num_segments-1]
        aggregation: How to aggregate ('mean', 'sum', 'max')
    
    Returns:
        Tensor of shape (num_segments,) with aggregated values
    """
    # Get the segments for the first channel if multi-channel
    if segments.dim() > 2:
        seg_2d = segments[0]  # Use first channel's segmentation
    else:
        seg_2d = segments
    
    n_segments = seg_2d.max().item() + 1
    segment_attrs = torch.zeros(n_segments, device=attributions.device)
    
    for seg_id in range(n_segments):
        mask = seg_2d == seg_id
        if attributions.dim() == 3:  # (C, H, W)
            # Average across all channels for this segment
            values = attributions[:, mask]
        else:  # (H, W)
            values = attributions[mask]
            
        if values.numel() > 0:
            if aggregation == "mean":
                segment_attrs[seg_id] = values.mean()
            elif aggregation == "sum":
                segment_attrs[seg_id] = values.sum()
            elif aggregation == "max":
                segment_attrs[seg_id] = values.abs().max()
    
    return segment_attrs


def create_segmented_attribution_map(
    segments: torch.Tensor,
    segment_attributions: torch.Tensor
) -> torch.Tensor:
    """Create a full attribution map from segment-level attributions.
    
    Args:
        segments: Shape (C, H, W) or (H, W) with segment indices
        segment_attributions: Shape (num_segments,) with attribution per segment
        
    Returns:
        Attribution map of same shape as segments
    """
    # Get 2D segments if multi-channel
    if segments.dim() > 2:
        seg_2d = segments[0]
        output_shape = segments.shape
    else:
        seg_2d = segments
        output_shape = segments.shape
        
    attr_map = torch.zeros(output_shape, device=segments.device)
    
    for seg_id, attr_value in enumerate(segment_attributions):
        mask = seg_2d == seg_id
        if attr_map.dim() == 3:
            attr_map[:, mask] = attr_value
        else:
            attr_map[mask] = attr_value
    
    return attr_map


def get_top_segments(
    attributions: torch.Tensor,
    segments: torch.Tensor,
    k: int = 5,
    aggregation: Literal["mean", "sum", "max"] = "mean"
) -> torch.Tensor:
    """Get the top-k most important segments.
    
    Args:
        attributions: Attribution tensor
        segments: Segments tensor
        k: Number of top segments to return
        aggregation: How to aggregate attributions
        
    Returns:
        Tensor of shape (k,) with segment indices
    """
    segment_attrs = aggregate_attributions_by_segments(
        attributions, segments, aggregation
    )
    
    # Get absolute values for ranking
    segment_importance = segment_attrs.abs()
    
    # Get top-k indices
    top_k = min(k, len(segment_importance))
    _, top_indices = torch.topk(segment_importance, top_k)
    
    return top_indices


def mask_segments(
    input: torch.Tensor,
    segments: torch.Tensor,
    segment_ids: torch.Tensor,
    mask_value: float = 0.0
) -> torch.Tensor:
    """Mask specific segments in the input.
    
    Args:
        input: Input tensor (C, H, W) or (H, W)
        segments: Segments tensor of same shape
        segment_ids: Tensor of segment IDs to mask
        mask_value: Value to use for masking
        
    Returns:
        Masked input tensor
    """
    masked = input.clone()
    
    # Get 2D segments if multi-channel
    if segments.dim() > 2:
        seg_2d = segments[0]
    else:
        seg_2d = segments
    
    # Create mask for all specified segments
    mask = torch.zeros_like(seg_2d, dtype=torch.bool)
    for seg_id in segment_ids:
        mask |= (seg_2d == seg_id)
    
    # Apply mask
    if input.dim() == 3:
        masked[:, mask] = mask_value
    else:
        masked[mask] = mask_value
    
    return masked


def compare_segment_attributions(
    explanations: dict,
    segment_id: int,
    aggregation: Literal["mean", "sum", "max"] = "mean"
) -> dict:
    """Compare how different methods attribute to a specific segment.
    
    Args:
        explanations: Dict mapping method names to explanations
        segment_id: The segment ID to compare
        aggregation: How to aggregate attributions
        
    Returns:
        Dict mapping method names to segment attribution values
    """
    results = {}
    
    for name, exp in explanations.items():
        seg_attrs = aggregate_attributions_by_segments(
            exp.attributions, 
            exp.segments,
            aggregation
        )
        
        if segment_id < len(seg_attrs):
            results[name] = seg_attrs[segment_id].item()
        else:
            results[name] = 0.0
    
    return results