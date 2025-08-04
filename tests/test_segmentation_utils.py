"""Tests for segmentation and attribution utilities."""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch
import numpy as np


class TestSegmentationUtils:
    """Test segmentation utility functions."""
    
    def test_patch_segment_image(self):
        """Test patch segmentation function."""
        from src.exlib.new_explainers.utils.masking import patch_segment_image
        
        # Create test image
        image = torch.randn(3, 224, 224)
        
        # Test different patch sizes
        for patch_size in [8, 16, 32]:
            segments = patch_segment_image(image, patch_size)
            
            assert segments.shape == (224, 224)
            assert segments.dtype == torch.long
            
            # Expected number of segments
            expected_segments = (224 // patch_size) ** 2
            assert segments.unique().numel() <= expected_segments
            
            # Check that segments are contiguous
            assert segments.min() == 0
            assert segments.max() == segments.unique().numel() - 1
    
    def test_patch_segment_non_divisible(self):
        """Test patch segmentation with non-divisible dimensions."""
        from src.exlib.new_explainers.utils.masking import patch_segment_image
        
        # Create image with dimensions not divisible by patch size
        image = torch.randn(3, 225, 225)  # 225 is not divisible by 16
        
        segments = patch_segment_image(image, patch_size=16)
        
        assert segments.shape == (225, 225)
        assert segments.dtype == torch.long
        
        # Should create extra segments for remainder pixels
        expected_segments = ((225 + 15) // 16) ** 2  # Ceiling division
        assert segments.unique().numel() <= expected_segments


class TestAttributionUtils:
    """Test attribution utility functions."""
    
    def test_aggregate_attributions_by_segments(self):
        """Test attribution aggregation by segments."""
        from src.exlib.new_explainers.utils.attribution_utils import aggregate_attributions_by_segments
        
        # Create simple attribution and segmentation
        attributions = torch.tensor([
            [1.0, 2.0, -1.0, -2.0],
            [3.0, 4.0, -3.0, -4.0],
            [5.0, 6.0, -5.0, -6.0],
            [7.0, 8.0, -7.0, -8.0]
        ])
        
        segments = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        
        # Test mean aggregation
        seg_attrs_mean = aggregate_attributions_by_segments(
            attributions, segments, aggregation="mean"
        )
        
        assert seg_attrs_mean.shape == (4,)  # 4 segments
        assert torch.allclose(seg_attrs_mean[0], torch.tensor(2.5))  # (1+2+3+4)/4
        assert torch.allclose(seg_attrs_mean[1], torch.tensor(-2.5))  # (-1-2-3-4)/4
        
        # Test sum aggregation
        seg_attrs_sum = aggregate_attributions_by_segments(
            attributions, segments, aggregation="sum"
        )
        
        assert torch.allclose(seg_attrs_sum[0], torch.tensor(10.0))  # 1+2+3+4
        assert torch.allclose(seg_attrs_sum[1], torch.tensor(-10.0))  # -1-2-3-4
        
        # Test max aggregation
        seg_attrs_max = aggregate_attributions_by_segments(
            attributions, segments, aggregation="max"
        )
        
        assert torch.allclose(seg_attrs_max[0], torch.tensor(4.0))  # max(abs(1,2,3,4))
        assert torch.allclose(seg_attrs_max[1], torch.tensor(4.0))  # max(abs(-1,-2,-3,-4))
    
    def test_aggregate_attributions_multichannel(self):
        """Test aggregation with multi-channel inputs."""
        from src.exlib.new_explainers.utils.attribution_utils import aggregate_attributions_by_segments
        
        # Create 3-channel attribution
        attributions = torch.randn(3, 4, 4)
        segments = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        # Expand to match attribution shape
        segments_expanded = segments.unsqueeze(0).expand(3, -1, -1)
        
        seg_attrs = aggregate_attributions_by_segments(
            attributions, segments_expanded, aggregation="mean"
        )
        
        assert seg_attrs.shape == (4,)  # 4 segments
    
    def test_create_segmented_attribution_map(self):
        """Test creating attribution map from segment values."""
        from src.exlib.new_explainers.utils.attribution_utils import create_segmented_attribution_map
        
        segments = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        
        segment_attributions = torch.tensor([1.0, -1.0, 2.0, -2.0])
        
        attr_map = create_segmented_attribution_map(segments, segment_attributions)
        
        assert attr_map.shape == segments.shape
        
        # Check that values are correctly assigned
        assert torch.all(attr_map[segments == 0] == 1.0)
        assert torch.all(attr_map[segments == 1] == -1.0)
        assert torch.all(attr_map[segments == 2] == 2.0)
        assert torch.all(attr_map[segments == 3] == -2.0)
    
    def test_get_top_segments(self):
        """Test getting top-k segments."""
        from src.exlib.new_explainers.utils.attribution_utils import get_top_segments
        
        # Create attribution with clear ranking
        attributions = torch.tensor([
            [0.1, 0.1, 5.0, 5.0],
            [0.1, 0.1, 5.0, 5.0],
            [2.0, 2.0, -10.0, -10.0],
            [2.0, 2.0, -10.0, -10.0]
        ])
        
        segments = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        
        top_segments = get_top_segments(attributions, segments, k=2, aggregation="mean")
        
        assert len(top_segments) == 2
        # Segment 3 has highest absolute mean (-10), segment 1 has second (5.0)
        assert 3 in top_segments
        assert 1 in top_segments
    
    def test_mask_segments(self):
        """Test masking specific segments."""
        from src.exlib.new_explainers.utils.attribution_utils import mask_segments
        
        input = torch.ones(3, 4, 4)
        segments = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        segments_expanded = segments.unsqueeze(0).expand(3, -1, -1)
        
        # Mask segments 0 and 2
        segment_ids = torch.tensor([0, 2])
        masked = mask_segments(input, segments_expanded, segment_ids, mask_value=0.0)
        
        assert masked.shape == input.shape
        
        # Check that specified segments are masked
        assert torch.all(masked[:, :2, :2] == 0.0)  # Segment 0
        assert torch.all(masked[:, 2:, :2] == 0.0)  # Segment 2
        
        # Check that other segments are unchanged
        assert torch.all(masked[:, :2, 2:] == 1.0)  # Segment 1
        assert torch.all(masked[:, 2:, 2:] == 1.0)  # Segment 3
    
    def test_compare_segment_attributions(self):
        """Test comparing segment attributions across methods."""
        from src.exlib.new_explainers.utils.attribution_utils import compare_segment_attributions
        from src.exlib.new_explainers import IntGradExplanation, LimeExplanation
        
        # Create mock explanations
        attributions = torch.randn(3, 4, 4)
        segments = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        segments_expanded = segments.unsqueeze(0).expand(3, -1, -1)
        
        explanations = {
            "intgrad": IntGradExplanation(
                attributions=attributions * 1.5,
                segments=segments_expanded,
                metadata={}
            ),
            "lime": LimeExplanation(
                attributions=attributions * 0.8,
                segments=segments_expanded,
                metadata={}
            )
        }
        
        # Compare segment 0
        comparison = compare_segment_attributions(explanations, segment_id=0)
        
        assert "intgrad" in comparison
        assert "lime" in comparison
        assert isinstance(comparison["intgrad"], float)
        assert isinstance(comparison["lime"], float)
        
        # IntGrad should have higher values (multiplied by 1.5)
        assert abs(comparison["intgrad"]) > abs(comparison["lime"])


def run_tests():
    """Run tests manually without pytest."""
    print("Testing Segmentation Utils...")
    
    seg_tests = TestSegmentationUtils()
    try:
        seg_tests.test_patch_segment_image()
        print("✓ Patch segment image test passed")
    except Exception as e:
        print(f"✗ Patch segment image test failed: {e}")
    
    try:
        seg_tests.test_patch_segment_non_divisible()
        print("✓ Patch segment non-divisible test passed")
    except Exception as e:
        print(f"✗ Patch segment non-divisible test failed: {e}")
    
    print("\nTesting Attribution Utils...")
    
    attr_tests = TestAttributionUtils()
    try:
        attr_tests.test_aggregate_attributions_by_segments()
        print("✓ Aggregate attributions test passed")
    except Exception as e:
        print(f"✗ Aggregate attributions test failed: {e}")
    
    try:
        attr_tests.test_aggregate_attributions_multichannel()
        print("✓ Aggregate multichannel test passed")
    except Exception as e:
        print(f"✗ Aggregate multichannel test failed: {e}")
    
    try:
        attr_tests.test_create_segmented_attribution_map()
        print("✓ Create segmented map test passed")
    except Exception as e:
        print(f"✗ Create segmented map test failed: {e}")
    
    try:
        attr_tests.test_get_top_segments()
        print("✓ Get top segments test passed")
    except Exception as e:
        print(f"✗ Get top segments test failed: {e}")
    
    try:
        attr_tests.test_mask_segments()
        print("✓ Mask segments test passed")
    except Exception as e:
        print(f"✗ Mask segments test failed: {e}")
    
    try:
        attr_tests.test_compare_segment_attributions()
        print("✓ Compare segment attributions test passed")
    except Exception as e:
        print(f"✗ Compare segment attributions test failed: {e}")


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, running tests manually\n")
        run_tests()