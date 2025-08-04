try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import torch
import torch.nn as nn
import numpy as np
from exlib.new_explainers import LimeImage, LimeText, LimeExplanation
from fixtures.models import get_vision_model, get_text_model
from fixtures.data import get_test_image, get_test_text_inputs


class TestLimeImage:
    """Test LimeImage explainer."""
    
    def test_basic_explanation(self, model_name="resnet50"):
        """Test basic explanation generation with real models."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=100, patch_size=16)
        
        explanation = explainer.explain(model, input)
        
        # Check output type
        assert isinstance(explanation, LimeExplanation)
        
        # Check attribution shape
        assert explanation.attributions.shape == input.shape
        
        # Check segments field
        assert hasattr(explanation, 'segments')
        assert explanation.segments is not None
        assert explanation.segments.shape == input.shape  # Now matches input shape
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_samples"] == 100
        assert "num_segments" in explanation.metadata
        
        # Check R² score
        assert explanation.r2_score is not None
        assert 0 <= explanation.r2_score <= 1
    
    def test_with_target(self, model_name="resnet50"):
        """Test explanation with specific target."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=100, patch_size=16)
        
        # Explain for specific target
        target = 2
        explanation = explainer.explain(model, input, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_segments_properties(self):
        """Test segments tensor properties."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=50, patch_size=16)
        
        explanation = explainer.explain(model, input)
        
        # Check segments field
        assert explanation.segments is not None
        # Segments should now match input shape (C, H, W)
        assert explanation.segments.shape == input.shape
        assert explanation.segments.dtype == torch.long
        
        # Check all channels have same segmentation
        for c in range(input.shape[0]):
            assert torch.allclose(explanation.segments[0], explanation.segments[c])
    
    def test_return_local_model(self):
        """Test returning fitted Ridge model."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=50, patch_size=16)
        
        explanation = explainer.explain(model, input, return_local_model=True)
        
        assert explanation.local_model is not None
        assert hasattr(explanation.local_model, 'coef_')
        assert hasattr(explanation.local_model, 'intercept_')
    
    def test_different_seeds(self):
        """Test that different random seeds produce different results."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        # Create two explainers with different seeds
        explainer1 = LimeImage(n_samples=50, patch_size=16, random_seed=42)
        explainer2 = LimeImage(n_samples=50, patch_size=16, random_seed=123)
        
        exp1 = explainer1.explain(model, input)
        exp2 = explainer2.explain(model, input)
        
        # Attributions should be different
        assert not torch.allclose(exp1.attributions, exp2.attributions)
    
    def test_vit_model(self):
        """Test with Vision Transformer."""
        model, config = get_vision_model("vit")
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=50, patch_size=16)
        
        explanation = explainer.explain(model, input)
        
        assert isinstance(explanation, LimeExplanation)
        assert explanation.attributions.shape == input.shape
    
    def test_custom_segmentation(self):
        """Test with custom segmentation function."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        def custom_segmentation(image):
            # Simple 2x2 grid segmentation
            h, w = image.shape[1], image.shape[2]
            segments = torch.zeros(h, w, dtype=torch.long)
            segments[:h//2, :w//2] = 0
            segments[:h//2, w//2:] = 1
            segments[h//2:, :w//2] = 2
            segments[h//2:, w//2:] = 3
            return segments
        
        explainer = LimeImage(n_samples=50, segmentation_fn=custom_segmentation)
        explanation = explainer.explain(model, input)
        
        # Verify custom segmentation was used
        assert explanation.segments[0].unique().numel() == 4
        assert explanation.metadata['num_segments'] == 4
    
    def test_quickshift_segmentation(self):
        """Test with quickshift segmentation from skimage."""
        try:
            from skimage.segmentation import quickshift
            import numpy as np
        except ImportError:
            # Skip test if skimage not available
            return
        
        model, config = get_vision_model("resnet50")
        input = get_test_image(size=(3, 224, 224))  # Ensure standard size
        
        def quickshift_segmentation(image):
            # Convert to numpy (H, W, C) format for skimage
            img_np = image.permute(1, 2, 0).cpu().numpy()
            # Normalize to [0, 1] range for quickshift
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            # Run quickshift
            segments = quickshift(img_np, kernel_size=3, max_dist=200, ratio=0.2)
            return torch.tensor(segments, device=image.device, dtype=torch.long)
        
        explainer = LimeImage(n_samples=50, segmentation_fn=quickshift_segmentation)
        explanation = explainer.explain(model, input)
        
        # Check that quickshift created segments
        assert hasattr(explanation, 'segments')
        assert explanation.segments.shape == input.shape
        # Quickshift usually creates many segments
        assert explanation.metadata['num_segments'] > 10


class TestLimeText:
    """Test LimeText explainer."""
    
    def test_basic_explanation(self):
        """Test basic text explanation with GPT2."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "This is a test sentence for LIME explanation"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        
        explainer = LimeText(
            embedding_layer=config["embedding_layer"],
            n_samples=100,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        explanation = explainer.explain(model, input_ids)
        
        # Check output
        assert isinstance(explanation, LimeExplanation)
        assert explanation.attributions.shape == (input_ids.shape[0],)  # seq_len
        
        # Check segments field for text
        assert hasattr(explanation, 'segments')
        assert explanation.segments is not None
        assert explanation.segments.shape == input_ids.shape
        # Each token should be its own segment
        expected_segments = torch.arange(len(input_ids))
        assert torch.allclose(explanation.segments, expected_segments)
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_samples"] == 100
        assert explanation.metadata["num_segments"] == len(input_ids)
        
        # Check R² score
        assert explanation.r2_score is not None
        assert 0 <= explanation.r2_score <= 1
    
    def test_with_target(self):
        """Test with specific target class."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Another test for LIME"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = LimeText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        target = 1
        explanation = explainer.explain(model, input_ids, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_return_local_model(self):
        """Test returning fitted Ridge model."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing local model return"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = LimeText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        explanation = explainer.explain(model, input_ids, return_local_model=True)
        
        assert explanation.local_model is not None
        assert hasattr(explanation.local_model, 'coef_')
        assert len(explanation.local_model.coef_) == input_ids.shape[0]
    
    def test_different_mask_tokens(self):
        """Test with different mask token IDs."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing mask tokens"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        # Use padding token as mask
        explainer1 = LimeText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        # Use a different token as mask
        explainer2 = LimeText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=1  # Different mask token
        )
        
        exp1 = explainer1.explain(model, input_ids)
        exp2 = explainer2.explain(model, input_ids)
        
        # Results should be different
        assert not torch.allclose(exp1.attributions, exp2.attributions, atol=1e-3)


def run_tests():
    """Run tests manually without pytest."""
    print("Testing LimeImage...")
    
    image_tests = TestLimeImage()
    try:
        image_tests.test_basic_explanation()
        print("✓ Basic image explanation test passed")
    except Exception as e:
        print(f"✗ Basic image explanation test failed: {e}")
    
    try:
        image_tests.test_with_target()
        print("✓ Image with target test passed")
    except Exception as e:
        print(f"✗ Image with target test failed: {e}")
    
    try:
        image_tests.test_segments_properties()
        print("✓ Segments properties test passed")
    except Exception as e:
        print(f"✗ Segments properties test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        image_tests.test_custom_segmentation()
        print("✓ Custom segmentation test passed")
    except Exception as e:
        print(f"✗ Custom segmentation test failed: {e}")
    
    try:
        image_tests.test_quickshift_segmentation()
        print("✓ Quickshift segmentation test passed")
    except Exception as e:
        print(f"✗ Quickshift segmentation test failed: {e}")
    
    try:
        image_tests.test_return_local_model()
        print("✓ Return local model test passed")
    except Exception as e:
        print(f"✗ Return local model test failed: {e}")
    
    try:
        image_tests.test_different_seeds()
        print("✓ Different seeds test passed")
    except Exception as e:
        print(f"✗ Different seeds test failed: {e}")
    
    try:
        image_tests.test_vit_model()
        print("✓ ViT model test passed")
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
    
    print("\nTesting LimeText...")
    
    text_tests = TestLimeText()
    try:
        text_tests.test_basic_explanation()
        print("✓ Basic text explanation test passed")
    except Exception as e:
        print(f"✗ Basic text explanation test failed: {e}")
    
    try:
        text_tests.test_with_target()
        print("✓ Text with target test passed")
    except Exception as e:
        print(f"✗ Text with target test failed: {e}")
    
    try:
        text_tests.test_return_local_model()
        print("✓ Text return local model test passed")
    except Exception as e:
        print(f"✗ Text return local model test failed: {e}")
    
    try:
        text_tests.test_different_mask_tokens()
        print("✓ Different mask tokens test passed")
    except Exception as e:
        print(f"✗ Different mask tokens test failed: {e}")


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    
    # Import after path is set
    from exlib.new_explainers import LimeImage, LimeText, LimeExplanation
    from fixtures.models import get_vision_model, get_text_model
    from fixtures.data import get_test_image, get_test_text_inputs
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, running tests manually\n")
        run_tests()