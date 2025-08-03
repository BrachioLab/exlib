try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import torch
import torch.nn as nn
import numpy as np


class TestLimeImage:
    """Test LimeImage explainer."""
    
    def test_basic_explanation(self, model_name="resnet50"):
        """Test basic explanation generation with real models."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=100, n_segments=10)
        
        explanation = explainer.explain(model, input)
        
        # Check output type
        assert isinstance(explanation, LimeExplanation)
        
        # Check attribution shape
        assert explanation.attributions.shape == input.shape
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_samples"] == 100
        assert explanation.metadata["n_segments"] <= 10  # May be less due to grid
        
        # Check R² score
        assert explanation.r2_score is not None
        assert 0 <= explanation.r2_score <= 1
    
    def test_with_target(self, model_name="resnet50"):
        """Test explanation with specific target."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=100, n_segments=10)
        
        # Explain for specific target
        target = 2
        explanation = explainer.explain(model, input, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_return_segments(self):
        """Test returning segmentation map."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=50, n_segments=10)
        
        explanation = explainer.explain(model, input, return_segments=True)
        
        assert explanation.segments is not None
        # Segments should be 2D (H, W)
        assert explanation.segments.dim() == 2
        assert explanation.segments.shape[0] == config["input_size"]
        assert explanation.segments.shape[1] == config["input_size"]
        assert explanation.segments.dtype == torch.long
    
    def test_return_local_model(self):
        """Test returning fitted Ridge model."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=50, n_segments=10)
        
        explanation = explainer.explain(model, input, return_local_model=True)
        
        assert explanation.local_model is not None
        assert hasattr(explanation.local_model, 'coef_')
        assert hasattr(explanation.local_model, 'intercept_')
    
    def test_different_seeds(self):
        """Test that different random seeds produce different results."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        # Create two explainers with different seeds
        explainer1 = LimeImage(n_samples=50, n_segments=10, random_seed=42)
        explainer2 = LimeImage(n_samples=50, n_segments=10, random_seed=123)
        
        exp1 = explainer1.explain(model, input)
        exp2 = explainer2.explain(model, input)
        
        # Attributions should be different
        assert not torch.allclose(exp1.attributions, exp2.attributions)
    
    def test_vit_model(self):
        """Test with Vision Transformer."""
        model, config = get_vision_model("vit")
        
        input = get_test_image(size=config["input_size"])
        explainer = LimeImage(n_samples=50, n_segments=10)
        
        explanation = explainer.explain(model, input)
        
        assert isinstance(explanation, LimeExplanation)
        assert explanation.attributions.shape == input.shape


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
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_samples"] == 100
        
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
        image_tests.test_return_segments()
        print("✓ Return segments test passed")
    except Exception as e:
        print(f"✗ Return segments test failed: {e}")
        import traceback
        traceback.print_exc()
    
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
    from src.exlib.new_explainers import LimeImage, LimeText, LimeExplanation
    from tests.fixtures import get_vision_model, get_text_model, get_test_image, get_test_text_inputs
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, running tests manually\n")
        run_tests()