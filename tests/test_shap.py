try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import torch
import torch.nn as nn
import numpy as np


class TestShapImage:
    """Test ShapImage explainer."""
    
    def test_basic_explanation(self, model_name="resnet50"):
        """Test basic explanation generation with real models."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = ShapImage(n_samples=100, n_segments=10)
        
        explanation = explainer.explain(model, input)
        
        # Check output type
        assert isinstance(explanation, ShapExplanation)
        
        # Check attribution shape
        assert explanation.attributions.shape == input.shape
        
        # Check expected value
        assert isinstance(explanation.expected_value, float)
        assert isinstance(explanation.base_value, float)
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_samples"] == 100
    
    def test_with_target(self, model_name="resnet50"):
        """Test explanation with specific target."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = ShapImage(n_samples=50, n_segments=10)
        
        # Explain for specific target
        target = 2
        explanation = explainer.explain(model, input, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_different_baselines(self):
        """Test different baseline types."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        # Test zero baseline
        explainer_zero = ShapImage(n_samples=50, n_segments=10, baseline="zero")
        exp_zero = explainer_zero.explain(model, input)
        
        # Test mean baseline
        explainer_mean = ShapImage(n_samples=50, n_segments=10, baseline="mean")
        exp_mean = explainer_mean.explain(model, input)
        
        # Attributions and expected values should be different
        assert not torch.allclose(exp_zero.attributions, exp_mean.attributions)
        assert abs(exp_zero.expected_value - exp_mean.expected_value) > 0.01
    
    def test_return_coalitions(self):
        """Test returning coalition information."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = ShapImage(n_samples=50, n_segments=10)
        
        explanation = explainer.explain(model, input, return_coalitions=True)
        
        assert explanation.coalitions is not None
        assert "coalitions" in explanation.coalitions
        assert "predictions" in explanation.coalitions
        assert explanation.coalitions["coalitions"].shape[0] == 50
    
    def test_custom_baseline(self):
        """Test with custom baseline tensor."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        baseline = torch.ones_like(input) * 0.3
        
        explainer = ShapImage(n_samples=50, n_segments=10, baseline=baseline)
        explanation = explainer.explain(model, input)
        
        assert explanation.attributions.shape == input.shape
        assert explanation.metadata["baseline_type"] == "custom"
    
    def test_vit_model(self):
        """Test with Vision Transformer."""
        model, config = get_vision_model("vit")
        
        input = get_test_image(size=config["input_size"])
        explainer = ShapImage(n_samples=50, n_segments=10)
        
        explanation = explainer.explain(model, input)
        
        assert isinstance(explanation, ShapExplanation)
        assert explanation.attributions.shape == input.shape


class TestShapText:
    """Test ShapText explainer."""
    
    def test_basic_explanation(self):
        """Test basic text explanation with GPT2."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "This is a test sentence for SHAP explanation"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        
        explainer = ShapText(
            embedding_layer=config["embedding_layer"],
            n_samples=100,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        explanation = explainer.explain(model, input_ids)
        
        # Check output
        assert isinstance(explanation, ShapExplanation)
        assert explanation.attributions.shape == (input_ids.shape[0],)  # seq_len
        
        # Check expected value
        assert isinstance(explanation.expected_value, float)
        assert isinstance(explanation.base_value, float)
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_samples"] == 100
    
    def test_with_target(self):
        """Test with specific target class."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Another test for SHAP"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = ShapText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        target = 1
        explanation = explainer.explain(model, input_ids, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_return_coalitions(self):
        """Test returning coalition information."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing coalitions"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = ShapText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        explanation = explainer.explain(model, input_ids, return_coalitions=True)
        
        assert explanation.coalitions is not None
        assert "masks" in explanation.coalitions
        assert "predictions" in explanation.coalitions
        assert explanation.coalitions["masks"].shape[0] == 50
        assert explanation.coalitions["masks"].shape[1] == input_ids.shape[0]
    
    def test_different_mask_tokens(self):
        """Test with different mask token IDs."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing mask tokens"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        # Use padding token as mask
        explainer1 = ShapText(
            embedding_layer=config["embedding_layer"],
            n_samples=50,
            mask_token_id=tokenizer.pad_token_id or 0
        )
        
        # Use a different token as mask
        explainer2 = ShapText(
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
    print("Testing ShapImage...")
    
    image_tests = TestShapImage()
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
        image_tests.test_different_baselines()
        print("✓ Different baselines test passed")
    except Exception as e:
        print(f"✗ Different baselines test failed: {e}")
    
    try:
        image_tests.test_return_coalitions()
        print("✓ Return coalitions test passed")
    except Exception as e:
        print(f"✗ Return coalitions test failed: {e}")
    
    try:
        image_tests.test_custom_baseline()
        print("✓ Custom baseline test passed")
    except Exception as e:
        print(f"✗ Custom baseline test failed: {e}")
    
    try:
        image_tests.test_vit_model()
        print("✓ ViT model test passed")
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
    
    print("\nTesting ShapText...")
    
    text_tests = TestShapText()
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
        text_tests.test_return_coalitions()
        print("✓ Text return coalitions test passed")
    except Exception as e:
        print(f"✗ Text return coalitions test failed: {e}")
    
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
    from src.exlib.new_explainers import ShapImage, ShapText, ShapExplanation
    from tests.fixtures import get_vision_model, get_text_model, get_test_image, get_test_text_inputs
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, running tests manually\n")
        run_tests()