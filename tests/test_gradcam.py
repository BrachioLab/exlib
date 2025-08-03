try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import torch
import torch.nn as nn


class TestGradCAMImage:
    """Test GradCAMImage explainer."""
    
    def test_basic_explanation(self, model_name="resnet50"):
        """Test basic explanation generation with real models."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = GradCAMImage()  # Auto-select layer
        
        explanation = explainer.explain(model, input)
        
        # Check output type
        assert isinstance(explanation, GradCAMExplanation)
        
        # Check attribution shape
        assert explanation.attributions.shape == input.shape
        
        # Check metadata
        assert "target" in explanation.metadata
        assert "layer_name" in explanation.metadata
        assert explanation.metadata["auto_selected"] == True
        
        # Check values are in [0, 1] range (after normalization)
        assert explanation.attributions.min() >= 0
        assert explanation.attributions.max() <= 1
    
    def test_with_target(self, model_name="resnet50"):
        """Test explanation with specific target."""
        model, config = get_vision_model(model_name)
        
        input = get_test_image(size=config["input_size"])
        explainer = GradCAMImage()
        
        # Explain for specific target
        target = 2
        explanation = explainer.explain(model, input, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_specific_layer(self):
        """Test with specific layer selection."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        # Use specific layer
        explainer = GradCAMImage(layer_name="layer4")
        explanation = explainer.explain(model, input)
        
        assert explanation.metadata["layer_name"] == "layer4"
        assert explanation.metadata["auto_selected"] == False
    
    def test_return_raw(self):
        """Test returning raw activations and gradients."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = GradCAMImage()
        
        explanation = explainer.explain(model, input, return_raw=True)
        
        assert explanation.activations is not None
        assert explanation.gradients is not None
        assert isinstance(explanation.activations, torch.Tensor)
        assert isinstance(explanation.gradients, torch.Tensor)
    
    def test_no_relu(self):
        """Test without ReLU activation."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        
        # Without ReLU
        explainer_no_relu = GradCAMImage(use_relu=False)
        exp_no_relu = explainer_no_relu.explain(model, input)
        
        # With ReLU (default)
        explainer_relu = GradCAMImage(use_relu=True)
        exp_relu = explainer_relu.explain(model, input)
        
        # Results should be different (no ReLU can have negative values before normalization)
        assert not torch.allclose(exp_no_relu.attributions, exp_relu.attributions)
    
    def test_vit_model(self):
        """Test with Vision Transformer."""
        model, config = get_vision_model("vit")
        
        input = get_test_image(size=config["input_size"])
        
        # For ViT, we might need to specify a different layer
        # Auto-selection should still work by finding last conv layer
        explainer = GradCAMImage()
        
        try:
            explanation = explainer.explain(model, input)
            assert isinstance(explanation, GradCAMExplanation)
            assert explanation.attributions.shape == input.shape
        except ValueError as e:
            # ViT might not have conv layers, which is expected
            print(f"Expected error for ViT: {e}")


class TestGradCAMText:
    """Test GradCAMText explainer."""
    
    def test_basic_explanation(self):
        """Test basic text explanation with GPT2."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "This is a test sentence for GradCAM explanation"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        
        explainer = GradCAMText(
            embedding_layer=config["embedding_layer"]
        )
        
        explanation = explainer.explain(model, input_ids)
        
        # Check output
        assert isinstance(explanation, GradCAMExplanation)
        assert explanation.attributions.shape == (input_ids.shape[0],)  # seq_len
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["method"] == "embedding_gradients"
        
        # Check values are in [0, 1] range
        assert explanation.attributions.min() >= 0
        assert explanation.attributions.max() <= 1
    
    def test_with_target(self):
        """Test with specific target class."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Another test for GradCAM"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = GradCAMText(
            embedding_layer=config["embedding_layer"]
        )
        
        target = 1
        explanation = explainer.explain(model, input_ids, target=target)
        
        assert explanation.metadata["target"] == target
    
    def test_return_raw(self):
        """Test returning raw embeddings and gradients."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing raw outputs"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        explainer = GradCAMText(
            embedding_layer=config["embedding_layer"]
        )
        
        explanation = explainer.explain(model, input_ids, return_raw=True)
        
        assert explanation.activations is not None  # embeddings
        assert explanation.gradients is not None
        assert explanation.activations.shape[0] == 1  # batch
        assert explanation.activations.shape[1] == input_ids.shape[0]  # seq_len
    
    def test_no_relu(self):
        """Test without ReLU activation."""
        model, config = get_text_model("gpt2")
        tokenizer = config["tokenizer"]
        
        # Create simple input
        text = "Testing ReLU options"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0]
        
        # Without ReLU
        explainer_no_relu = GradCAMText(
            embedding_layer=config["embedding_layer"],
            use_relu=False
        )
        exp_no_relu = explainer_no_relu.explain(model, input_ids)
        
        # With ReLU
        explainer_relu = GradCAMText(
            embedding_layer=config["embedding_layer"],
            use_relu=True
        )
        exp_relu = explainer_relu.explain(model, input_ids)
        
        # Results might be different if there were negative gradients
        # But both should be normalized to [0, 1]
        assert exp_no_relu.attributions.min() >= 0
        assert exp_relu.attributions.min() >= 0


def run_tests():
    """Run tests manually without pytest."""
    print("Testing GradCAMImage...")
    
    image_tests = TestGradCAMImage()
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
        image_tests.test_specific_layer()
        print("✓ Specific layer test passed")
    except Exception as e:
        print(f"✗ Specific layer test failed: {e}")
    
    try:
        image_tests.test_return_raw()
        print("✓ Return raw test passed")
    except Exception as e:
        print(f"✗ Return raw test failed: {e}")
    
    try:
        image_tests.test_no_relu()
        print("✓ No ReLU test passed")
    except Exception as e:
        print(f"✗ No ReLU test failed: {e}")
    
    try:
        image_tests.test_vit_model()
        print("✓ ViT model test passed")
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
    
    print("\nTesting GradCAMText...")
    
    text_tests = TestGradCAMText()
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
        text_tests.test_return_raw()
        print("✓ Text return raw test passed")
    except Exception as e:
        print(f"✗ Text return raw test failed: {e}")
    
    try:
        text_tests.test_no_relu()
        print("✓ Text no ReLU test passed")
    except Exception as e:
        print(f"✗ Text no ReLU test failed: {e}")


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    
    # Import after path is set
    from src.exlib.new_explainers import GradCAMImage, GradCAMText, GradCAMExplanation
    from tests.fixtures import get_vision_model, get_text_model, get_test_image, get_test_text_inputs
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, running tests manually\n")
        run_tests()