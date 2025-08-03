try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import torch
import torch.nn as nn
from src.exlib.new_explainers import IntGradImage, IntGradText, IntGradExplanation
from tests.fixtures import get_vision_model, get_text_model, get_test_image, get_test_text_inputs

class TestIntGradImage:
    """Test IntGradImage explainer."""
    
    def test_basic_explanation(self, model_name="resnet50"):
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
    
    def test_with_target(self, model_name="resnet50"):
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
    
    def test_convergence_delta(self, model_name="resnet50"):
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
    if HAS_PYTEST:
        pytest.main([__file__])
    else:
        print("pytest not available, run tests manually")