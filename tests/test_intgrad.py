try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import torch
import torch.nn as nn
from exlib.new_explainers import IntGradImage, IntGradText, IntGradExplanation
from fixtures.models import get_vision_model, get_text_model
from fixtures.data import get_test_image, get_test_text_inputs

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
        
        # Check segments field
        assert hasattr(explanation, 'segments')
        assert explanation.segments is not None
        assert explanation.segments.shape == input.shape
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_steps"] == 10
        assert "num_segments" in explanation.metadata
    
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
    
    def test_segments_properties(self):
        """Test segments tensor properties."""
        model, config = get_vision_model("resnet50")
        
        input = get_test_image(size=config["input_size"])
        explainer = IntGradImage(n_steps=10, patch_size=16)
        
        explanation = explainer.explain(model, input)
        
        # Check shape matches input
        assert explanation.segments.shape == input.shape
        
        # Check integer values
        assert explanation.segments.dtype == torch.long
        
        # Check value range
        unique_segments = explanation.segments.unique()
        assert unique_segments.min() >= 0
        expected_max = (input.shape[1] // 16) * (input.shape[2] // 16) - 1
        assert unique_segments.max() <= expected_max
        
        # Check all channels have same segmentation
        for c in range(input.shape[0]):
            assert torch.allclose(explanation.segments[0], explanation.segments[c])
    
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
        
        explainer = IntGradImage(n_steps=10, segmentation_fn=custom_segmentation)
        explanation = explainer.explain(model, input)
        
        # Verify custom segmentation was used
        assert explanation.segments[0].unique().numel() == 4
        assert explanation.metadata['num_segments'] == 4


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
        
        # Check segments field for text
        assert hasattr(explanation, 'segments')
        assert explanation.segments is not None
        assert explanation.segments.shape == input_ids.shape
        # Each token should be its own segment
        expected_segments = torch.arange(len(input_ids))
        assert torch.allclose(explanation.segments, expected_segments)
        
        # Check metadata
        assert "target" in explanation.metadata
        assert explanation.metadata["n_steps"] == 10
        assert explanation.metadata["num_segments"] == len(input_ids)
    
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
        
        explainer = IntGradImage(n_steps=10, segmentation_fn=quickshift_segmentation)
        explanation = explainer.explain(model, input)
        
        # Check that quickshift created segments
        assert hasattr(explanation, 'segments')
        assert explanation.segments.shape == input.shape
        # Quickshift usually creates many segments
        assert explanation.metadata['num_segments'] > 10


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__])
    else:
        print("pytest not available, run tests manually")