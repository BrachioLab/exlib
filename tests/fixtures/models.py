import torch
import torch.nn as nn
from typing import Tuple, Union

def get_vision_model(model_name: str = "resnet50") -> Tuple[nn.Module, dict]:
    """Get a vision model for testing.
    
    Args:
        model_name: Either "resnet50" or "vit"
        
    Returns:
        Tuple of (model, config) where config contains model-specific info
    """
    if model_name == "resnet50":
        from torchvision import models
        model = models.resnet50(pretrained=False)
        model.eval()
        config = {
            "input_size": (224, 224),
            "num_classes": 1000,
            "output_type": "tensor"  # Direct logits
        }
        return model, config
        
    elif model_name == "vit":
        from transformers import ViTForImageClassification, ViTConfig
        # Use a smaller ViT for faster testing
        config_vit = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_labels=1000
        )
        model = ViTForImageClassification(config_vit)
        model.eval()
        config = {
            "input_size": (224, 224),
            "num_classes": 1000,
            "output_type": "dataclass"  # Returns ModelOutput dataclass
        }
        return model, config
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_text_model(model_name: str = "gpt2") -> Tuple[nn.Module, dict]:
    """Get a text model for testing.
    
    Args:
        model_name: Currently only "gpt2" supported
        
    Returns:
        Tuple of (model, config) where config contains model-specific info
    """
    if model_name == "gpt2":
        from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer
        
        # Create a small GPT2 for testing
        config_gpt2 = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            num_labels=2,  # Binary classification
            pad_token_id=50256
        )
        model = GPT2ForSequenceClassification(config_gpt2)
        model.eval()
        
        # Also get tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        config = {
            "tokenizer": tokenizer,
            "max_length": 512,
            "num_classes": 2,
            "embedding_layer": model.transformer.wte,  # Word token embeddings
            "output_type": "dataclass"
        }
        return model, config
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Helper function to extract logits from model outputs
def get_model_logits(output: Union[torch.Tensor, object]) -> torch.Tensor:
    """Extract logits from model output.
    
    Args:
        output: Either a tensor (ResNet) or dataclass (ViT/GPT2)
        
    Returns:
        Logits tensor
    """
    if isinstance(output, torch.Tensor):
        return output
    else:
        # Handle HuggingFace model outputs
        return output.logits