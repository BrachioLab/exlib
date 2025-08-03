import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class IntGradExplanation:
    """Integrated Gradients explanation results."""
    attributions: torch.Tensor
    metadata: dict = None
    convergence_delta: Optional[float] = None

class IntGradImage:
    """Integrated Gradients explainer for images."""
    
    def __init__(
        self, 
        n_steps: int = 50,
        baseline: Union[str, torch.Tensor, Callable] = "zero"
    ):
        """
        Args:
            n_steps: Number of integration steps
            baseline: Baseline for integration. Can be:
                - "zero": Black image
                - "mean": Dataset mean (will use 0.5 for each channel)
                - torch.Tensor: Custom baseline image
                - Callable: Function that takes input and returns baseline
        """
        self.n_steps = n_steps
        self.baseline = baseline
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_convergence_delta: bool = False
    ) -> IntGradExplanation:
        """Generate integrated gradients explanation.
        
        Args:
            model: PyTorch model to explain
            input: Input tensor of shape (C, H, W)
            target: Target class (if None, uses predicted class)
            return_convergence_delta: Whether to compute convergence check
            
        Returns:
            IntGradExplanation with attributions
        """
        # Ensure input requires grad
        input = input.detach().requires_grad_(True)
        
        # Add batch dimension if needed
        if input.dim() == 3:
            input_batch = input.unsqueeze(0)
        else:
            input_batch = input
            
        # Get target class if not provided
        if target is None:
            with torch.no_grad():
                output = model(input_batch)
                # Handle both tensor and dataclass outputs
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                target = logits.argmax(dim=1).item()
        
        # Create baseline
        baseline = self._get_baseline(input)
        
        # Compute integrated gradients
        attributions = self._integrate_gradients(
            model, input, baseline, target
        )
        
        # Compute convergence delta if requested
        convergence_delta = None
        if return_convergence_delta:
            convergence_delta = self._compute_convergence_delta(
                model, input, baseline, attributions, target
            )
        
        return IntGradExplanation(
            attributions=attributions,
            metadata={
                "target": target,
                "n_steps": self.n_steps,
                "baseline_type": self.baseline if isinstance(self.baseline, str) else "custom"
            },
            convergence_delta=convergence_delta
        )
    
    def _get_baseline(self, input: torch.Tensor) -> torch.Tensor:
        """Get baseline for given input."""
        if isinstance(self.baseline, str):
            if self.baseline == "zero":
                return torch.zeros_like(input)
            elif self.baseline == "mean":
                return torch.ones_like(input) * 0.5
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline}")
        elif isinstance(self.baseline, torch.Tensor):
            return self.baseline
        elif callable(self.baseline):
            return self.baseline(input)
        else:
            raise ValueError(f"Invalid baseline type: {type(self.baseline)}")
    
    def _integrate_gradients(
        self,
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        target: int
    ) -> torch.Tensor:
        """Compute integrated gradients."""
        # Initialize accumulator
        integrated_grads = torch.zeros_like(input)
        
        # Compute gradients for each step
        for step in range(self.n_steps):
            # Interpolate between baseline and input
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (input - baseline)
            interpolated = interpolated.detach().requires_grad_(True)
            
            # Forward pass
            if interpolated.dim() == 3:
                output = model(interpolated.unsqueeze(0))
            else:
                output = model(interpolated)
            
            # Handle both tensor and dataclass outputs
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Get gradient for target class
            model.zero_grad()
            logits[0, target].backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad / self.n_steps
        
        # Multiply by input - baseline
        integrated_grads *= (input - baseline)
        
        return integrated_grads
    
    def _compute_convergence_delta(
        self,
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        attributions: torch.Tensor,
        target: int
    ) -> float:
        """Compute convergence delta for sanity check."""
        # Get model outputs
        with torch.no_grad():
            if input.dim() == 3:
                out_input = model(input.unsqueeze(0))
                out_baseline = model(baseline.unsqueeze(0))
            else:
                out_input = model(input)
                out_baseline = model(baseline)
            
            # Handle dataclass outputs
            if hasattr(out_input, 'logits'):
                input_output = out_input.logits[0, target]
                baseline_output = out_baseline.logits[0, target]
            else:
                input_output = out_input[0, target]
                baseline_output = out_baseline[0, target]
        
        # Compute sum of attributions
        attr_sum = attributions.sum()
        
        # Compute expected difference
        expected_diff = input_output - baseline_output
        
        # Return absolute difference
        return abs(attr_sum - expected_diff).item()


class IntGradText:
    """Integrated Gradients explainer for text."""
    
    def __init__(
        self,
        embedding_layer: nn.Module,
        n_steps: int = 50,
        baseline: Union[str, int] = "pad"
    ):
        """
        Args:
            embedding_layer: The embedding layer of the model
            n_steps: Number of integration steps
            baseline: Baseline token ID or "pad" for padding token (usually 0)
        """
        self.embedding_layer = embedding_layer
        self.n_steps = n_steps
        self.baseline = baseline
    
    def explain(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: Optional[int] = None,
        return_convergence_delta: bool = False
    ) -> IntGradExplanation:
        """Generate integrated gradients explanation for text.
        
        Args:
            model: PyTorch model to explain
            input: Input tensor of token IDs, shape (seq_len,)
            target: Target class (if None, uses predicted class)
            return_convergence_delta: Whether to compute convergence check
            
        Returns:
            IntGradExplanation with attributions
        """
        # Ensure we have token IDs
        if input.dtype not in [torch.long, torch.int]:
            raise ValueError("Input must be token IDs (long or int tensor)")
        
        # Add batch dimension if needed
        if input.dim() == 1:
            input_batch = input.unsqueeze(0)
        else:
            input_batch = input
        
        # Get embeddings
        input_embeds = self.embedding_layer(input_batch)
        
        # Get target class if not provided
        if target is None:
            with torch.no_grad():
                output = model(input_batch)
                # Handle both tensor and dataclass outputs
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                target = logits.argmax(dim=1).item()
        
        # Create baseline embeddings
        baseline_embeds = self._get_baseline_embeds(input_batch)
        
        # Compute integrated gradients on embeddings
        embeds_grads = self._integrate_gradients_embeds(
            model, input_batch, input_embeds, baseline_embeds, target
        )
        
        # Average across embedding dimension to get token-level attributions
        attributions = embeds_grads.norm(dim=-1).squeeze(0)
        
        # Compute convergence delta if requested
        convergence_delta = None
        if return_convergence_delta:
            convergence_delta = self._compute_convergence_delta(
                model, input_batch, baseline_embeds, embeds_grads, target
            )
        
        return IntGradExplanation(
            attributions=attributions,
            metadata={
                "target": target,
                "n_steps": self.n_steps,
                "baseline": self.baseline
            },
            convergence_delta=convergence_delta
        )
    
    def _get_baseline_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get baseline embeddings."""
        if self.baseline == "pad":
            baseline_ids = torch.zeros_like(input_ids)
        elif isinstance(self.baseline, int):
            baseline_ids = torch.full_like(input_ids, self.baseline)
        else:
            raise ValueError(f"Invalid baseline: {self.baseline}")
        
        return self.embedding_layer(baseline_ids)
    
    def _integrate_gradients_embeds(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
        target: int
    ) -> torch.Tensor:
        """Compute integrated gradients on embeddings."""
        integrated_grads = torch.zeros_like(input_embeds)
        
        for step in range(self.n_steps):
            # Interpolate embeddings
            alpha = step / self.n_steps
            interpolated_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interpolated_embeds = interpolated_embeds.detach().requires_grad_(True)
            
            # Forward pass with interpolated embeddings
            output = model.forward(inputs_embeds=interpolated_embeds)
            
            # Handle dataclass outputs
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Get gradient
            model.zero_grad()
            logits[0, target].backward()
            
            # Accumulate
            integrated_grads += interpolated_embeds.grad / self.n_steps
        
        # Multiply by difference
        integrated_grads *= (input_embeds - baseline_embeds)
        
        return integrated_grads
    
    def _compute_convergence_delta(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        baseline_embeds: torch.Tensor,
        embeds_grads: torch.Tensor,
        target: int
    ) -> float:
        """Compute convergence delta."""
        with torch.no_grad():
            out_input = model(input_ids)
            out_baseline = model.forward(inputs_embeds=baseline_embeds)
            
            # Handle dataclass outputs
            if hasattr(out_input, 'logits'):
                input_output = out_input.logits[0, target]
                baseline_output = out_baseline.logits[0, target]
            else:
                input_output = out_input[0, target]
                baseline_output = out_baseline[0, target]
        
        attr_sum = embeds_grads.sum()
        expected_diff = input_output - baseline_output
        
        return abs(attr_sum - expected_diff).item()