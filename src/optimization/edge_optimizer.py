#!/usr/bin/env python3
"""
Edge Optimization Module for Robot State Diffusion Model

This module provides quantization, pruning, and other optimization techniques
to make the diffusion model suitable for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time
from dataclasses import dataclass
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.diffusion_model import RobotStateDiffusionTransformer, ModelConfig


@dataclass
class OptimizationConfig:
    """Configuration for edge optimization"""
    # Quantization
    use_quantization: bool = True
    quantization_backend: str = "fbgemm"  # "fbgemm" for x86, "qnnpack" for ARM
    quantization_mode: str = "dynamic"  # "dynamic", "static", "qat"
    
    # Pruning
    use_pruning: bool = True
    pruning_amount: float = 0.3  # 30% of parameters to prune
    structured_pruning: bool = False
    
    # Model optimization
    use_torch_jit: bool = True
    optimize_for_inference: bool = True
    
    # Hardware targets
    target_inference_time_ms: float = 50.0  # Target inference time
    target_model_size_mb: float = 10.0  # Target model size


class EdgeOptimizer:
    """Edge optimization for diffusion models"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.original_model = None
        self.optimized_model = None
        self.optimization_results = {}
    
    def optimize_model(self, model: nn.Module, sample_input: torch.Tensor, 
                      robot_types: torch.Tensor) -> nn.Module:
        """Apply all optimization techniques to the model"""
        print("Starting edge optimization...")
        
        self.original_model = model
        optimized_model = model.eval()
        
        # Store original metrics
        original_metrics = self._measure_model_performance(
            optimized_model, sample_input, robot_types
        )
        
        # Apply optimizations
        if self.config.use_pruning:
            print("Applying pruning...")
            optimized_model = self._apply_pruning(optimized_model)
        
        if self.config.use_quantization:
            print("Applying quantization...")
            optimized_model = self._apply_quantization(optimized_model, sample_input, robot_types)
        
        if self.config.use_torch_jit:
            print("Applying TorchScript optimization...")
            optimized_model = self._apply_torch_jit(optimized_model, sample_input, robot_types)
        
        # Measure optimized metrics
        optimized_metrics = self._measure_model_performance(
            optimized_model, sample_input, robot_types
        )
        
        # Store results
        self.optimization_results = {
            'original': original_metrics,
            'optimized': optimized_metrics,
            'improvements': self._calculate_improvements(original_metrics, optimized_metrics)
        }
        
        self.optimized_model = optimized_model
        return optimized_model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to reduce model parameters"""
        if self.config.structured_pruning:
            # Structured pruning (remove entire channels/filters)
            return self._apply_structured_pruning(model)
        else:
            # Unstructured pruning (remove individual weights)
            return self._apply_unstructured_pruning(model)
    
    def _apply_unstructured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply unstructured magnitude-based pruning"""
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                # Only prune large layers to avoid destroying small critical layers
                if module.weight.numel() > 1000:
                    parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            print("Warning: No suitable parameters found for pruning")
            return model
        
        # Apply conservative pruning amount
        actual_amount = min(self.config.pruning_amount, 0.3)  # Cap at 30%
        print(f"Applying pruning to {len(parameters_to_prune)} modules with amount: {actual_amount}")
        
        try:
            # Apply pruning one module at a time for better control
            for module, param_name in parameters_to_prune:
                prune.l1_unstructured(module, param_name, amount=actual_amount)
                # Make pruning permanent immediately
                prune.remove(module, param_name)
        except Exception as e:
            print(f"Error during pruning: {e}")
            # Return original model if pruning fails
            return model
        
        return model
    
    def _apply_structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning (remove entire neurons/channels)"""
        # This is more complex and model-specific
        # For now, we'll use a simplified approach
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                if module.weight.size(0) > 64:  # Only prune larger layers
                    prune.ln_structured(
                        module, name='weight', amount=self.config.pruning_amount, 
                        n=2, dim=0
                    )
                    prune.remove(module, 'weight')
        
        return model
    
    def _apply_quantization(self, model: nn.Module, sample_input: torch.Tensor, 
                           robot_types: torch.Tensor) -> nn.Module:
        """Apply quantization to reduce model precision"""
        if self.config.quantization_mode == "dynamic":
            return self._apply_dynamic_quantization(model)
        elif self.config.quantization_mode == "static":
            return self._apply_static_quantization(model, sample_input, robot_types)
        elif self.config.quantization_mode == "qat":
            return self._apply_qat(model, sample_input, robot_types)
        else:
            raise ValueError(f"Unknown quantization mode: {self.config.quantization_mode}")
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization (weights quantized, activations in FP32)"""
        try:
            torch.backends.quantized.engine = self.config.quantization_backend
            
            # Use simple dynamic quantization for linear layers only
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Only quantize Linear layers
                dtype=torch.qint8,
                inplace=False
            )
            
            return quantized_model
        except Exception as e:
            print(f"Dynamic quantization failed: {e}")
            print("Returning original model without quantization")
            return model
    
    def _apply_static_quantization(self, model: nn.Module, sample_input: torch.Tensor,
                                  robot_types: torch.Tensor) -> nn.Module:
        """Apply static quantization (both weights and activations quantized)"""
        torch.backends.quantized.engine = self.config.quantization_backend
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig(self.config.quantization_backend)
        
        # Prepare model
        prepared_model = torch.quantization.prepare(model, inplace=False)
        
        # Calibration (run a few forward passes with representative data)
        prepared_model.eval()
        with torch.no_grad():
            for _ in range(10):  # Calibration passes
                timesteps = torch.randint(0, 1000, (sample_input.size(0),))
                _ = prepared_model(sample_input, timesteps, robot_types)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        return quantized_model
    
    def _apply_qat(self, model: nn.Module, sample_input: torch.Tensor,
                   robot_types: torch.Tensor) -> nn.Module:
        """Apply Quantization Aware Training (requires retraining)"""
        # This would require retraining the model with fake quantization
        # For now, we'll use dynamic quantization as fallback
        print("QAT requires retraining - falling back to dynamic quantization")
        return self._apply_dynamic_quantization(model)
    
    def _apply_torch_jit(self, model: nn.Module, sample_input: torch.Tensor,
                        robot_types: torch.Tensor) -> nn.Module:
        """Apply TorchScript compilation for optimization"""
        model.eval()
        
        # Create example inputs for tracing
        timesteps = torch.randint(0, 1000, (sample_input.size(0),))
        
        try:
            # For quantized models, tracing might not work well
            if any('quantized' in str(type(m)) for m in model.modules()):
                print("Skipping TorchScript for quantized model")
                return model
            
            # Try tracing first (faster)
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model, 
                    (sample_input, timesteps, robot_types),
                    strict=False,
                    check_trace=False  # Disable trace checking for complex models
                )
            
            if self.config.optimize_for_inference:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            return traced_model
            
        except Exception as e:
            print(f"TorchScript optimization failed: {e}")
            print("Returning original model without TorchScript compilation")
            return model
    
    def _measure_model_performance(self, model: nn.Module, sample_input: torch.Tensor,
                                  robot_types: torch.Tensor) -> Dict:
        """Measure model performance metrics"""
        model.eval()
        
        # Model size
        model_size_mb = self._get_model_size_mb(model)
        
        # Inference time
        inference_time_ms = self._measure_inference_time(model, sample_input, robot_types)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory usage
        memory_mb = self._estimate_memory_usage(model, sample_input)
        
        return {
            'model_size_mb': model_size_mb,
            'inference_time_ms': inference_time_ms,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': memory_mb
        }
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        try:
            if hasattr(model, 'state_dict'):
                # Standard PyTorch model
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                return (param_size + buffer_size) / (1024 * 1024)
            else:
                # TorchScript model or other - use rough estimation
                total_params = 0
                try:
                    for param in model.parameters():
                        total_params += param.numel()
                    # Assume 4 bytes per parameter for FP32, 1 byte for quantized
                    bytes_per_param = 1 if any('quantized' in str(type(m)) for m in model.modules()) else 4
                    return (total_params * bytes_per_param) / (1024 * 1024)
                except:
                    # Fallback for very unusual model types
                    return 0.0
        except Exception as e:
            print(f"Warning: Could not calculate model size: {e}")
            return 0.0
    
    def _measure_inference_time(self, model: nn.Module, sample_input: torch.Tensor,
                               robot_types: torch.Tensor, num_runs: int = 100) -> float:
        """Measure average inference time"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                timesteps = torch.randint(0, 1000, (sample_input.size(0),))
                try:
                    # Try normal forward pass first
                    _ = model(sample_input, timesteps, robot_types)
                except Exception as e:
                    try:
                        # Try without robot_types for simpler models
                        _ = model(sample_input, timesteps)
                    except Exception as e2:
                        try:
                            # For TorchScript or very simple models
                            _ = model(sample_input)
                        except Exception as e3:
                            print(f"Warning: Could not run forward pass during warmup: {e3}")
                            break
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        successful_runs = 0
        with torch.no_grad():
            for _ in range(num_runs):
                timesteps = torch.randint(0, 1000, (sample_input.size(0),))
                try:
                    _ = model(sample_input, timesteps, robot_types)
                    successful_runs += 1
                except Exception:
                    try:
                        _ = model(sample_input, timesteps)
                        successful_runs += 1
                    except Exception:
                        try:
                            _ = model(sample_input)
                            successful_runs += 1
                        except Exception:
                            continue  # Skip this run
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        if successful_runs == 0:
            print("Warning: No successful inference runs")
            return float('inf')
        
        avg_time_ms = ((end_time - start_time) / successful_runs) * 1000
        return avg_time_ms
    
    def _estimate_memory_usage(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Estimate memory usage during inference"""
        # Rough estimation based on model parameters and input size
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        input_memory = sample_input.numel() * sample_input.element_size()
        
        # Estimate activation memory (rough approximation)
        activation_memory = input_memory * 10  # Very rough estimate
        
        total_memory_mb = (param_memory + input_memory + activation_memory) / (1024 * 1024)
        return total_memory_mb
    
    def _calculate_improvements(self, original: Dict, optimized: Dict) -> Dict:
        """Calculate improvement percentages"""
        improvements = {}
        
        for key in original:
            if key in optimized and isinstance(original[key], (int, float)):
                if key in ['inference_time_ms', 'model_size_mb', 'total_params', 'memory_mb']:
                    # Lower is better
                    improvement = ((original[key] - optimized[key]) / original[key]) * 100
                    improvements[f"{key}_reduction_percent"] = improvement
                else:
                    # Higher might be better (depends on metric)
                    improvement = ((optimized[key] - original[key]) / original[key]) * 100
                    improvements[f"{key}_change_percent"] = improvement
        
        return improvements
    
    def save_optimized_model(self, output_path: str):
        """Save the optimized model"""
        if self.optimized_model is None:
            raise ValueError("No optimized model available. Run optimize_model first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        if hasattr(self.optimized_model, 'save'):
            # TorchScript model
            self.optimized_model.save(str(output_path))
        else:
            # Regular PyTorch model
            torch.save(self.optimized_model.state_dict(), output_path)
        
        # Save optimization results
        results_path = output_path.parent / f"{output_path.stem}_optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        
        print(f"Optimized model saved to: {output_path}")
        print(f"Optimization results saved to: {results_path}")
    
    def print_optimization_summary(self):
        """Print a summary of optimization results"""
        if not self.optimization_results:
            print("No optimization results available.")
            return
        
        original = self.optimization_results['original']
        optimized = self.optimization_results['optimized']
        improvements = self.optimization_results['improvements']
        
        print("\n" + "="*60)
        print("EDGE OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"\nModel Size:")
        print(f"  Original: {original['model_size_mb']:.2f} MB")
        print(f"  Optimized: {optimized['model_size_mb']:.2f} MB")
        print(f"  Reduction: {improvements.get('model_size_mb_reduction_percent', 0):.1f}%")
        
        print(f"\nInference Time:")
        print(f"  Original: {original['inference_time_ms']:.2f} ms")
        print(f"  Optimized: {optimized['inference_time_ms']:.2f} ms")
        print(f"  Reduction: {improvements.get('inference_time_ms_reduction_percent', 0):.1f}%")
        
        print(f"\nParameters:")
        print(f"  Original: {original['total_params']:,}")
        print(f"  Optimized: {optimized['total_params']:,}")
        print(f"  Reduction: {improvements.get('total_params_reduction_percent', 0):.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"  Original: {original['memory_mb']:.2f} MB")
        print(f"  Optimized: {optimized['memory_mb']:.2f} MB")
        print(f"  Reduction: {improvements.get('memory_mb_reduction_percent', 0):.1f}%")
        
        # Check if targets are met
        print(f"\nTarget Achievement:")
        target_size_met = optimized['model_size_mb'] <= self.config.target_model_size_mb
        target_time_met = optimized['inference_time_ms'] <= self.config.target_inference_time_ms
        
        print(f"  Model Size Target ({self.config.target_model_size_mb} MB): {'✅' if target_size_met else '❌'}")
        print(f"  Inference Time Target ({self.config.target_inference_time_ms} ms): {'✅' if target_time_met else '❌'}")


def optimize_trained_model(model_path: str, output_dir: str, config: Optional[OptimizationConfig] = None):
    """Optimize a trained diffusion model for edge deployment"""
    if config is None:
        config = OptimizationConfig()
    
    # Load trained model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract config
    model_config_dict = checkpoint.get('config', {}).get('model_config', {})
    if isinstance(model_config_dict, dict):
        model_config = ModelConfig(**model_config_dict)
    else:
        model_config = ModelConfig(max_state_dim=72)
    
    # Create model
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.diffusion_model import create_diffusion_model
    model = create_diffusion_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create sample inputs
    batch_size = 4
    seq_len = model_config.max_sequence_length
    state_dim = model_config.max_state_dim
    
    sample_input = torch.randn(batch_size, seq_len, state_dim)
    robot_types = torch.randint(0, len(model_config.robot_types), (batch_size,))
    
    # Optimize
    optimizer = EdgeOptimizer(config)
    optimized_model = optimizer.optimize_model(model, sample_input, robot_types)
    
    # Save results
    output_path = Path(output_dir)
    optimizer.save_optimized_model(output_path / "optimized_model.pt")
    optimizer.print_optimization_summary()
    
    return optimized_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize diffusion model for edge deployment")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="optimized_models",
                        help="Output directory for optimized model")
    parser.add_argument("--target_size_mb", type=float, default=10.0,
                        help="Target model size in MB")
    parser.add_argument("--target_time_ms", type=float, default=50.0,
                        help="Target inference time in ms")
    parser.add_argument("--pruning_amount", type=float, default=0.3,
                        help="Amount of parameters to prune (0.0-1.0)")
    parser.add_argument("--no_quantization", action="store_true",
                        help="Disable quantization")
    parser.add_argument("--no_pruning", action="store_true",
                        help="Disable pruning")
    
    args = parser.parse_args()
    
    # Create optimization config
    config = OptimizationConfig(
        target_model_size_mb=args.target_size_mb,
        target_inference_time_ms=args.target_time_ms,
        pruning_amount=args.pruning_amount,
        use_quantization=not args.no_quantization,
        use_pruning=not args.no_pruning
    )
    
    # Run optimization
    optimize_trained_model(args.model_path, args.output_dir, config)
