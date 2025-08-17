import torch
import torch.nn as nn
from typing import Dict, Any
from .base import (
    SparsificationMethod, get_quantizable_layers, get_weight_tensor, 
    get_quantization_strategy, is_critical_layer
)



class QuantizationInformedSparsification(SparsificationMethod):
    
    def __init__(self, chunk_size: int = 1000000):
        super().__init__()
        self.chunk_size = chunk_size  # Process large tensors in chunks
    
    def apply(self, model: nn.Module, target_sparsity: float) -> Dict[str, Any]:
        print(f"\nProgressive Joint Sparsification (target: {target_sparsity:.1%})")
        
        cost_distribution = []
        layer_info = []
        
        for name, module in get_quantizable_layers(model):
            if is_critical_layer(name):
                continue
                
            weight = get_weight_tensor(module)
            cost_map = self._compute_quantization_cost(module, weight)
            
            # Store layer info and cost statistics (not full tensor)
            layer_info.append((name, module, weight, cost_map))
            cost_distribution.append(self._get_cost_stats(cost_map))
            
            print(f"   {name}: mean_cost={cost_map.mean():.3f} | shape={list(weight.shape)}")
        
        if not cost_distribution:
            return {"actual_sparsity": 0.0, "method": self.get_name()}
        
        # Step 2: Find global threshold using approximate quantile
        threshold = self._find_approximate_threshold(
            [stats["flattened_costs"] for stats in cost_distribution],
            target_sparsity
        )
        print(f"   📊 Global threshold: {threshold:.4f}")
        
        # Step 3: Apply pruning progressively
        total_params = 0
        total_pruned = 0
        layer_results = []
        
        for name, module, weight, cost_map in layer_info:
            # Process in chunks if tensor is large
            if weight.numel() > self.chunk_size:
                mask = self._chunked_pruning(weight, cost_map, threshold)
            else:
                mask = (cost_map <= threshold).float()
            
            # Apply mask
            weight.data *= mask
            
            # Record stats
            pruned = (mask == 0).sum().item()
            total = weight.numel()
            
            total_params += total
            total_pruned += pruned
            
            layer_results.append({
                'name': name,
                'sparsity': pruned / total,
                'pruned_count': pruned,
                'avg_cost': cost_map.mean().item()
            })
            
            print(f"   {name}: pruned {pruned/total:.1%} (kept {total-pruned})")
        
        actual_sparsity = total_pruned / total_params if total_params > 0 else 0
        print(f"   ✅ Final sparsity: {actual_sparsity:.1%}")
        
        return {
            "actual_sparsity": actual_sparsity,
            "method": self.get_name(),
            "threshold": threshold.item(),
            "layer_results": layer_results
        }
    
    def _compute_quantization_cost(self, module, weight: torch.Tensor) -> torch.Tensor:
        try:
            # Try to get quantization strategy
            strategy = getattr(module, 'strategy', None)
            if strategy is None:
                return 1.0 / (torch.abs(weight.data) + 1e-8)
            
            # Get quantized weights
            quantized = strategy.quantize_weight(weight, per_channel=True)
            
            if hasattr(quantized, 'second_word_mask'):
                # Use second-word pattern if available
                sw_mask = quantized.second_word_mask.float()
                cost = torch.where(sw_mask > 0, 3.0, 1.0)
                return cost / (torch.abs(weight.data) + 1e-6)
            else:
                # Fallback to quantization error
                dequantized = quantized.dequantize() if hasattr(quantized, 'dequantize') else quantized
                return torch.abs(weight.data - dequantized)
                
        except Exception as e:
            print(f"      Cost computation fallback: {str(e)}")
            return 1.0 / (torch.abs(weight.data) + 1e-8)
    
    def _get_cost_stats(self, cost_map: torch.Tensor) -> Dict[str, Any]:
        """Get statistics without storing full tensor"""
        return {
            "mean": cost_map.mean().item(),
            "std": cost_map.std().item(),
            "min": cost_map.min().item(),
            "max": cost_map.max().item(),
            "flattened_costs": cost_map.flatten().cpu()  # Move to CPU to save GPU memory
        }
    
    def _find_approximate_threshold(self, cost_chunks, target_sparsity):
        """Find threshold without concatenating all tensors"""
        # Collect sample quantiles from each chunk
        sample_quantiles = []
        for chunk in cost_chunks:
            if len(chunk) > 10000:  # Subsample large chunks
                chunk = chunk[torch.randperm(len(chunk))[:10000]]
            sample_quantiles.append(torch.quantile(chunk, target_sparsity))
        
        # Take median of sample quantiles as threshold
        return torch.median(torch.stack(sample_quantiles))
    
    def _chunked_pruning(self, weight, cost_map, threshold):
        """Process large tensors in chunks"""
        mask = torch.ones_like(weight)
        flat_weights = weight.view(-1)
        flat_costs = cost_map.view(-1)
        
        for i in range(0, len(flat_weights), self.chunk_size):
            chunk_end = min(i + self.chunk_size, len(flat_weights))
            chunk_mask = (flat_costs[i:chunk_end] <= threshold).float()
            flat_weights[i:chunk_end] *= chunk_mask
            
        return mask
    
    def get_name(self) -> str:
        return "progressive_joint_sparsification"