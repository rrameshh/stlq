import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import time

class ActivationAwareSparsification:
    def __init__(self):
        self.activation_stats = {}
        self.name = "activation_aware"
    
    def collect_activation_statistics(self, model, data_loader, num_batches=10):
        print(f"Collecting activation statistics from {num_batches} batches...")
        start_time = time.time()
        
        activation_data = {}
        
        def make_activation_hook(layer_name):
            def hook_fn(module, input, output):
                if layer_name not in activation_data: 
                    activation_data[layer_name] = []
                
                # Get the input tensor (first element if tuple)
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                # store mean absolute activation for this batch
                mean_abs_activation = torch.mean(torch.abs(input_tensor)).item()
                activation_data[layer_name].append(mean_abs_activation)
            
            return hook_fn
        
        # register hooks for quantizable layers
        hooks = []
        quantizable_layers = self._find_quantizable_layers(model)
        
        print(f"Found {len(quantizable_layers)} quantizable layers")
        
        for layer_name, layer_module in quantizable_layers:
            hook = layer_module.register_forward_hook(make_activation_hook(layer_name))
            hooks.append(hook)
        
        # Collect statistics
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                
                _ = model(inputs)
                
                if (batch_idx + 1) % max(1, num_batches // 4) == 0:
                    print(f"  Processed batch {batch_idx + 1}/{num_batches}")
        

        for hook in hooks:
            hook.remove()

        self.activation_stats = {}
        for layer_name, values in activation_data.items():
            self.activation_stats[layer_name] = {
                'mean_abs_activations': np.mean(values),
                'std_abs_activations': np.std(values),
                'num_samples': len(values)
            }
        
        elapsed_time = time.time() - start_time
        print(f"Activation statistics collected in {elapsed_time:.2f}s")
        print(f"Statistics available for {len(self.activation_stats)} layers")
        
        print("\nSample activation statistics:")
        for i, (layer_name, stats) in enumerate(self.activation_stats.items()):
            if i < 5:
                print(f"  {layer_name}: mean={stats['mean_abs_activations']:.4f}, "
                      f"std={stats['std_abs_activations']:.4f}")
            elif i == 5:
                print(f"  ... and {len(self.activation_stats) - 5} more layers")
                break
        
        return self.activation_stats
    
    def apply_sparsification(self, model, target_sparsity):
        print(f"\nApplying activation-aware sparsification (target: {target_sparsity:.1%})")
        
        if not self.activation_stats:
            raise ValueError("Must collect activation statistics before applying sparsification")
        
        # collect importance scores
        print("Computing importance scores...")
        all_importance_scores = []
        
        quantizable_layers = self._find_quantizable_layers(model)
        
        for layer_name, layer_module in quantizable_layers:
            if layer_name not in self.activation_stats:
                print(f"Warning: No activation stats for {layer_name}, skipping...")
                continue

            weight_tensor = self._get_weight_tensor(layer_module)
            if weight_tensor is None:
                continue
            
            # Compute importance scores: |weight| × expected_activation
            activation_magnitude = self.activation_stats[layer_name]['mean_abs_activations']
            importance_scores = torch.abs(weight_tensor) * activation_magnitude
            
            all_importance_scores.extend(importance_scores.flatten().tolist())

        total_weights = len(all_importance_scores)
        weights_to_prune = int(target_sparsity * total_weights)
        
        print(f"Total weights: {total_weights:,}")
        print(f"Weights to prune: {weights_to_prune:,}")
        
        # find global importance threshold 
        print("Computing global threshold...")
        threshold = self._compute_quantile_safe(all_importance_scores, target_sparsity)
        
        print(f"Importance threshold: {threshold:.6f}")
        
        del all_importance_scores
        
        # apply sparsification masks layer by layer
        print("Applying sparsification masks...")
        layer_results = {}
        total_pruned = 0
        
        for layer_name, layer_module in quantizable_layers:
            if layer_name not in self.activation_stats:
                continue
                
            weight_tensor = self._get_weight_tensor(layer_module)
            if weight_tensor is None:
                continue
            
            # compute importance for this layer
            activation_magnitude = self.activation_stats[layer_name]['mean_abs_activations']
            importance_scores = torch.abs(weight_tensor) * activation_magnitude
            
            # create sparsity mask
            sparsity_mask = importance_scores >= threshold
            
            # apply mask
            weight_tensor.data *= sparsity_mask.float()
            
            # Calculate layer statistics
            layer_total = weight_tensor.numel()
            layer_pruned = (sparsity_mask == 0).sum().item()
            layer_sparsity = layer_pruned / layer_total
            
            layer_results[layer_name] = {
                'total_weights': layer_total,
                'pruned_weights': layer_pruned,
                'sparsity': layer_sparsity,
                'activation_magnitude': activation_magnitude,
                'importance_threshold': threshold
            }
            
            total_pruned += layer_pruned
            
            print(f"  {layer_name}: {layer_sparsity:.1%} sparsity "
                  f"({layer_pruned:,}/{layer_total:,} weights)")
        
        # Final statistics
        actual_sparsity = total_pruned / total_weights
        
        results = {
            'method': self.name,
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'total_weights': total_weights,
            'total_pruned': total_pruned,
            'importance_threshold': threshold,
            'layer_results': layer_results
        }
        
        print(f"\nSparsification complete!")
        print(f"Target sparsity: {target_sparsity:.1%}")
        print(f"Actual sparsity: {actual_sparsity:.1%}")
        print(f"Total pruned: {total_pruned:,}/{total_weights:,} weights")
        
        return results
    
    def _find_quantizable_layers(self, model):
        quantizable_layers = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'strategy') and hasattr(module.strategy, 'quantize_weight'):
                quantizable_layers.append((name, module))
            elif hasattr(module, 'linear') or hasattr(module, 'conv2d'):
                quantizable_layers.append((name, module))
        
        return quantizable_layers
    
    def _get_weight_tensor(self, layer_module):
        if hasattr(layer_module, 'linear') and hasattr(layer_module.linear, 'weight'):
            return layer_module.linear.weight
        elif hasattr(layer_module, 'conv2d') and hasattr(layer_module.conv2d, 'weight'):
            return layer_module.conv2d.weight
        elif hasattr(layer_module, 'weight'):
            return layer_module.weight
        else:
            return None
    
    def get_sparsification_summary(self, model):
        summary = {
            'total_weights': 0,
            'zero_weights': 0,
            'layer_sparsities': {}
        }
        
        quantizable_layers = self._find_quantizable_layers(model)
        
        for layer_name, layer_module in quantizable_layers:
            weight_tensor = self._get_weight_tensor(layer_module)
            if weight_tensor is None:
                continue
            
            total = weight_tensor.numel()
            zeros = (weight_tensor == 0).sum().item()
            sparsity = zeros / total
            
            summary['total_weights'] += total
            summary['zero_weights'] += zeros
            summary['layer_sparsities'][layer_name] = sparsity
        
        summary['overall_sparsity'] = summary['zero_weights'] / summary['total_weights']
        
        return summary
    
    def _compute_quantile_safe(self, values, quantile):
        n_values = len(values)
        
        if n_values < 1_000_000:
            return torch.quantile(torch.tensor(values), quantile).item()
        elif n_values < 50_000_000:
            import numpy as np
            return float(np.quantile(values, quantile))
        else:
            # use chunking
            import numpy as np
            print(f"Using chunked quantile computation for {n_values:,} values...")
            
            # conver to numpy array in chunks to avoid memory issues
            chunk_size = 10_000_000
            all_chunks = []
            
            for i in range(0, n_values, chunk_size):
                chunk = values[i:i + chunk_size]
                all_chunks.extend(chunk)
                
                # clean up memory
                if len(all_chunks) >= 20_000_000:
                    partial_quantile = np.quantile(all_chunks, quantile)
                    print(f"  Processed {i + len(chunk):,}/{n_values:,} values, partial quantile: {partial_quantile:.6f}")
                    if i > 30_000_000:  
                        # take a stratified sample of remaining values
                        remaining = values[i:]
                        sample_size = min(1_000_000, len(remaining))
                        step = max(1, len(remaining) // sample_size)
                        sampled = remaining[::step]
                        all_chunks.extend(sampled)
                        break
            
            return float(np.quantile(all_chunks, quantile))