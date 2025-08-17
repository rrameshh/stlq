import torch
import torch.nn as nn
from typing import Dict, Any

class DynamicCoEvolutionSparsification:
    def __init__(self, 
                 adaptation_interval: int = 5,
                 initial_sw_target: float = 0.25,
                 sw_learning_rate: float = 0.1,
                 efficiency_threshold: float = 200.0):
        

        self.adaptation_interval = adaptation_interval
        self.initial_sw_target = initial_sw_target
        self.sw_learning_rate = sw_learning_rate
        self.efficiency_threshold = efficiency_threshold
        
        self.layer_sw_targets = {}
        self.layer_sparsity_budgets = {}
        self.layer_efficiency_history = {}
        self.initialized = False
        self.sparsification_applied = False
        
    def get_name(self) -> str:
        return "dynamic_coevolution"
    
    def apply(self, model: nn.Module, target_sparsity: float) -> Dict[str, Any]:
        """
        Main entry point - called once when sparsification epoch is reached
        """
        print(f"\nDynamic Co-Evolution Sparsification (target: {target_sparsity:.1%})")
        
        # Step 1: Initialize adaptive targets (breaks 0.25 lock immediately)
        if not self.initialized:
            self._initialize_layer_targets(model)
            self.initialized = True
        
        # Step 2: Apply sparsification using current targets
        results = self._apply_sparsification_with_current_targets(model, target_sparsity)
        self.sparsification_applied = True
        
        return results
    
    def adapt_targets(self, model, epoch):
        """
        Called every epoch from trainer to adapt quantization targets
        """
        if not self.initialized or not self.sparsification_applied:
            return
            
        if epoch % self.adaptation_interval == 0:
            self._perform_adaptation_step(model, epoch)
    
    def _initialize_layer_targets(self, model):
        
        for name, module in model.named_modules():
            if not hasattr(module, 'strategy'):
                continue
                
            # Calculate diverse initial target based on layer characteristics
            layer_depth = self._get_layer_depth(name)
            layer_type = self._get_layer_type(name)
            
            # Base target varies by depth
            if layer_depth < 3:
                base_target = 0.12  # Early layers: conservative
            elif layer_depth < 9:
                base_target = 0.20 + (layer_depth - 3) * 0.02  # Middle: gradual increase
            else:
                base_target = 0.32  # Late layers: aggressive
            
            # Adjust by layer type
            if layer_type == 'attention_qkv':
                base_target *= 1.15  # QKV needs more precision
            elif layer_type == 'attention_proj':
                base_target *= 1.05  # Projection moderate precision
            elif layer_type == 'mlp_fc1':
                base_target *= 0.95  # First MLP layer can be compressed
            elif layer_type == 'mlp_fc2':
                base_target *= 0.85  # Output layer more compressible
            
            # Add small random variation to ensure diversity
            import random
            random.seed(hash(name) % 2**32)  # Deterministic per layer
            variation = 0.9 + random.random() * 0.2  # ±10%
            final_target = base_target * variation
            
            # Clamp to reasonable range
            final_target = max(0.05, min(0.45, final_target))
            
            # Store and apply immediately
            self.layer_sw_targets[name] = final_target
            self.layer_sparsity_budgets[name] = 1.0  # Start with neutral sparsity budget
            self.layer_efficiency_history[name] = []
            
            # 🔥 CRITICAL: Apply new target immediately to break 0.25 lock
            old_target = module.strategy.target_second_word_ratio
            module.strategy.target_second_word_ratio = final_target
            
            print(f"      {name}: {old_target:.3f} → {final_target:.3f}")
    
    

    def _apply_sparsification_with_current_targets(self, model, target_sparsity):
        
        # Collect costs efficiently (don't store all in memory)
        layer_info = []
        cost_samples = []  # Just samples, not full tensors
        
        for name, module in model.named_modules():
            if not self._is_quantizable_layer(module) or self._is_critical_layer(name):
                continue
                
            weight = self._get_weight_tensor(module)
            cost_map = self._compute_quantization_cost(module, weight)
            
            # Store layer info
            layer_info.append((name, module, weight, cost_map))
            
            # Sample costs for threshold estimation (memory efficient)
            flat_costs = cost_map.flatten()
            if flat_costs.numel() > 50000:  # Sample large tensors
                sample_indices = torch.randperm(flat_costs.numel())[:50000]
                cost_samples.extend(flat_costs[sample_indices].cpu().tolist())
            else:
                cost_samples.extend(flat_costs.cpu().tolist())
            
            current_target = self.layer_sw_targets.get(name, 0.25)
            print(f"   {name}: mean_cost={cost_map.mean():.1f}, SW_target={current_target:.3f}")
        
        if not cost_samples:
            return {"actual_sparsity": 0.0, "method": self.get_name()}
        
        # Memory-efficient threshold computation
        print(f"   Computing threshold from {len(cost_samples):,} sampled costs...")
        
        # Convert to tensor and compute quantile
        cost_tensor = torch.tensor(cost_samples, dtype=torch.float32)
        threshold = torch.quantile(cost_tensor, target_sparsity)
        
        print(f"   Global threshold: {threshold:.1f}")
        
        # Apply sparsification layer by layer (memory efficient)
        total_params = 0
        total_pruned = 0
        layer_results = []
        
        for name, module, weight, cost_map in layer_info:
            # Process in chunks if tensor is large
            if weight.numel() > 1000000:  # 1M threshold
                mask = self._chunked_pruning(weight, cost_map, threshold)
            else:
                mask = (cost_map <= threshold).float()
            
            # Apply mask
            weight.data *= mask
            
            # Count results
            pruned = (mask == 0).sum().item()
            total = weight.numel()
            
            total_params += total
            total_pruned += pruned
            
            layer_results.append({
                'name': name,
                'sparsity': pruned / total,
                'pruned_count': pruned,
                'sw_target': self.layer_sw_targets.get(name, 0.25),
                'avg_cost': cost_map.mean().item()
            })
            
            print(f"   {name}: pruned {pruned/total:.1%}")
            
            # Clean up memory
            del cost_map
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        actual_sparsity = total_pruned / total_params if total_params > 0 else 0
        print(f"   ✅ Final sparsity: {actual_sparsity:.1%}")
        
        return {
            "actual_sparsity": actual_sparsity,
            "method": self.get_name(),
            "threshold": threshold.item(),
            "layer_results": layer_results,
            "adaptive_targets": dict(self.layer_sw_targets)
        }

    def _chunked_pruning(self, weight, cost_map, threshold):
        """Process large tensors in chunks to avoid memory issues"""
        print(f"      Processing large tensor ({weight.numel():,} elements) in chunks...")
        
        mask = torch.ones_like(weight, dtype=torch.float32)
        chunk_size = 500000  # 500K elements per chunk
        
        # Flatten for processing
        flat_weight = weight.view(-1)
        flat_cost = cost_map.view(-1)
        flat_mask = mask.view(-1)
        
        for i in range(0, flat_weight.numel(), chunk_size):
            end_idx = min(i + chunk_size, flat_weight.numel())
            
            # Process chunk
            chunk_cost = flat_cost[i:end_idx]
            chunk_mask = (chunk_cost <= threshold).float()
            
            # Apply to weight and mask
            flat_weight[i:end_idx] *= chunk_mask
            flat_mask[i:end_idx] = chunk_mask
            
            # Progress indicator for large tensors
            if end_idx % (chunk_size * 4) == 0:
                progress = end_idx / flat_weight.numel() * 100
                print(f"         Progress: {progress:.1f}%")
        
        return mask

    def _safe_quantile(self, tensor_list, quantile_value, max_samples=100000):
        """
        Safe quantile computation that handles large tensors
        """
        all_samples = []
        
        for tensor in tensor_list:
            flat_tensor = tensor.flatten()
            
            if flat_tensor.numel() > max_samples // len(tensor_list):
                # Sample from large tensors
                sample_size = max_samples // len(tensor_list)
                indices = torch.randperm(flat_tensor.numel())[:sample_size]
                samples = flat_tensor[indices]
            else:
                samples = flat_tensor
            
            all_samples.append(samples.cpu())
        
        # Concatenate all samples
        if all_samples:
            combined_samples = torch.cat(all_samples)
            return torch.quantile(combined_samples, quantile_value)
        else:
            return torch.tensor(0.0)


    def _perform_adaptation_step(self, model, epoch):
        """Adapt quantization targets based on efficiency feedback"""
        print(f"\nCo-Evolution Adaptation Step (Epoch {epoch})")
        
        for name, module in model.named_modules():
            if name not in self.layer_sw_targets:
                continue
                
            # Measure efficiency
            efficiency = self._measure_layer_efficiency(module, name)
            self.layer_efficiency_history[name].append(efficiency)
            
            # Adapt target
            old_target = self.layer_sw_targets[name]
            new_target = self._adapt_single_layer_target(name, efficiency)
            
            # Apply new target
            module.strategy.target_second_word_ratio = new_target
            self.layer_sw_targets[name] = new_target
            
            if abs(new_target - old_target) > 0.01:  # Only log significant changes
                print(f"   📈 {name}: {old_target:.3f} → {new_target:.3f} "
                      f"(eff: {efficiency:.1f})")
    
    def _compute_quantization_cost(self, module, weight):
        """Compute quantization cost using current targets"""
        try:
            strategy = getattr(module, 'strategy', None)
            if not strategy:
                return 1.0 / (torch.abs(weight.data) + 1e-8)
            
            # Get quantization with current adaptive target
            quantized = strategy.quantize_weight(weight, per_channel=True)
            
            if hasattr(quantized, 'second_word_mask') and quantized.second_word_mask is not None:
                sw_mask = quantized.second_word_mask
                magnitude = torch.abs(weight.data)
                
                # Cost: higher for second-word weights, inversely related to magnitude
                cost = torch.ones_like(weight.data, dtype=torch.float32)
                cost[sw_mask] = 3.0  # Second word penalty
                cost[~sw_mask] = 1.0  # Single word
                
                return cost / (magnitude + 1e-6)
            else:
                return 1.0 / (torch.abs(weight.data) + 1e-8)
                
        except Exception as e:
            return 1.0 / (torch.abs(weight.data) + 1e-8)
    
    def _measure_layer_efficiency(self, module, name):
        """Measure current layer efficiency"""
        try:
            weight = self._get_weight_tensor(module)
            quantized = module.strategy.quantize_weight(weight, per_channel=True)
            
            if hasattr(quantized, 'second_word_mask'):
                sw_ratio = quantized.second_word_mask.float().mean().item()
                # Efficiency = balance between compression and accuracy
                # Lower SW ratio = better compression
                # But too low might hurt accuracy
                target = self.layer_sw_targets.get(name, 0.25)
                
                # Penalty for being too far from target
                deviation_penalty = abs(sw_ratio - target) * 100
                base_efficiency = (1.0 - sw_ratio) * 300  # Reward compression
                
                return max(50, base_efficiency - deviation_penalty)
            else:
                return 150.0  # Default
                
        except Exception:
            return 150.0
    
    def _adapt_single_layer_target(self, name, efficiency):
        """Adapt a single layer's target based on efficiency"""
        current_target = self.layer_sw_targets[name]
        
        # Simple adaptation rule
        if efficiency < self.efficiency_threshold:
            # Poor efficiency - try reducing second word usage
            new_target = current_target * 0.95
        elif efficiency > self.efficiency_threshold * 1.3:
            # Very good efficiency - can afford more precision
            new_target = current_target * 1.02
        else:
            # Good efficiency - small random exploration
            import random
            factor = 0.98 + random.random() * 0.04
            new_target = current_target * factor
        
        return max(0.05, min(0.45, new_target))
    
    # Helper methods
    def _get_layer_depth(self, name):
        if 'blocks.' in name:
            try:
                return int(name.split('blocks.')[1].split('.')[0])
            except:
                return 6
        return 0
    
    def _get_layer_type(self, name):
        if 'attn.qkv' in name:
            return 'attention_qkv'
        elif 'attn.proj' in name:
            return 'attention_proj'
        elif 'mlp.fc1' in name:
            return 'mlp_fc1'
        elif 'mlp.fc2' in name:
            return 'mlp_fc2'
        else:
            return 'other'
    
    def _is_quantizable_layer(self, module):
        return hasattr(module, 'strategy') and (hasattr(module, 'linear') or hasattr(module, 'conv2d'))
    
    def _is_critical_layer(self, name):
        critical_patterns = ['patch_embed', 'head', 'norm', 'cls_token', 'pos_embed']
        return any(pattern in name for pattern in critical_patterns)
    
    def _get_weight_tensor(self, module):
        if hasattr(module, 'linear'):
            return module.linear.weight
        elif hasattr(module, 'conv2d'):
            return module.conv2d.weight
        else:
            raise ValueError(f"Cannot extract weight from {type(module)}")

__all__ = ['DynamicCoEvolutionSparsification']