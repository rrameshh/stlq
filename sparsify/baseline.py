# sparsification/baseline_methods.py
import torch
from typing import Dict, Any
from .base import (
    SparsificationMethod, get_quantizable_layers, get_weight_tensor, is_critical_layer
)

class MagnitudeSparsification(SparsificationMethod):
    """Traditional magnitude-based sparsification for comparison"""
    
    def get_name(self) -> str:
        return "magnitude"
    
    def apply(self, model, target_sparsity: float) -> Dict[str, Any]:
        print(f"\n📏 Magnitude-based sparsification: {target_sparsity:.1%} target")
        
        # Collect all magnitudes for global threshold
        all_magnitudes = []
        layer_info = []
        
        for name, module in get_quantizable_layers(model):
            if is_critical_layer(name):
                continue
                
            weight = get_weight_tensor(module)
            magnitude = torch.abs(weight.data)
            all_magnitudes.append(magnitude.flatten())
            layer_info.append((name, module, weight, magnitude))
        
        if not all_magnitudes:
            return {"actual_sparsity": 0.0, "method": self.get_name()}
        
        # Global threshold
        global_magnitudes = torch.cat(all_magnitudes)
        threshold = torch.quantile(global_magnitudes, target_sparsity)
        
        # Apply sparsification
        total_params = 0
        total_pruned = 0
        layer_results = []
        
        for name, module, weight, magnitude in layer_info:
            mask = (magnitude > threshold).float()
            weight.data *= mask
            
            pruned = (mask == 0).sum().item()
            total = weight.numel()
            
            total_params += total
            total_pruned += pruned
            
            layer_results.append({
                'name': name,
                'sparsity': pruned / total,
                'pruned_weights': pruned,
                'total_weights': total
            })
            
            print(f"   {name}: {pruned/total:.1%} sparsity")
        
        actual_sparsity = total_pruned / total_params if total_params > 0 else 0
        
        return {
            "actual_sparsity": actual_sparsity,
            "method": self.get_name(),
            "layer_results": layer_results
        }

class SNIPSparsification(SparsificationMethod):
    """SNIP sparsification for comparison"""
    
    def get_name(self) -> str:
        return "snip"
    
    def apply(self, model, target_sparsity: float) -> Dict[str, Any]:
        print(f"\n🎯 SNIP sparsification: {target_sparsity:.1%} target")
        
        # This is a simplified SNIP - in practice you'd need gradients
        # For now, fall back to magnitude with some noise for demonstration
        print("   (Using magnitude + noise as SNIP approximation)")
        
        all_scores = []
        layer_info = []
        
        for name, module in get_quantizable_layers(model):
            if is_critical_layer(name):
                continue
                
            weight = get_weight_tensor(module)
            # Approximate SNIP score with magnitude + noise
            magnitude = torch.abs(weight.data)
            noise = torch.randn_like(magnitude) * 0.1
            score = magnitude + noise
            
            all_scores.append(score.flatten())
            layer_info.append((name, module, weight, score))
        
        if not all_scores:
            return {"actual_sparsity": 0.0, "method": self.get_name()}
        
        # Global threshold
        global_scores = torch.cat(all_scores)
        threshold = torch.quantile(global_scores, target_sparsity)
        
        # Apply sparsification
        total_params = 0
        total_pruned = 0
        layer_results = []
        
        for name, module, weight, score in layer_info:
            mask = (score > threshold).float()
            weight.data *= mask
            
            pruned = (mask == 0).sum().item()
            total = weight.numel()
            
            total_params += total
            total_pruned += pruned
            
            layer_results.append({
                'name': name,
                'sparsity': pruned / total,
                'pruned_weights': pruned
            })
            
            print(f"   {name}: {pruned/total:.1%} sparsity")
        
        actual_sparsity = total_pruned / total_params if total_params > 0 else 0
        
        return {
            "actual_sparsity": actual_sparsity,
            "method": self.get_name(),
            "layer_results": layer_results
        }