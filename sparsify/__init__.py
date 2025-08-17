# sparsification/__init__.py
from .progressive_joint import QuantizationInformedSparsification
from .baseline import MagnitudeSparsification
from .dynamic_coevolution import DynamicCoEvolutionSparsification

def create_sparsifier(config):
    """Factory function - integrates with your config system"""
    if not config.sparsification.enabled:
        return None
        
    method = config.sparsification.method
    
    if method == "quantization_informed_progressive":
        return QuantizationInformedSparsification(
        )
    
    elif method == "dynamic_coevolution":  # NEW: Single integrated method
       return DynamicCoEvolutionSparsification(
           adaptation_interval=getattr(config.sparsification, 'adaptation_interval', 5),
           initial_sw_target=getattr(config.sparsification, 'initial_sw_target', 0.25),
           sw_learning_rate=getattr(config.sparsification, 'sw_learning_rate', 0.1),
           efficiency_threshold=getattr(config.sparsification, 'efficiency_threshold', 200.0)
       )
    elif method == "magnitude":
        return MagnitudeSparsification()
    else:
        raise ValueError(f"Unknown sparsification method: {method}")

__all__ = ['create_sparsifier', 'QuantizationInformedSparsification']