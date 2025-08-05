# quantization/hooks.py - Missing from your refactored code
import torch
from typing import Optional

class SwitchQuantizationModeHook:
    """
    Hook to switch from calibration to activation quantization after N iterations.
    
    This is CRITICAL for QAT training - your refactored code is missing this!
    """
    
    def __init__(self, model, switch_iteration: int = 5000):
        """
        Args:
            model: Model with quantized layers
            switch_iteration: Iteration number to switch to activation quantization
        """
        self.model = model
        self.switch_iteration = switch_iteration
        self.switched = False
        self.current_iteration = 0
    
    def after_train_iter(self, iteration: int) -> bool:
        """
        Call this after each training iteration.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            bool: True if quantization mode was switched this iteration
        """
        self.current_iteration = iteration
        
        if not self.switched and iteration >= self.switch_iteration:
            self._enable_activation_quantization()
            self.switched = True
            return True
        
        return False
    
    def _enable_activation_quantization(self):
        """Enable activation quantization for all quantized layers."""
        count = 0
        for module in self.model.modules():
            if hasattr(module, 'activation_quantization'):
                module.activation_quantization = True
                count += 1
        
        print(f"🔄 Switched to activation quantization mode ({count} layers affected)")
    
    def get_status(self) -> dict:
        """Get current hook status."""
        return {
            'switched': self.switched,
            'current_iteration': self.current_iteration,
            'switch_iteration': self.switch_iteration,
            'iterations_remaining': max(0, self.switch_iteration - self.current_iteration)
        }


def create_switch_hook(model, switch_iteration: int = 5000) -> SwitchQuantizationModeHook:
    """Factory function to create switch hook."""
    return SwitchQuantizationModeHook(model, switch_iteration)