import torch
from typing import Optional

class SwitchQuantizationModeHook:

    def __init__(self, model, switch_iteration: int = 5000):

        self.model = model
        self.switch_iteration = switch_iteration
        self.switched = False
        self.current_iteration = 0
    
    def after_train_iter(self, iteration: int) -> bool:

        self.current_iteration = iteration
        
        if not self.switched and iteration >= self.switch_iteration:
            self._enable_activation_quantization()
            self.switched = True
            return True
        
        return False
    
    def _enable_activation_quantization(self):
        count = 0
        for module in self.model.modules():
            if hasattr(module, 'activation_quantization'):
                module.activation_quantization = True
                count += 1
        
        print(f"Switched to activation quantization mode ({count} layers affected)")
    
    def get_status(self) -> dict:
        return {
            'switched': self.switched,
            'current_iteration': self.current_iteration,
            'switch_iteration': self.switch_iteration,
            'iterations_remaining': max(0, self.switch_iteration - self.current_iteration)
        }


def create_switch_hook(model, switch_iteration: int = 5000) -> SwitchQuantizationModeHook:
    return SwitchQuantizationModeHook(model, switch_iteration)