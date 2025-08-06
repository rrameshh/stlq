from abc import ABC, abstractmethod
import torch

class QuantizationStrategy(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def quantize_weight(self, weight: torch.Tensor, per_channel: bool = True):
        """Quantize a weight tensor."""
        pass


    @abstractmethod
    def quantize_bias(self, bias: torch.Tensor, quantized_input: torch.Tensor, 
                     quantized_weight: torch.Tensor):
        """Quantize bias tensor (typically requires input and weight scales)."""
        pass
    
