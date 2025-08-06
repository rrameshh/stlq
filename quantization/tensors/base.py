from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, Callable, Any

class QuantizedTensorBase(ABC):
    """Base class for all quantized tensor implementations."""
    
    @abstractmethod
    def dequantize(self) -> torch.Tensor:
        """Convert quantized tensor back to floating point."""
        pass
    
    @abstractmethod
    def map(self, func: Callable):
        """Apply a function to all tensor components."""
        pass
    
    def reshape(self, *shape):
        """Reshape all tensor components."""
        return self.map(lambda x: x.reshape(*shape))
    
    def permute(self, *permutation):
        """Permute dimensions of all tensor components."""
        return self.map(lambda x: x.permute(*permutation))
    
    @property
    @abstractmethod
    def shape(self):
        """Return the shape of the quantized tensor."""
        pass
    
    @property
    @abstractmethod
    def device(self):
        """Return the device of the quantized tensor."""
        pass
    
    @property
    @abstractmethod
    def dtype(self):
        """Return the data type of the quantized tensor."""
        pass
    
    def to(self, device):
        """Move tensor to device."""
        return self.map(lambda x: x.to(device))
    
    def cuda(self):
        """Move tensor to CUDA."""
        return self.to('cuda')
    
    def cpu(self):
        """Move tensor to CPU."""
        return self.to('cpu')