# sparsify/__init__.py
from .activation_aware import ActivationAwareSparsification

def create_sparsifier(config):
    if not hasattr(config, 'sparsification') or not config.sparsification.enabled:
        return None
    method = config.sparsification.method.lower()
    if method == "activation_aware":
        return ActivationAwareSparsification()
    else:
        raise ValueError(f"Unknown sparsification method: {method}")
__all__ = ['ActivationAwareSparsification', 'create_sparsifier']