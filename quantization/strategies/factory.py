from .linear import LinearStrategy
from .adaptive_log import AdaptiveLogStrategy
from .new_log import LogStrategy

def create_strategy(config):
    if config.method == "linear":
        return LinearStrategy(config)
    elif config.method == "log":
        return LogStrategy(config)
    elif config.method == "adaptive_log":
        return AdaptiveLogStrategy(config)
    else:
        raise ValueError(f"Unknown quantization method: {config.method}")
