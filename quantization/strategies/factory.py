from .linear import LinearStrategy

from .new_log import LogStrategy

def create_strategy(config):
    if config.method == "linear":
        return LinearStrategy(config)
    elif config.method == "log":
        return LogStrategy(config)
    else:
        raise ValueError(f"Unknown quantization method: {config.method}")