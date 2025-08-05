# data/__init__.py - Single entry point for all data loading
from .vision import get_cifar10_dataloaders, get_imagenet100_dataloaders
from .language import get_shakespeare_dataloaders, get_imdb_dataloaders, get_sst2_dataloaders

DATA_LOADERS = {
    'cifar10': get_cifar10_dataloaders,
    'imagenet100': get_imagenet100_dataloaders,
    'shakespeare': get_shakespeare_dataloaders,
    'imdb': get_imdb_dataloaders,
    'sst2': get_sst2_dataloaders,
}

def get_data_loaders(config):
    """Single function to get any data loader."""
    dataset = config.data.dataset
    
    if dataset not in DATA_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    loader_func = DATA_LOADERS[dataset]
    
    # Common arguments
    kwargs = {
        'batch_size': config.data.batch_size,
        'num_workers': config.data.num_workers,
    }
    
    # Add dataset-specific arguments
    if dataset in ['shakespeare']:
        kwargs.update({
            'seq_len': config.data.seq_len,
            'char_level': config.model.char_level,
        })
    elif dataset in ['imdb', 'sst2']:
        kwargs.update({
            'max_length': config.data.seq_len,
        })
    
    result = loader_func(**kwargs)
    
    # Handle different return formats
    if len(result) == 3:  # train_loader, val_loader, extra_info
        train_loader, val_loader, extra = result
        # Set vocab_size or num_classes in config
        if dataset in ['shakespeare']:
            config.model.vocab_size = extra
        elif dataset in ['imdb', 'sst2']:
            config.model.num_classes = extra
        return train_loader, val_loader
    else:
        # Set num_classes for vision datasets
        if dataset == 'cifar10':
            config.model.num_classes = 10
        elif dataset == 'imagenet100':
            config.model.num_classes = 100
        return result
    

    