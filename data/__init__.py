# data/__init__.py - Fixed imports
from .vision_datasets import get_cifar10_dataloaders, get_imagenet100_dataloaders
from .text_datasets import get_shakespeare_dataloaders, get_imdb_dataloaders, get_sst2_dataloaders

DATA_LOADERS = {
    'cifar10': get_cifar10_dataloaders,
    'imagenet100': get_imagenet100_dataloaders,
    'shakespeare': get_shakespeare_dataloaders,
    'imdb': get_imdb_dataloaders,
    'sst2': get_sst2_dataloaders,
}

def get_data_loaders(config):
    dataset = config.data.dataset
    
    if dataset not in DATA_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    loader_func = DATA_LOADERS[dataset]
    
    # Common arguments
    kwargs = {
        'batch_size': config.training.batch_size,
        'num_workers': config.data.num_workers,
    }
    

    if dataset in ['shakespeare']:
        kwargs.update({
            'seq_len': config.data.seq_len,
            'char_level': getattr(config.model, 'char_level', True),
        })
    elif dataset in ['imdb', 'sst2']:
        kwargs.update({
            'max_length': config.data.seq_len,
        })
    
    result = loader_func(**kwargs)
    

    if len(result) == 3:  # train_loader, val_loader, extra_info
        train_loader, val_loader, extra = result
        if dataset in ['shakespeare']:
            config.model.vocab_size = extra
        elif dataset in ['imdb', 'sst2']:
            config.model.num_classes = extra
        return train_loader, val_loader
    else:
        if dataset == 'cifar10':
            config.model.num_classes = 10
        elif dataset == 'imagenet100':
            config.model.num_classes = 100
        return result