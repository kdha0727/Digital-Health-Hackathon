import torch
import torch.utils.data

from sklearn.model_selection import KFold


def get_loader(dataset, train=True, sampler=None, batch_size=10, drop_last=False):
    """Loader Magic"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=2,
        drop_last=drop_last,
    )


def get_kfold_loaders(dataset, n_splits=5, shuffle=False, random_state=None, batch_size=None):
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    loader_options = {}
    if batch_size is not None:
        loader_options.update(batch_size)
    for train_idx, val_idx in kfold.split(dataset):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = get_loader(dataset, sampler=train_sampler, **loader_options)
        val_loader = get_loader(dataset, sampler=val_sampler, **loader_options)
        yield train_loader, val_loader


__all__ = ['get_loader', 'get_kfold_loaders']
