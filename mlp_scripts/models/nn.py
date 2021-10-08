import torch.nn as nn


class MLP(nn.Sequential):

    def __init__(self, channels):
        super().__init__()
        features = 311
        modules = []
        for layer in channels:
            modules.append(nn.Linear(features, layer))
            modules.append(nn.BatchNorm1d(layer))
            modules.append(nn.Sigmoid())
            features = layer
        modules.append(nn.Linear(features, 1))
        modules.append(nn.Sigmoid())
        super().__init__(*modules)


__all__ = ['MLP']
