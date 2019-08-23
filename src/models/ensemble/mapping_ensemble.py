from typing import List

import torch
import torch.nn as nn
from firelab.config import Config

from src.models.layer_ops import ModuleOperation

from .ensemble_base import EnsembleBase

class MappingEnsemble(EnsembleBase):
    def __init__(self, torch_model_cls, num_models:int, mapping_config:Config):
        super(MappingEnsemble, self).__init__(torch_model_cls, num_models)

        self.dummy_model = torch_model_cls()
        self.register_param('coords', torch.stack([
            torch.randn(mapping_config.dense_sizes[0]) for _ in range(num_models)]))
        self.register_module('mapping', self.init_mapping(mapping_config))

    def init_mapping(self, config):
        def dense_layer(i:int):
            return nn.Linear(config.dense_sizes[i-1], config.dense_sizes[i])

        def activation():
            if config.activation.lower() == 'relu':
                return nn.ReLU()
            elif config.activation.lower() == 'tanh':
                return nn.Tanh()
            else:
                raise NotImplementedError

        output_size = sum(p.numel() for p in self.dummy_model.parameters())
        layers = [(dense_layer(i), activation()) for i in range(1, len(config.dense_sizes))]
        layers = [l for pair in layers for l in pair]
        layers.append(nn.Linear(config.dense_sizes[-1], output_size)) # Last layer goes without activation

        return nn.Sequential(*layers)

    def get_model_weights_by_id(self, i:int):
        return self.mapping(self.coords[i])
