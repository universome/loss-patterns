from typing import List

import torch

from src.models.layer_ops import convert_sequential_model_to_op
from src.utils import weight_vector

from .ensemble_base import EnsembleBase


class PlaneEnsemble(EnsembleBase):
    COORDS_INIT_STRATEGIES = set([
        'isotropic_normal',
        'standard_uniform',
    ])

    def __init__(self, torch_model_cls, num_models:int, coords_init_strategy:str='isotropic_normal'):
        super(PlaneEnsemble, self).__init__(torch_model_cls, num_models)

        assert coords_init_strategy in self.COORDS_INIT_STRATEGIES, \
            f"Unknown init strategy: {coords_init_strategy}"

        self.dummy_model = torch_model_cls()
        self.register_param('coords', torch.stack([self.sample_coords(coords_init_strategy) for _ in range(num_models)]))
        self.register_param('origin_param', weight_vector(torch_model_cls().parameters()))
        self.register_param('right_param', weight_vector(torch_model_cls().parameters()))
        self.register_param('up_param', weight_vector(torch_model_cls().parameters()))

    def sample_coords(self, init_strategy:str):
        if init_strategy == 'isotropic_normal':
            return torch.randn(2) / 10
        elif init_strategy == 'standard_uniform':
            return torch.rand(2) / 10
        else:
            raise NotImplementedError

    def get_model_weights_by_id(self, i:int):
        x, y = self.coords[i]
        w = self.origin_param + x * self.right_param + y * self.up_param

        return w
