from typing import List

import torch

from src.models.layer_ops import ModuleOperation, convert_sequential_model_to_op
from src.utils import weight_vector


class EnsembleModel(ModuleOperation):
    COORDS_INIT_STRATEGIES = set([
        'isotropic_normal',
        'standard_uniform',
    ])

    def __init__(self, torch_model_cls, num_models:int, coords_init_strategy:str='isotropic_normal'):
        super(EnsembleModel, self).__init__()

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

    def get_model_by_id(self, i:int):
        w = self.get_model_weights_by_id(i)
        model_op = convert_sequential_model_to_op(w, self.dummy_model)

        return model_op

    def get_model_weights_by_id(self, i:int):
        x, y = self.coords[i]
        w = self.origin_param + x * self.right_param + y * self.up_param

        return w

    def run_model_by_id(self, i, x):
        return self.get_model_by_id(i)(x)

    def run_ensemble(self, x, weights:List[float]=None):
        num_models = self.coords.size(0)
        preds = [self.run_model_by_id(i, x) for i in range(num_models)]
        preds = torch.stack(preds)

        if not weights is None:
            assert weights.size() == (num_models, )
            assert weights.sum() == 1
            assert torch.all(weights >= 0)

            preds = preds * weights.view(-1, 1, 1)
        else:
            # Uniform distribution over the models
            preds = preds / num_models

        return preds.sum(dim=0)

    def forward(self, x):
        return self.run_ensemble(x)
