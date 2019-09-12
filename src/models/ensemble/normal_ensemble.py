from typing import List

import torch

from .ensemble_base import EnsembleBase
from src.utils import weight_vector


class NormalEnsemble(EnsembleBase):
    def __init__(self, torch_model_cls, num_models:int):
        super(NormalEnsemble, self).__init__(torch_model_cls, num_models)

        for i in range(num_models):
            self.register_param(f'model_{i}', weight_vector(torch_model_cls().parameters()))

    def get_model_weights_by_id(self, i:int):
        return getattr(self, f'model_{i}')
