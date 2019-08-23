from typing import List

import torch

from src.models.layer_ops import ModuleOperation, convert_sequential_model_to_op
from src.utils import weight_vector


class EnsembleBase(ModuleOperation):
    def __init__(self, torch_model_cls, num_models:int):
        super(EnsembleBase, self).__init__()

        self.coords = [None] * num_models
        self.dummy_model = torch_model_cls()


    def get_model_weights_by_id(self, i:int):
        raise NotImplementedError

    def get_model_by_id(self, i:int):
        w = self.get_model_weights_by_id(i)
        model_op = convert_sequential_model_to_op(w, self.dummy_model)

        return model_op

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
