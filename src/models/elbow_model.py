import numpy as np
import torch.nn as nn

from .layer_ops import ModuleOperation
from .simple_model import SimpleModel, SimpleModelOperation
from src.utils import param_sizes, weight_vector


class ElbowModel(ModuleOperation):
    def __init__(self, w_1=None, w_2=None, w_3=None):
        self.w_1 = w_1 if not w_1 is None else weight_vector(SimpleModel().parameters())
        self.w_2 = w_2 if not w_2 is None else weight_vector(SimpleModel().parameters())
        self.w_3 = nn.Parameter(w_3 if not w_3 is None else weight_vector(SimpleModel().parameters()))

    def sample(self):
        alpha = np.random.random()
        beta = np.random.random()

        # Randomly choosing a link
        min_to_use = self.w_1 if beta > 0.5 else self.w_2

        # Randomly choosing a point on a link
        w = min_to_use * (1 - alpha) + self.w_3 * alpha

        return w

    def run_from_weights(self, w, x):
        model = SimpleModelOperation(w).train(self.training)

        return model(x)

    def __call__(self, x):
        if self.training:
            w = self.sample()
        else:
            w = self.w_3

        return self.run_from_weights(w, x)

    def to(self, *args, **kwargs):
        self.w_1 = self.w_1.to(*args, **kwargs)
        self.w_2 = self.w_2.to(*args, **kwargs)
        self.w_3 = nn.Parameter(self.w_3.to(*args, **kwargs))

        return self

    def parameters(self):
        return [self.w_3]
