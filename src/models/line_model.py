import torch.nn as nn
import numpy as np

from .module_op import ModuleOperation
from .simple_model import SimpleModel, SimpleModelOperation
from src.utils import weight_vector, weight_to_param, param_sizes


class LineModel(ModuleOperation):
    def __init__(self):
        super(LineModel, self).__init__()

        self.w_1 = nn.Parameter(weight_vector(SimpleModel().parameters()))
        self.w_2 = nn.Parameter(weight_vector(SimpleModel().parameters()))
        self.param_sizes = param_sizes(SimpleModel().parameters())

    def get_distance(self):
        return (self.w_1 - self.w_2).norm()

    def __call__(self, x):
        if self.training:
            alpha = np.random.rand()
            w = (1 - alpha) * self.w_1 + alpha * self.w_2
        else:
            w = 0.5 * self.w_1 + 0.5 * self.w_2

        # params = weight_to_param(w, self.param_sizes)
        model = SimpleModelOperation(w)

        if not self.training:
            model = model.eval()

        return model(x)

    def weights_distance(self):
        return (self.w_1 - self.w_2).norm()

    def parameters(self):
        return [self.w_1, self.w_2]

    def to(self, *args, **kwargs):
        self.w_1 = nn.Parameter(self.w_1.to(*args, **kwargs))
        self.w_2 = nn.Parameter(self.w_2.to(*args, **kwargs))

        return self
