import torch
import torch.nn as nn

from src.models.layer_ops import *
from src.model_zoo.layers import Flatten
from src.utils import weight_to_param, param_sizes


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.nn = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            Flatten(),
            nn.Linear(800, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.nn(x)

    def get_activations(self, x):
        activations = [x]

        with torch.no_grad():
            for m in self.nn.children():
                activations.append(m(activations[-1]))

        return activations


class SimpleModelOperation(ModuleOperation):
    def __init__(self, weight, dropout_p:float=0.):
        super(SimpleModelOperation, self).__init__()

        params = weight_to_param(weight, param_sizes(SimpleModel().parameters()))

        self.model = SequentialOp(
            Conv2dOp(params[0], params[1]),
            nn.ReLU(inplace=True),
            MaxPool2dOp(2, 2),
            #DropoutOp(dropout_p, is2d=True),

            Conv2dOp(params[2], params[3]),
            nn.ReLU(inplace=True),
            MaxPool2dOp(2, 2),
            #DropoutOp(dropout_p, is2d=True),

            Flatten(),
            LinearOp(params[4], params[5]),
            nn.ReLU(inplace=True),
            #DropoutOp(dropout_p),
            LinearOp(params[6], params[7]),
        )

    def get_modules(self):
        return [self.model]

    def __call__(self, X):
        return self.model(X)
