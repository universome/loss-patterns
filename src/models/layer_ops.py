from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_zoo.layers import Flatten, Noop
from src.utils import weight_to_param, param_sizes

from src.models.layers import ReparametrizedBatchNorm2d
from src.models.conv_model import ConvBlock


class ModuleOperation:
    def __init__(self):
        self.trainig = True
        self._parameters = {}

    def train(self, is_enabled:bool=True):
        self.training = is_enabled

        for m in self.get_modules():
            m.train(is_enabled)

        return self

    def eval(self):
        return self.train(False)

    def get_modules(self):
        return []

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        for param_name, param_value in self._parameters.items():
            self.register_param(param_name, param_value.to(*args, **kwargs))

        return self

    def parameters(self):
        return [p for p in self._parameters.values()]

    def state_dict(self):
        return OrderedDict([(k, v.data.cpu().numpy()) for k, v in self._parameters.items()])

    def load_state_dict(self, state_dict:OrderedDict):
        for k, v in state_dict.items():
            self.register_param(k, torch.Tensor(v))

    def register_param(self, param_name, param_value):
        setattr(self, param_name, nn.Parameter(param_value))

        self._parameters[param_name] = getattr(self, param_name)



class SequentialOp(ModuleOperation):
    def __init__(self, *modules):
        super(SequentialOp, self).__init__()

        self.modules = modules

    def __call__(self, X):
        for m in self.modules:
            X = m(X)

        return X

    def get_modules(self):
        return self.modules


class Conv2dOp(ModuleOperation):
    def __init__(self, weights, bias=None, stride:int=1, padding:int=0):
        super(Conv2dOp, self).__init__()

        self.weights = weights
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def __call__(self, X):
        return F.conv2d(X, self.weights, bias=self.bias, stride=self.stride, padding=self.padding)


class LinearOp(ModuleOperation):
    def __init__(self, weights, bias=None):
        super(LinearOp, self).__init__()

        self.weights = weights
        self.bias = bias

    def __call__(self, X):
        return F.linear(X, self.weights, self.bias)


class MaxPool2dOp(ModuleOperation):
    def __init__(self, kernel_size:int, stride:int):
        super(MaxPool2dOp, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, X):
        return F.max_pool2d(X, self.kernel_size, stride=self.stride)


class DropoutOp(ModuleOperation):
    def __init__(self, p:float, is2d:bool=False):
        super(DropoutOp, self).__init__()

        self.p = p
        self.is2d = is2d

    def __call__(self, x):
        if self.training:
            return x
        elif self.is2d:
            return F.dropout2d(x, p=self.p)
        else:
            return F.dropout(x, p=self.p)


class BatchNormOp(ModuleOperation):
    def __init__(self, weight, bias):
        super(BatchNormOp, self).__init__()

        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        # We do not keep running mean/var because anyway
        # during interpolation we won't be able to use it
        dummy_mean = torch.zeros_like(self.bias)
        dummy_var = torch.ones_like(self.weight)

        return F.batch_norm(x, dummy_mean, dummy_var,
            weight=self.weight, bias=self.bias, training=True)


class ReparametrizedBatchNorm2dOp(ModuleOperation):
    def __init__(self, weight, bias):
        super(ReparametrizedBatchNorm2dOp, self).__init__()

        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        # We do not keep running mean/var because anyway
        # during interpolation we won't be able to use it
        dummy_mean = torch.zeros_like(self.bias)
        dummy_var = torch.ones_like(self.weight)

        return F.batch_norm(x, dummy_mean, dummy_var,
            weight=self.weight + 0.5, bias=self.bias, training=True)


class ResidualOp(ModuleOperation):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x) + x


def convert_sequential_model_to_op(weight, dummy_model) -> ModuleOperation:
    ops = []
    params = weight_to_param(weight, param_sizes(dummy_model.parameters()))

    assert isinstance(dummy_model, nn.Sequential), \
        f"Expected model to be nn.Sequential, but got {dummy_model}"

    for module in dummy_model.children():
        if isinstance(module, nn.Linear):
            ops.append(LinearOp(*params[:2]))
            params = params[2:]
        elif isinstance(module, nn.Conv2d):
            ops.append(Conv2dOp(*params[:2], padding=module.padding[0], stride=module.stride[0]))
            params = params[2:]
        elif isinstance(module, nn.BatchNorm2d):
            ops.append(BatchNormOp(*params[:2]))
            params = params[2:]
        elif isinstance(module, ReparametrizedBatchNorm2d):
            ops.append(ReparametrizedBatchNorm2dOp(*params[:2]))
            params = params[2:]
        elif isinstance(module, nn.ReLU):
            ops.append(nn.ReLU(inplace=True))
        elif isinstance(module, nn.SELU):
            ops.append(nn.SELU(inplace=True))
        elif isinstance(module, nn.Tanh):
            ops.append(nn.Tanh())
        elif isinstance(module, nn.Sigmoid):
            ops.append(nn.Sigmoid())
        elif isinstance(module, nn.Dropout):
            ops.append(nn.Dropout(module.p))
        elif isinstance(module, nn.MaxPool2d):
            ops.append(nn.MaxPool2d(module.kernel_size, module.stride))
        elif isinstance(module, Flatten):
            ops.append(Flatten())
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            ops.append(nn.AdaptiveAvgPool2d(module.output_size))
        elif isinstance(module, Noop):
            ops.append(Noop())
        elif isinstance(module, nn.Sequential):
            num_params_in_module:int = len(list(module.parameters()))
            curr_weight = torch.cat([p.view(-1) for p in params[:num_params_in_module]])
            ops.append(convert_sequential_model_to_op(curr_weight, module))
            params = params[num_params_in_module:]
        elif isinstance(module, ConvBlock):
            num_params_in_module:int = len(list(module.parameters()))
            curr_weight = torch.cat([p.view(-1) for p in params[:num_params_in_module]])
            block_op = convert_sequential_model_to_op(curr_weight, module.block)

            if module.is_residual:
                ops.append(ResidualOp(block_op))
            else:
                ops.append(block_op)

            params = params[num_params_in_module:]
        else:
            raise NotImplementedError("Module of type %s is not supported." % type(module))

    assert len(params) == 0, "Not all params was used. Conversion turned out to be wrong."

    return SequentialOp(*ops)
