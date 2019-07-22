from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_zoo.layers import Flatten, Noop
from src.utils import weight_to_param, param_sizes


class ModuleOperation:
    def __init__(self):
        self.trainig = True

    def train(self, is_enabled:bool=True):
        self.training = is_enabled

        for m in self.get_modules():
            m.train(is_enabled)

        return self

    def eval(self):
        return self.train(False)

    def get_modules(self):
        return []

    # def to(self, *args, **kwargs):
    #     for name, p in self.named_params:
    #         setattr(self, name, p.to(*args, **kwargs))



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


class ReparametrizedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(ReparametrizedBatchNorm2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        super(ReparametrizedBatchNorm2d, self).reset_parameters()
        self.weight.data.add_(-0.5) # So it has zero mean

    def forward(self, x):
        self._check_input_dim(x)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight + 0.5, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


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


def convert_sequential_model_to_op(weight, dummy_model) -> ModuleOperation:
    ops = []
    params = weight_to_param(weight, param_sizes(dummy_model.parameters()))

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
        else:
            raise NotImplementedError("Module of type %s is not supported." % type(module))

    assert len(params) == 0, "Not all params was used. Conversion turned out to be wrong."

    return SequentialOp(*ops)
