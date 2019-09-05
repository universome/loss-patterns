from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_zoo.layers import Flatten, Identity, Add
from src.utils import weight_to_param, param_sizes

from src.models.layers import ReparametrizedBatchNorm2d
from src.models.conv_model import ConvBlock


class ModuleOperation:
    def __init__(self):
        self.trainig = True
        self._parameters = {}
        self._modules = {}

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
            param_value = None if param_value is None else param_value.to(*args, **kwargs)
            self.register_param(param_name, param_value)

        for module in self._modules.values():
            module.to(*args, **kwargs)

        return self

    def parameters(self):
        own_params = [p for p in self._parameters.values()]
        module_params = [p for m in self._modules.values() for p in m.parameters()]

        return own_params + module_params

    def state_dict(self):
        result = OrderedDict([(k, v.data.cpu().numpy()) for k, v in self._parameters.items()])

        for module_name, module in self._modules.items():
            result[f'module:{module_name}'] = module.state_dict()

        return result

    def load_state_dict(self, state_dict:OrderedDict):
        for k, v in state_dict.items():
            if k.startswith('module:'):
                module_name = k[7:]
                module = getattr(self, module_name)
                module.load_state_dict(v)
            else:
                self.register_param(k, torch.Tensor(v))

    def register_param(self, param_name, param_value):
        param = None if param_value is None else nn.Parameter(param_value)
        setattr(self, param_name, param)
        self._parameters[param_name] = getattr(self, param_name)

    def register_module(self, name, module):
        setattr(self, name, module)
        self._modules[name] = getattr(self, name)


class SequentialOp(ModuleOperation):
    def __init__(self, *modules):
        super(SequentialOp, self).__init__()

        self.modules = modules

        for i, m in enumerate(self.modules):
            self.register_module(f'module_{i}', m)

    def __call__(self, x):
        for m in self.modules:
            x = m(x)

        return x

    def get_modules(self):
        return list(self._modules.values())


class Conv2dOp(ModuleOperation):
    def __init__(self, weight, bias=None, stride:int=1, padding:int=0, detach:bool=False):
        super(Conv2dOp, self).__init__()

        if detach:
            self.register_param('weight', weight)
            self.register_param('bias', bias)
        else:
            self.weight = weight
            self.bias = bias

        self.stride = stride
        self.padding = padding

    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight, self.bias]

    def __call__(self, X):
        return F.conv2d(X, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)


class LinearOp(ModuleOperation):
    def __init__(self, weight, bias=None, detach:bool=False):
        super(LinearOp, self).__init__()

        if detach:
            self.register_param('weight', weight)
            self.register_param('bias', bias)
        else:
            self.weight = weight
            self.bias = bias

    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight, self.bias]

    def __call__(self, X):
        return F.linear(X, self.weight, self.bias)


class MaxPool2dOp(ModuleOperation):
    def __init__(self, kernel_size:int, stride:int, padding:int, dilation:int):
        super(MaxPool2dOp, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, X):
        return F.max_pool2d(X, self.kernel_size, stride=self.stride, dilation=self.dilation)


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
    def __init__(self, weight, bias, detach:bool=False):
        super(BatchNormOp, self).__init__()

        if detach:
            self.register_param('weight', weight)
            self.register_param('bias', bias)
        else:
            self.weight = weight
            self.bias = bias

    def parameters(self):
        return [self.weight, self.bias]

    def __call__(self, x):
        # We do not keep running mean/var because anyway
        # during interpolation we won't be able to use it
        dummy_mean = torch.zeros_like(self.bias)
        dummy_var = torch.ones_like(self.weight)

        return F.batch_norm(x, dummy_mean, dummy_var,
            weight=self.weight, bias=self.bias, training=True)


class ReparametrizedBatchNorm2dOp(ModuleOperation):
    def __init__(self, weight, bias, running_mean=None, running_var=None, training:bool=True, detach:bool=False):
        super(ReparametrizedBatchNorm2dOp, self).__init__()

        if detach:
            self.register_param('weight', weight)
            self.register_param('bias', bias)
        else:
            self.weight = weight
            self.bias = bias

        # self.running_mean = torch.zeros_like(self.bias) if running_mean is None else running_mean
        # self.running_var = torch.ones_like(self.bias) if running_mean is None else running_mean
        # self.training = training

    def parameters(self):
        return [self.weight, self.bias]

    def __call__(self, x):
        dummy_mean = torch.zeros_like(self.bias)
        dummy_var = torch.ones_like(self.weight)

        return F.batch_norm(x, dummy_mean, dummy_var,
            weight=(self.weight + 1), bias=self.bias, training=True)


class ResidualOp(ModuleOperation):
    def __init__(self, transform):
        self.register_module('transform', transform)

    def __call__(self, x):
        return self.transform(x) + x


class AddOp(ModuleOperation):
    def __init__(self, transform_a, transform_b):
        super(AddOp, self).__init__()

        self.register_module('transform_a', transform_a)
        self.register_module('transform_b', transform_b)

    def __call__(self, x):
        return self.transform_a(x) + self.transform_b(x)


def convert_sequential_model_to_op(weight, dummy_model, detach:bool=False) -> ModuleOperation:
    ops = []
    # TODO: we should keep both weights and buffers...
    params = weight_to_param(weight, param_sizes(dummy_model.parameters()))

    assert isinstance(dummy_model, nn.Sequential), \
        f"Expected model to be nn.Sequential, but got {dummy_model}"

    for module in dummy_model.children():
        if isinstance(module, nn.Linear):
            ops.append(LinearOp(*params[:2], detach=detach))
            params = params[2:]
        elif isinstance(module, nn.Conv2d):
            num_params = 1 if module.bias is None else 2
            ops.append(Conv2dOp(*params[:num_params], padding=module.padding[0], stride=module.stride[0], detach=detach))
            params = params[num_params:]
        elif isinstance(module, ReparametrizedBatchNorm2d):
            ops.append(ReparametrizedBatchNorm2dOp(*params[:2], detach=detach))
            params = params[2:]
        elif isinstance(module, nn.BatchNorm2d):
            ops.append(BatchNormOp(*params[:2], detach=detach))
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
            ops.append(nn.MaxPool2d(module.kernel_size, module.stride, module.padding, module.dilation))
        elif isinstance(module, Flatten):
            ops.append(Flatten())
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            ops.append(nn.AdaptiveAvgPool2d(module.output_size))
        elif isinstance(module, nn.AdaptiveMaxPool2d):
            ops.append(nn.AdaptiveMaxPool2d(module.output_size))
        elif isinstance(module, Identity):
            ops.append(Identity())
        elif isinstance(module, nn.Sequential):
            num_params_in_module:int = len(list(module.parameters()))
            curr_weight = torch.cat([p.view(-1) for p in params[:num_params_in_module]])
            ops.append(convert_sequential_model_to_op(curr_weight, module, detach))
            params = params[num_params_in_module:]
        elif isinstance(module, ConvBlock):
            num_params_in_module:int = len(list(module.parameters()))
            curr_weight = torch.cat([p.view(-1) for p in params[:num_params_in_module]])
            block_op = convert_sequential_model_to_op(curr_weight, module.block, detach)

            if module.is_residual:
                ops.append(ResidualOp(block_op))
            else:
                ops.append(block_op)

            params = params[num_params_in_module:]
        elif isinstance(module, Add):
            num_params_in_transform_a:int = len(list(module.transform_a.parameters()))
            num_params_in_transform_b:int = len(list(module.transform_b.parameters()))
            num_params_in_module = num_params_in_transform_a + num_params_in_transform_b

            params_a = [p.view(-1) for p in params[:num_params_in_transform_a]]
            params_b = [p.view(-1) for p in params[num_params_in_transform_a:num_params_in_module]]

            transform_a_w = torch.cat(params_a) if len(params_a) > 0 else torch.empty(0)
            transform_b_w = torch.cat(params_b) if len(params_b) > 0 else torch.empty(0)

            transform_a_op = convert_sequential_model_to_op(transform_a_w, module.transform_a, detach)
            transform_b_op = convert_sequential_model_to_op(transform_b_w, module.transform_b, detach)

            ops.append(AddOp(transform_a_op, transform_b_op))

            params = params[num_params_in_module:]
        else:
            raise NotImplementedError("Module of type %s is not supported." % type(module))

    assert len(params) == 0, "Not all params was used. Conversion turned out to be wrong."

    return SequentialOp(*ops)
