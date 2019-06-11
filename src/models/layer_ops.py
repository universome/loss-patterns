import torch
import torch.nn.functional as F

from src.models.module_op import ModuleOperation


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


class ConvOp(ModuleOperation):
    def __init__(self, weights, bias=None, stride:int=1, padding:int=0):
        super(ConvOp, self).__init__()

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
