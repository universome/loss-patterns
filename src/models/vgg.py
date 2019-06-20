from typing import List

import torch
import torch.nn as nn

from .layer_ops import *
from src.utils import weight_to_param, param_sizes, weight_vector
from src.model_zoo.layers import Flatten


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, head_size=512, init_weights=True):
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, head_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(head_size, head_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(head_size, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG11(VGG):
    def __init__(self, n_input_channels=1, **kwargs):
        features = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        super(VGG11, self).__init__(features, **kwargs)


class VGG11Operation(ModuleOperation):
    param_sizes = param_sizes(VGG11().parameters())

    def __init__(self, weight, dropout_p:float=0.5):
        super(VGG11Operation, self).__init__()

        params = weight_to_param(weight, self.param_sizes)

        self.model = SequentialOp(
            Conv2dOp(params[0], params[1], padding=1),
            BatchNormOp(params[2], params[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dOp(params[4], params[5], padding=1),
            BatchNormOp(params[6], params[7]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dOp(params[8], params[9], padding=1),
            BatchNormOp(params[10], params[11]),
            nn.ReLU(inplace=True),

            Conv2dOp(params[12], params[13], padding=1),
            BatchNormOp(params[14], params[15]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dOp(params[16], params[17], padding=1),
            BatchNormOp(params[18], params[19]),
            nn.ReLU(inplace=True),

            Conv2dOp(params[20], params[21], padding=1),
            BatchNormOp(params[22], params[23]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2dOp(params[24], params[25], padding=1),
            BatchNormOp(params[26], params[27]),
            nn.ReLU(inplace=True),

            Conv2dOp(params[28], params[29], padding=1),
            BatchNormOp(params[30], params[31]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            LinearOp(params[32], params[33]),
            nn.ReLU(True),
            nn.Dropout(),
            LinearOp(params[34], params[35]),
            nn.ReLU(True),
            nn.Dropout(),
            LinearOp(params[36], params[37]),
        )

    def get_modules(self):
        return [self.model]

    def __call__(self, X):
        return self.model(X)


# def gather_vgg_params(VGG_model):
#     weight = weight_vector(VGG_model.parameters())
#     bn_stats = []

#     for l in VGG_model.features.children():
#         if isinstance(l, nn.BatchNorm2d):
#             bn_stats.extend([l.running_mean, l.running_var])

#     return weight, bn_stats
