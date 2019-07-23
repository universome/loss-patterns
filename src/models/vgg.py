from typing import List

import torch
import torch.nn as nn

from src.utils import weight_to_param, param_sizes, weight_vector
from src.model_zoo.layers import Flatten, Noop
from .layers import ReparametrizedBatchNorm2d


class VGG(nn.Module):
    def __init__(self, conv_body, num_classes:int=10, head_size:int=512, init_weights:bool=True):
        super(VGG, self).__init__()

        self.model = nn.Sequential(
            conv_body,
            nn.AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            nn.Linear(512 * 7 * 7, head_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(head_size, head_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(head_size, num_classes),
        )

        if False:
            self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ReparametrizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG11(VGG):
    def __init__(self, n_input_channels=1, use_bn=True, **kwargs):
        conv_body = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(64) if use_bn else Noop(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(128) if use_bn else Noop(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(256) if use_bn else Noop(),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(256) if use_bn else Noop(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(512) if use_bn else Noop(),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(512) if use_bn else Noop(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(512) if use_bn else Noop(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(512) if use_bn else Noop(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        super(VGG11, self).__init__(conv_body, **kwargs)
