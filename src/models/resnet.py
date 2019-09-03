import torch
import torch.nn as nn

from src.model_zoo.layers import Flatten, Noop, Add
from .layer_ops import ReparametrizedBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def make_residual_block(in_planes, out_planes, stride=1, downsample=None):
    downsample = nn.Sequential(Noop()) if downsample is None else downsample

    return nn.Sequential(
        Add(
            nn.Sequential(
                conv3x3(in_planes, out_planes, stride),
                ReparametrizedBatchNorm2d(out_planes),
                nn.ReLU(inplace=True),

                conv3x3(out_planes, out_planes),
                ReparametrizedBatchNorm2d(out_planes),
            ),
            downsample
        ),
        nn.ReLU(inplace=True)
    )


def make_simple_block(channels_in:int, channels_out:int):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
        ReparametrizedBatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


class ResNet18(nn.Module):
    def __init__(self, n_input_channels=3, n_classes=1000):
        super(ResNet18, self).__init__()
        self._construct_model(n_input_channels, n_classes)
        self._init_weights()

    def _construct_model(self, n_input_channels, n_classes):
        self.nn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            ReparametrizedBatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            self._make_layer(64, 64),
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, n_classes)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, ReparametrizedBatchNorm2d):
                nn.init.constant_(m.weight, 0) # set to 0 because we use +1 in it
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, stride=1):
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                ReparametrizedBatchNorm2d(out_planes),
            )
        else:
            downsample = None

        return nn.Sequential(
            make_residual_block(in_planes, out_planes, stride, downsample),
            make_residual_block(out_planes, out_planes)
        )

    def forward(self, x):
        return self.nn(x)


class FastResNet(ResNet18):
    def _construct_model(self, n_input_channels, n_classes):
        self.nn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1, bias=False),
            # ReparametrizedBatchNorm2d(64),
            # nn.ReLU(inplace=True),

            # make_simple_block(64, 128),
            # make_residual_block(128, 128),
            # make_simple_block(128, 256),
            # make_simple_block(256, 512),
            # make_residual_block(512, 512),

            # # nn.AdaptiveMaxPool2d((1, 1)),
            # nn.MaxPool2d(4, 4),
            # Flatten(),
            # nn.Linear(512, n_classes)

            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1, bias=False),
            ReparametrizedBatchNorm2d(64),
            nn.ReLU(inplace=True),

            make_simple_block(64, 128),

            make_residual_block(128, 128),
            make_residual_block(128, 128),

            make_simple_block(128, 256),

            make_residual_block(256, 256),
            make_residual_block(256, 256),

            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, n_classes)
        )
