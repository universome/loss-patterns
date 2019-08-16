from typing import List
import torch.nn as nn
from firelab.config import Config

from src.model_zoo.layers import Flatten, Noop
from src.models.layers import ReparametrizedBatchNorm2d


class ConvModel(nn.Module):
    def __init__(self, config:Config):
        super(ConvModel, self).__init__()

        conv_sizes = config.get('conv_sizes', [1, 8, 32, 64])
        dense_sizes = config.get('dense_sizes', [576, 128])
        use_bn = config.get('use_bn', False)
        use_dropout = config.get('use_dropout', False)
        use_maxpool = config.get('use_maxpool', False)
        use_skip_connection = config.get('use_skip_connection', False)
        activation = config.get('activation', 'relu')

        if activation == 'relu':
            self.activation = lambda: nn.ReLU(inplace=True)
        elif activation == 'selu':
            self.activation = lambda: nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = lambda: nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = lambda: nn.Sigmoid()
        else:
            raise NotImplementedError(f'Unknown activation function: {activation}')

        conv_body = nn.Sequential(*[
            ConvBlock(conv_sizes[i], conv_sizes[i+1], use_bn, use_skip_connection, \
                      use_maxpool, self.activation) for i in range(len(conv_sizes) - 1)])
        dense_head = nn.Sequential(*[
            self._create_dense_block(dense_sizes[i], dense_sizes[i+1], use_dropout) for i in range(len(dense_sizes) - 1)
        ])

        self.nn = nn.Sequential(
            conv_body,
            nn.AdaptiveAvgPool2d((4, 4)),
            Flatten(),
            dense_head,
            nn.Linear(dense_sizes[-1], 10)
        )

    def _create_dense_block(self, in_size:int, out_size:int, use_dropout:bool):
        block = nn.Sequential()
        block.add_module('linear', nn.Linear(in_size, out_size))
        block.add_module('activation', self.activation())

        if use_dropout:
            block.add_module('dropout', nn.Dropout())

        return block

    def forward(self, x):
        return self.nn(x)


class ConvBlock(nn.Module):
    def __init__(self, in_size:int,
                       out_size:int,
                       use_bn:bool,
                       use_skip_connection:bool,
                       use_maxpool:bool,
                       activation:nn.Module):
        super(ConvBlock, self).__init__()

        self.is_residual = use_skip_connection and (in_size == out_size)

        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            ReparametrizedBatchNorm2d(out_size) if use_bn else Noop(),
            activation(),
            Noop() if (self.is_residual or not use_maxpool) else nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        out = self.block(x)

        if self.is_residual:
            out = out + x

        return out
