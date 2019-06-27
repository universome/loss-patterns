from typing import List
import torch.nn as nn
from firelab.config import Config

from src.model_zoo.layers import Flatten
from src.models.layer_ops import ReparametrizedBatchNorm2d


class ConvModel(nn.Module):
    def __init__(self, config:Config):
        super(ConvModel, self).__init__()

        conv_sizes = config.get('conv_sizes', [1, 8, 32, 64])
        dense_sizes = config.get('dense_sizes', [576, 128])
        use_bn = config.get('use_bn', False)
        use_dropout = config.get('use_dropout', False)
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

        self.nn = nn.Sequential(
            *self._create_blocks(self._create_conv_block, conv_sizes, use_bn, use_dropout),
            Flatten(),
            *self._create_blocks(self._create_dense_block, dense_sizes, use_bn, use_dropout),
            nn.Linear(dense_sizes[-1], 10)
        )

    def _create_blocks(self, block_builder, sizes, use_bn:bool, use_dropout:bool) -> List[nn.Sequential]:
        return [block_builder(sizes[i], sizes[i+1], use_bn, use_dropout) for i in range(len(sizes) - 1)]

    def _create_conv_block(self, in_size:int, out_size:int, use_bn:bool, use_dropout:bool):
        block = nn.Sequential()
        block.add_module('conv2d', nn.Conv2d(in_size, out_size, 3, padding=1))
        if use_bn:
            block.add_module('bn2d', ReparametrizedBatchNorm2d(out_size)) # TODO: track_running_stats=False?
        block.add_module('activation', self.activation())
        block.add_module('maxpool', nn.MaxPool2d(2, 2))

        return block

    def _create_dense_block(self, in_size:int, out_size:int, use_bn:bool, use_dropout:bool):
        block = nn.Sequential()
        block.add_module('linear', nn.Linear(in_size, out_size))
        block.add_module('activation', self.activation())

        if use_dropout:
            block.add_module('dropout', nn.Dropout())

        return block

    def forward(self, x):
        return self.nn(x)
