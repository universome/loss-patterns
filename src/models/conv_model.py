from typing import List
import torch.nn as nn
from firelab.config import Config

from src.model_zoo.layers import Flatten


class ConvModel(nn.Module):
    def __init__(self, config:Config):
        super(ConvModel, self).__init__()

        conv_sizes = config.get('conv_sizes', [1, 8, 32, 64])
        dense_sizes = config.get('dense_sizes', [576, 128])
        use_bn = config.get('use_bn', False)

        self.nn = nn.Sequential(
            *self._create_blocks(self._create_conv_block, conv_sizes, use_bn),
            Flatten(),
            *self._create_blocks(self._create_dense_block, dense_sizes, use_bn),
            nn.Linear(dense_sizes[-1], 10)
        )

    def _create_blocks(self, block_builder, sizes, use_bn=False) -> List[nn.Sequential]:
        return [block_builder(sizes[i], sizes[i+1], use_bn) for i in range(len(sizes) - 1)]

    def _create_conv_block(self, in_size:int, out_size:int, use_bn:bool):
        block = nn.Sequential()
        block.add_module('conv2d', nn.Conv2d(in_size, out_size, 3, padding=1))
        if use_bn:
            block.add_module('bn2d', nn.BatchNorm2d(out_size))
        block.add_module('relu', nn.ReLU(inplace=True))
        block.add_module('maxpool', nn.MaxPool2d(2, 2))

        return block

    def _create_dense_block(self, in_size:int, out_size:int, use_bn:bool):
        block = nn.Sequential()
        block.add_module('linear', nn.Linear(in_size, out_size))
        block.add_module('relu', nn.ReLU(inplace=True))

        return block

    def forward(self, x):
        return self.nn(x)
