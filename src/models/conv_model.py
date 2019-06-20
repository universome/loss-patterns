from typing import List
import torch.nn as nn

from src.model_zoo.layers import Flatten

class ConvModel(nn.Module):
    def __init__(self, conv_sizes=[1, 8, 32], dense_sizes=[800, 128]):
        super(ConvModel, self).__init__()

        self.nn = nn.Sequential(
            *self._create_blocks(self._create_conv_block, conv_sizes),
            Flatten(),
            *self._create_blocks(self._create_dense_block, dense_sizes),
            nn.Linear(dense_sizes[-1], 10)
        )

    def _create_blocks(self, block_builder, sizes) -> List[nn.Sequential]:
        return [block_builder(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]

    def _create_conv_block(self, in_size:int, out_size:int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def _create_dense_block(self, in_size:int, out_size:int):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.nn(x)
