import random
from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from src.models.module_op import ModuleOperation
from src.utils import weight_vector


class MaskModel(ModuleOperation):
    def __init__(self, mask:List[List[int]], torch_model_cls, model_op_cls, scaling:float=1.):
        self.mask = mask
        self.torch_model_cls = torch_model_cls
        self.model_op_cls = model_op_cls
        self.lower_left = nn.Parameter(weight_vector(self.torch_model_cls().parameters()))
        self.upper_left = nn.Parameter(weight_vector(self.torch_model_cls().parameters()))
        self.lower_right = nn.Parameter(weight_vector(self.torch_model_cls().parameters()))
        self.scaling = scaling
        self.is_good_mode = True

    def __call__(self, x):
        #if self.training:
        #    w = self.cell_center(*self.sample_idx())
        #else:
        #    w = self.cell_center(2, 4)
        if self.is_good_mode:
            w = self.sample_class_weight(1)
        else:
            w = self.sample_class_weight(0)

        return self.run_from_weights(w, x)

    def run_from_weights(self, w, x):
        return self.model_op_cls(w)(x)

    def sample_idx(self) -> Tuple[int, int]:
        # i = random.randint(0, self.mask.shape[0] - 1)
        # j = random.randint(0, self.mask.shape[1] - 1)
        if np.random.rand() > 0.5:
            return self.sample_class_idx(0)
        else:
            return self.sample_class_idx(1)

    def sample_class_idx(self, cls_idx:int) -> Tuple[int, int]:
        return random.choice(self.get_class_idx(cls_idx))

    def get_class_idx(self, cls_idx:int) -> List[Tuple[int, int]]:
        idx = np.indices(self.mask.shape).transpose(1,2,0)
        pos_idx = idx[np.where(self.mask == cls_idx)]

        return pos_idx

    def sample_class_weight(self, cls_idx:int):
        return self.cell_center(*self.sample_class_idx(cls_idx))

    def cell_center(self, i, j):
        #return self.lower_left + (i / self.mask.shape[0]) * self.upper_left + (j / self.mask.shape[1]) * self.lower_right
        return self.lower_left + self.scaling * (i * self.upper_left + j * self.lower_right)

    def compute_reg(self):
        orthogonalization_reg = torch.dot(self.lower_right, self.upper_left).pow(2)
        norm_reg = (self.upper_left.norm() - self.lower_right.norm()).pow(2)

        return orthogonalization_reg, norm_reg

    def to(self, *args, **kwargs):
        self.lower_left = nn.Parameter(self.lower_left.to(*args, **kwargs))
        self.upper_left = nn.Parameter(self.upper_left.to(*args, **kwargs))
        self.lower_right = nn.Parameter(self.lower_right.to(*args, **kwargs))

        return self

    def parameters(self):
        return [self.lower_left, self.upper_left, self.lower_right]

    def state_dict(self):
        return OrderedDict([
            ('lower_left', self.lower_left.cpu().numpy()),
            ('upper_left', self.upper_left.cpu().numpy()),
            ('lower_right', self.lower_right.cpu().numpy()),
        ])
