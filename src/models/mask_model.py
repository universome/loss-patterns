import random
from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from src.models.layer_ops import ModuleOperation, convert_sequential_model_to_op
from src.utils import weight_vector, orthogonalize


class MaskModel(ModuleOperation):
    def __init__(self, mask:List[List[int]], torch_model_cls,
                       scaling:float=1., should_center_origin:bool=False, parametrization_type:str="simple"):
        """
        @params
            - should_center_origin: specifies, if our origin should be in the center of the mask
        """

        self.mask = mask
        self.dummy_model = torch_model_cls()
        self.scaling = scaling
        self.is_good_mode = True
        self.should_center_origin = should_center_origin
        self.parametrization_type = parametrization_type

        self.origin_param = nn.Parameter(weight_vector(torch_model_cls().parameters()))
        self.right_param = nn.Parameter(weight_vector(torch_model_cls().parameters()))
        self.up_param = nn.Parameter(weight_vector(torch_model_cls().parameters()))

        # assert torch.dot(self.right_param, self.up_param) < 5, \
        #     f"Dot product is too high ({torch.dot(self.right_param, self.up_param)}). Looks suspicious."

    @property
    def origin(self):
        return self.origin_param

    @property
    def right(self):
        if self.parametrization_type == 'simple':
            return self.right_param
        elif self.parametrization_type == 'difference':
            return self.right_param - self.origin_param
        elif self.parametrization_type == 'up_orthogonal':
            return self.right_param - self.origin
        else:
            raise NotImplementedError(f'Unknown parametrization type {self.parametrization_type}')

    @property
    def up(self):
        if self.parametrization_type == 'simple':
            return self.up_param
        elif self.parametrization_type == 'difference':
            return self.up_param - self.origin_param
        elif self.parametrization_type == 'up_orthogonal':
            return orthogonalize(self.right, self.up_param, adjust_len_to_v1=True)
        else:
            raise NotImplementedError(f'Unknown parametrization type {self.parametrization_type}')

    def __call__(self, x):
        if self.is_good_mode:
            w = self.sample_class_weight(1)
        else:
            w = self.sample_class_weight(-1)

        return self.run_from_weights(w, x)

    def run_from_weights(self, w, x):
        model_op = convert_sequential_model_to_op(w, self.dummy_model)

        return model_op(x)

    def sample_idx(self) -> Tuple[int, int]:
        if np.random.rand() > 0.5:
            return self.sample_class_idx(-1)
        else:
            return self.sample_class_idx(1)

    def sample_class_idx(self, cls_idx:int) -> Tuple[int, int]:
        return random.choice(self.get_class_idx(cls_idx))

    def get_class_idx(self, cls_idx:int) -> List[Tuple[int, int]]:
        idx = np.indices(self.mask.shape).transpose(1,2,0)
        pos_idx = idx[np.where(self.mask == cls_idx)]

        return pos_idx

    def sample_class_weight(self, cls_idx:int):
        return self.compute_point(*self.sample_class_idx(cls_idx))

    def compute_point(self, x, y, should_orthogonalize:bool=False):
        up = self.up
        right = self.right

        if should_orthogonalize:
            up = orthogonalize(right, up, adjust_len_to_v1=True)

        if self.should_center_origin:
            x -= (self.mask.shape[0] // 2)
            y -= (self.mask.shape[1] // 2)

        return self.origin + self.scaling * (x * up + y * right)

    def compute_ort_reg(self):
        return torch.dot(self.up, self.right).abs()

    def compute_norm_reg(self):
        return (self.up.norm() - self.right.norm()).abs()

    def to(self, *args, **kwargs):
        self.origin_param = nn.Parameter(self.origin_param.to(*args, **kwargs))
        self.right_param = nn.Parameter(self.right_param.to(*args, **kwargs))
        self.up_param = nn.Parameter(self.up_param.to(*args, **kwargs))

        return self

    def parameters(self):
        return [self.origin_param, self.right_param, self.up_param]

    def state_dict(self):
        return OrderedDict([
            ('origin', self.origin_param.cpu().numpy()),
            ('right_param', self.right_param.cpu().numpy()),
            ('up_param', self.up_param.cpu().numpy()),
        ])
