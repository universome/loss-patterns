from typing import Tuple

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class TriangleLR(LambdaLR):
    def __init__(self, optimizer, epoch_size:int, knots:Tuple[int, int, int], values:Tuple[float, float, float], *args, **kwargs):
        assert len(knots) == len(values) == 3, f"Wrong lengths: {len(knots)} {len(values)}"

        self.epoch_size = epoch_size
        self.knots = knots
        self.values = values

        super(TriangleLR, self).__init__(optimizer, self.triangle_scheduler, *args, **kwargs)

    def triangle_scheduler(self, it:int):
        step = it / self.epoch_size

        if step > self.knots[-1]:
            return self.values[-1]
        else:
            return np.interp([step], self.knots, self.values)[0]
