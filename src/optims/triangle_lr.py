from typing import Tuple

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class TriangleLR(LambdaLR):
    def __init__(self, optimizer, epoch_size:int, knots:Tuple[int, int, int], values:Tuple[float, float, float], *args, **kwargs):
        assert len(knots) == len(values) == 3, f"Wrong lengths: {len(knots)} {len(values)}"

        def triangle_scheduler(it:int):
            step = it / epoch_size

            if step > knots[-1]:
                return values[-1]
            else:
                return np.interp([step], knots, values)[0]

        super(TriangleLR, self).__init__(optimizer, triangle_scheduler, *args, **kwargs)
