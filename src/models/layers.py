import torch.nn as nn
import torch.nn.functional as F


class ReparametrizedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(ReparametrizedBatchNorm2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        super(ReparametrizedBatchNorm2d, self).reset_parameters()
        nn.init.constant_(self.weight, 0) # Since we are doing +1 there

    def forward(self, x):
        self._check_input_dim(x)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight + 1., self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
